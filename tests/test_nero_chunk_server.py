"""The HTTP boundary must preserve the audited engine's observations and output."""

import base64
import copy
import io

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.deploy.nero_chunk_server import CONTRACT, SERVER_ROLES, WIRE, NeroChunkService, validate_schema
from openwam.deploy.nero_contract import conditions_from_observation
from tests.dataloader.test_nero_multitask import bucket as bucket


def schema():
    names = ["left_arm", "right_arm"]
    return {
        "version": "velabot.io/v2",
        "action_dim": 20,
        "include_base": False,
        "layout": [
            x for n in names for x in [*[f"{n}.arm.{i}" for i in range(9)], f"{n}.effector.position.opening_width"]
        ],
        "groups": [
            dict(
                group=n,
                arm_format="rot6d",
                arm_dim=9,
                effector_dim=1,
                effector_protocol="agilex.parallel_gripper.width_m/v1",
                effector_names=["opening_width"],
                effector_position_unit="width_m",
                effector_position_limits=[0.0, 0.1],
                effector_position_velocity_max=[],
            )
            for n in names
        ],
    }


def setup_service(bucket):
    _, dl, eefs, _ = bucket
    cfg = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    state = eefs[0][0].copy()
    outputs = np.tile(state, (32, 1))
    outputs[:, 0] += 0.01
    outputs[0, 9] = -0.001  # Preserve raw absolute-model overshoot, not a hidden new clamp.

    class Engine:
        calls = 0

        def generate(self, conditions):
            self.calls += 1
            self.conditions = conditions
            return {"actions": outputs.copy()}

    engine = Engine()
    service = NeroChunkService(engine, cfg, {"checkpoint": "fake", "model_representation": "eef"})
    reply = service.request("/handshake", dict(wire_format=WIRE, contract=CONTRACT, io_schema=schema()))
    rng = np.random.default_rng(14)
    images = {role: Image.fromarray(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)) for role in SERVER_ROLES}
    encoded = {}
    for role, im in images.items():
        b = io.BytesIO()
        im.save(b, format="PNG")
        encoded[role] = base64.b64encode(b.getvalue()).decode()
    request = dict(
        wire_format=WIRE,
        session_id=reply["session_id"],
        request_id="one",
        state=state.tolist(),
        images=encoded,
        prompt="task\nverbatim",
    )
    return service, request, images, outputs


def test_one_generation_and_exact_observation_and_output(bucket):
    service, request, images, outputs = setup_service(bucket)
    reply = service.request("/infer", request)
    direct = conditions_from_observation(
        service.preprocessor, dict(images=images, state=request["state"], prompt=request["prompt"]), 42
    )
    assert service.engine.calls == 1
    for key in direct:
        if key == "first_frame_image":
            np.testing.assert_array_equal(np.asarray(direct[key][0]), np.asarray(service.engine.conditions[key][0]))
        elif key == "proprio":
            np.testing.assert_array_equal(direct[key], service.engine.conditions[key])
        else:
            assert direct[key] == service.engine.conditions[key]
    np.testing.assert_array_equal(reply["actions"], outputs)
    np.testing.assert_array_equal(reply["anchor_state"], request["state"])


def test_reset_invalidates_old_session(bucket):
    service, request, _, _ = setup_service(bucket)
    reply = service.request("/reset", request)
    assert reply["session_id"] != request["session_id"]
    with pytest.raises(ValueError, match="handshake"):
        service.request("/infer", request)
    assert service.engine.calls == 0


@pytest.mark.parametrize("bad", ["camera", "state", "prompt", "jpeg"])
def test_invalid_inputs_fail_before_generation(bucket, bad):
    service, request, _, _ = setup_service(bucket)
    if bad == "camera":
        request["images"].pop("left_wrist_camera")
    if bad == "state":
        request["state"][9] = -1
    if bad == "prompt":
        request["prompt"] = ""
    if bad == "jpeg":
        b = io.BytesIO()
        Image.new("RGB", (8, 8)).save(b, format="JPEG")
        request["images"]["head_camera"] = base64.b64encode(b.getvalue()).decode()
    with pytest.raises(ValueError):
        service.request("/infer", request)
    assert service.engine.calls == 0


def test_schema_must_not_just_echo_client_claims():
    s = schema()
    validate_schema(s)
    for key, value in [("arm_format", "joint"), ("effector_position_unit", "normalized"), ("group", "right_arm")]:
        bad = copy.deepcopy(s)
        bad["groups"][0][key] = value
        with pytest.raises(ValueError):
            validate_schema(bad)
