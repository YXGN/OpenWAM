"""CPU HTTP integration for the OpenWAM front and the VelaBot client."""

from __future__ import annotations

import base64
import io
import threading
from collections.abc import Mapping
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from openwam_front import (
    ACTION_DIM,
    ACTION_HORIZON,
    CLIENT_IMAGE_ROLES,
    MODEL_CONTRACTS,
    PROMPT,
    SERVER_IMAGE_ROLES,
    ModelIdentity,
    ModelOutput,
    OpenWAMFront,
    _Handler,
    make_multiview_canvas,
)
from PIL import Image

from velabot.contracts.descriptor import ActuatorGroupSpec, EndEffectorSpec, RobotDescriptor
from velabot.contracts.io_schema import ArmFormat, IOSchema
from velabot.inference.client import negotiate
from velabot.inference.protocol import OpenWAMCodec, ProtocolTransport
from velabot.inference.session import InferenceSession
from velabot.inference.strategies.receding_horizon import RecedingHorizonStrategy
from velabot.inference.transport import HttpJsonTransport


def _schema() -> IOSchema:
    def arm(name: str) -> ActuatorGroupSpec:
        return ActuatorGroupSpec(
            name=name,
            kind="arm",
            dof=7,
            joint_names=tuple(f"{name}_j{i}" for i in range(7)),
            end_effector=EndEffectorSpec(
                name=f"{name}_gripper",
                kind="gripper_2f",
                dof=1,
                protocol_key="agilex.parallel_gripper.width_m/v1",
                position_unit="width_m",
                joint_names=("opening_width",),
                limits=(0.0, 0.1),
            ),
        )

    descriptor = RobotDescriptor(
        robot_id="openwam_test",
        morphology="dual_arm",
        actuator_groups=(arm("left_arm"), arm("right_arm")),
    )
    return IOSchema.from_descriptor(descriptor, arm_format=ArmFormat.ROT6D, include_base=False)


def _jpeg(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG", quality=95)
    return stream.getvalue()


class _Model:
    def __init__(self, action_mode: str = "eef", source_profile: str = "legacy") -> None:
        source = {
            "repository": "OpenWAM-Official/OpenWAM",
            "revision": "c171854a18a2acbeaadf46a746c4118c0e0c00f7",
            "patch_sha256": None,
            "files_sha256": {},
        }
        if source_profile == "integrated":
            source = {
                "repository": "YXGN/OpenWAM",
                "base_revision": "059cbfbfc068355f3d60fe9b014ff6469d1e94a3",
                "revision": "059cbfbfc068355f3d60fe9b014ff6469d1e94a3",
                "patch_sha256": "a" * 64,
                "files_sha256": {"velabot/source_files.json": "b" * 64},
            }
        if action_mode == "delta":
            source.update(
                patch_sha256="a" * 64,
                files_sha256={"openwam/deploy/engine.py": "b" * 64},
            )
        self.identity = ModelIdentity(
            checkpoint="checkpoint_step_3000.safetensors",
            config="config.yaml",
            stats="normalization_stats.npy",
            source=source,
            model_contract=dict(MODEL_CONTRACTS[action_mode]),
            action_mode=action_mode,
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            stats_sha256="c" * 64,
        )
        self.images: list[Image.Image] = []
        self.states: list[list[float]] = []
        self.prompts: list[str] = []

    def reset(self) -> None:
        pass

    def generate(self, *, image: Image.Image, state: list[float], prompt: str) -> ModelOutput:
        self.images.append(image.copy())
        self.states.append(list(state))
        self.prompts.append(prompt)
        row = [0.0] * ACTION_DIM
        row[0:9] = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        row[9] = -1.0
        row[10:19] = [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        row[19] = 1.0
        return ModelOutput([list(row) for _ in range(ACTION_HORIZON)], step=11, latency_ms=0.5)


class _RecordingFront(OpenWAMFront):
    def __init__(self, model: _Model) -> None:
        super().__init__(model)
        self.infer_requests: list[Mapping[str, object]] = []

    def infer(self, request: Mapping[str, object]) -> dict:
        self.infer_requests.append(dict(request))
        return super().infer(request)


@pytest.mark.parametrize("action_mode", ("eef", "delta"))
@pytest.mark.parametrize("source_profile", ("legacy", "integrated"))
def test_real_http_client_negotiates_and_streams_native_openwam_chunk(action_mode, source_profile):
    schema = _schema()
    model = _Model(action_mode, source_profile)
    front = _RecordingFront(model)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.front = front  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = HttpJsonTransport(f"http://127.0.0.1:{server.server_port}", timeout_sec=2.0)
    protocol = ProtocolTransport(transport, OpenWAMCodec(schema))
    raw_images = {
        "head_fpv": _jpeg((320, 256), (20, 30, 40)),
        "left_hand": _jpeg((160, 128), (50, 60, 70)),
        "right_hand": _jpeg((160, 128), (80, 90, 100)),
    }
    observation = {
        "state": [0.0] * 20,
        "images": raw_images,
        "prompt": PROMPT,
    }
    observation["state"][9] = 0.075
    observation["state"][19] = 0.025
    try:
        info = negotiate(protocol, schema)
        assert info.name == "openwam"
        assert info.action_dim == ACTION_DIM
        assert info.action_horizon == ACTION_HORIZON
        assert info.dt_sec == 1.0 / 60.0
        assert info.image_roles == CLIENT_IMAGE_ROLES
        assert info.raw["io_schema"] == schema.to_wire_dict()
        assert info.raw["action_mode"] == action_mode
        assert info.raw["model_contract"] == MODEL_CONTRACTS[action_mode]
        if source_profile == "integrated":
            assert info.raw["source"]["repository"] == "YXGN/OpenWAM"
            assert info.raw["source"]["base_revision"] == "059cbfbfc068355f3d60fe9b014ff6469d1e94a3"
        else:
            assert info.raw["source"]["revision"] == "c171854a18a2acbeaadf46a746c4118c0e0c00f7"

        strategy = RecedingHorizonStrategy(observation_getter=lambda: observation)
        session = InferenceSession(
            protocol,
            strategy,
            expected_action_dim=ACTION_DIM,
            expected_horizon=ACTION_HORIZON,
        )
        session.step_once()
        snapshot = session.state.snapshot()
        assert snapshot is not None
        actions = snapshot.actions
        assert actions.shape == (ACTION_HORIZON, ACTION_DIM)
        assert actions[0, 9] == pytest.approx(0.0)
        assert actions[0, 19] == pytest.approx(0.1)
        assert session.stats.accepted == 1

        # The client sends the source JPEG bytes untouched; only the JSON
        # envelope adds base64 and the front owns decoding/preprocessing.
        wire_request = front.infer_requests[0]
        wire_images = wire_request["images"]
        assert isinstance(wire_images, Mapping)
        for client_role, raw in raw_images.items():
            server_role = {
                "head_fpv": "head_camera",
                "left_hand": "left_wrist_camera",
                "right_hand": "right_wrist_camera",
            }[client_role]
            assert base64.b64decode(wire_images[server_role]) == raw  # type: ignore[index]

        expected_canvas = make_multiview_canvas(
            {
                server_role: Image.open(io.BytesIO(raw_images[client_role])).convert("RGB")
                for client_role, server_role in zip(CLIENT_IMAGE_ROLES, SERVER_IMAGE_ROLES, strict=True)
            }
        )
        assert model.images[0].size == (320, 384)
        for point in ((0, 0), (0, 256), (160, 256)):
            assert model.images[0].getpixel(point) == expected_canvas.getpixel(point)
        assert np.asarray(model.images[0]).tobytes() == np.asarray(expected_canvas).tobytes()
        assert model.states[0][9] == pytest.approx(0.5)
        assert model.states[0][19] == pytest.approx(-0.5)
        assert model.prompts == [PROMPT]

        # Native normalized grippers map back to physical width in metres.
        assert actions[0, 9] == 0.0
        assert actions[0, 19] == 0.1
        assert actions[0, 0] == 0.1
        assert actions[0, 10] == 0.4
    finally:
        protocol.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
