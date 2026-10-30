"""The HTTP boundary must preserve the audited engine's observations and output."""

import base64
import copy
import io
import json
import logging
import threading
from contextlib import contextmanager
from http.client import HTTPConnection
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.deploy import nero_chunk_server
from openwam.deploy.nero_chunk_server import (
    CONTRACT,
    SERVER_ROLES,
    WIRE,
    NeroChunkService,
    make_http_server,
    validate_schema,
)
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


def setup_service(bucket, *, gripper_threshold_m=None, gripper_close_offset_m=None):
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
    service = NeroChunkService(
        engine, cfg, {"checkpoint": "fake", "model_representation": "eef"},
        gripper_threshold_m=gripper_threshold_m, gripper_close_offset_m=gripper_close_offset_m,
    )
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
    assert "gripper_threshold_m" not in reply
    assert "gripper_close_offset_m" not in reply
    assert "raw_gripper_widths_m" not in reply


def test_gripper_threshold_changes_only_output_widths_and_preserves_raw_values(bucket):
    service, request, images, outputs = setup_service(bucket, gripper_threshold_m=0.08)
    threshold = np.float32(0.08)
    widths = np.array(
        [
            -0.001,
            0,
            0.027,
            0.049,
            np.nextafter(threshold, np.float32(0)),
            threshold,
            np.nextafter(threshold, np.float32(1)),
            0.1,
            0.1004,
        ],
        dtype=np.float32,
    )
    outputs[:, 9] = np.resize(widths, 32)
    outputs[:, 19] = np.resize(widths[::-1], 32)
    original = outputs.copy()
    # Check the same policy is advertised on handshake and applied by /infer.
    handshake = service.request("/handshake", dict(wire_format=WIRE, contract=CONTRACT, io_schema=schema()))
    assert handshake["gripper_threshold_m"] == 0.08
    request["session_id"] = handshake["session_id"]
    reply = service.request("/infer", request)
    expected = original.copy()
    expected[:, 9] = np.resize([0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1], 32)
    expected[:, 19] = np.resize([0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0], 32)
    np.testing.assert_array_equal(reply["actions"], expected)
    np.testing.assert_array_equal(reply["raw_gripper_widths_m"], original[:, [9, 19]])
    np.testing.assert_array_equal(outputs, original)
    np.testing.assert_array_equal(reply["anchor_state"], request["state"])
    direct = conditions_from_observation(
        service.preprocessor, dict(images=images, state=request["state"], prompt=request["prompt"]), 42
    )
    np.testing.assert_array_equal(service.engine.conditions["proprio"], direct["proprio"])
    assert reply["gripper_threshold_m"] == 0.08
    assert service.engine.calls == 1
    json.dumps(reply, allow_nan=False)


def test_gripper_close_offset_is_continuous_in_metres_and_preserves_state_and_pose(bucket):
    service, request, images, outputs = setup_service(bucket, gripper_close_offset_m=0.002)
    raw = [-0.001, 0, 0.001, 0.002, 0.003, 0.027, 0.079, 0.080, 0.081, 0.100, 0.1004, 0.103]
    commanded = [0, 0, 0, 0, 0.001, 0.025, 0.077, 0.078, 0.079, 0.098, 0.0984, 0.1]
    outputs[:, 9] = np.resize(raw, 32)
    outputs[:, 19] = np.resize(raw[::-1], 32)
    original = outputs.copy()
    handshake = service.request("/handshake", dict(wire_format=WIRE, contract=CONTRACT, io_schema=schema()))
    assert handshake["gripper_close_offset_m"] == 0.002 and "gripper_threshold_m" not in handshake
    request["session_id"] = handshake["session_id"]
    pose_dims = [d for d in range(20) if d not in (9, 19)]
    # Repeat requests: no cumulative subtraction, engine-array mutation, or hidden latch.
    for _ in range(2):
        reply = service.request("/infer", request)
        actions = np.asarray(reply["actions"])
        np.testing.assert_allclose(actions[:, 9], np.resize(commanded, 32), atol=1e-8, rtol=0)
        np.testing.assert_allclose(actions[:, 19], np.resize(commanded[::-1], 32), atol=1e-8, rtol=0)
        np.testing.assert_array_equal(actions[:, pose_dims], original[:, pose_dims])
        np.testing.assert_array_equal(reply["raw_gripper_widths_m"], original[:, [9, 19]])
        np.testing.assert_array_equal(outputs, original)
        np.testing.assert_array_equal(reply["anchor_state"], request["state"])
        assert reply["gripper_close_offset_m"] == 0.002 and "gripper_threshold_m" not in reply
    direct = conditions_from_observation(
        service.preprocessor, dict(images=images, state=request["state"], prompt=request["prompt"]), 42
    )
    np.testing.assert_array_equal(service.engine.conditions["proprio"], direct["proprio"])
    assert service.engine.calls == 2
    json.dumps(reply, allow_nan=False)


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


def test_prompt_switch_is_verbatim_on_the_same_session(bucket):
    service, request, _, _ = setup_service(bucket)
    session = request["session_id"]
    for index, prompt in enumerate(["Task A\n保留换行", "  Task B\nnew goal  ", "Task A\n保留换行"]):
        request.update(prompt=prompt, request_id=f"switch-{index}")
        response = service.request("/infer", request)
        assert response["session_id"] == session
        assert service.engine.conditions["prompt"] == prompt
    assert service.engine.calls == 3


def test_prompt_log_references_repeat_and_prints_changes_without_changing_actions(bucket, caplog):
    service, request, _, outputs = setup_service(bucket)
    prompts = ["Task A\n保留换行", "Task A\n保留换行", "  Task B\nnew goal  ", "Task A\n保留换行"]
    with caplog.at_level(logging.INFO):
        for prompt in prompts:
            request["prompt"] = prompt
            response = service.request("/infer", request)
            assert service.engine.conditions["prompt"] == prompt
            np.testing.assert_array_equal(response["actions"], outputs)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 4
    for message, revision in zip(messages, [1, 1, 2, 3], strict=True):
        assert f"infer id='one' prompt=P{revision} infer_ms=" in message
        assert " server_ms=" in message
        assert "\n" not in message
    for index in (0, 2, 3):
        assert messages[index].endswith(f" text={prompts[index]!r}")
    assert " text=" not in messages[1]


@pytest.mark.parametrize("profile", [False, True])
def test_server_timing_includes_queue_and_preprocessing(bucket, monkeypatch, caplog, profile):
    service, request, _, _ = setup_service(bucket)
    service.engine._profile = profile
    now = [100.0]
    monkeypatch.setattr(nero_chunk_server, "time", SimpleNamespace(monotonic=lambda: now[0]))

    @contextmanager
    def queued_lock():
        now[0] += 0.04
        yield

    service._lock = queued_lock()
    original_preprocess = nero_chunk_server.conditions_from_observation
    original_generate = service.engine.generate

    def preprocess(*args):
        now[0] += 0.03
        return original_preprocess(*args)

    def generate(conditions):
        now[0] += 0.2
        result = original_generate(conditions)
        if profile:
            result["timings_ms"] = {"setup": 10.0, "vae": 20.0, "denoise": 160.0, "action": 10.0}
            result["t5_cache_hit"] = True
        return result

    monkeypatch.setattr(nero_chunk_server, "conditions_from_observation", preprocess)
    monkeypatch.setattr(service.engine, "generate", generate)
    with caplog.at_level(logging.INFO):
        response = service.request("/infer", request)
    assert response["server_total_sec"] == pytest.approx(0.27)
    assert "infer_ms=200.0 server_ms=270.0" in caplog.records[0].getMessage()
    assert "timings_ms" not in response
    if profile:
        message = caplog.records[1].getMessage()
        for value in ("queue=40.0", "obs=30.0", "denoise=160.0", "total=270.0", "t5_cache=hit"):
            assert value in message
    else:
        assert len(caplog.records) == 1


def test_inference_log_reports_actual_graph_backend(bucket, caplog):
    service, request, _, _ = setup_service(bucket)
    calls = []

    def status(*, include_signature):
        calls.append(include_signature)
        return {"last_backend": "cuda_graph"}

    service.engine.architecture = SimpleNamespace(_mot_cuda_graph_runner=SimpleNamespace(status=status))
    with caplog.at_level(logging.INFO):
        service.request("/infer", request)
    assert calls == [False]
    assert "backend=cuda_graph" in caplog.records[-1].getMessage()


@pytest.mark.parametrize("profile", [False, True])
def test_http_logs_one_summary_per_inference_and_retains_error_access_logs(bucket, caplog, profile):
    service, request, _, outputs = setup_service(bucket)
    service.engine._profile = profile
    server = make_http_server(service, port=0)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    connection = HTTPConnection(*server.server_address, timeout=10)
    try:
        with caplog.at_level(logging.INFO):
            for prompt, expected_status in [("Task A", 200), ("Task A", 200), ("", 400)]:
                request["prompt"] = prompt
                connection.request("POST", "/infer", json.dumps(request), {"Content-Type": "application/json"})
                response = connection.getresponse()
                body = json.loads(response.read())
                assert response.status == expected_status
                if expected_status == 200:
                    np.testing.assert_array_equal(body["actions"], outputs)
                    assert "timings_ms" not in body
                else:
                    assert body["error"] == "An explicit task prompt is required"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == (5 if profile else 3)
    assert sum(message.startswith("infer ") for message in messages) == 2
    details = [message for message in messages if message.startswith("timing_ms ")]
    assert len(details) == (2 if profile else 0)
    for message in details:
        assert "request=" in message and "response=" in message and "queue=" in message
    assert any('"POST /infer HTTP/1.1" 400' in message for message in messages)
    assert not any('"POST /infer HTTP/1.1" 200' in message for message in messages)
