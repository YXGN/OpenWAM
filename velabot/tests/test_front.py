from __future__ import annotations

import base64
import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
from openwam_front import (
    ACTION_DIM,
    ACTION_HORIZON,
    CLIENT_IMAGE_ROLES,
    MODEL_CONTRACT,
    MODEL_CONTRACTS,
    OPENWAM_IO_SCHEMA,
    PROMPT,
    ModelIdentity,
    ModelOutput,
    OpenWAMFront,
    ProtocolError,
    _Handler,
    load_config,
    make_multiview_canvas,
)
from PIL import Image


class FakeModel:
    def __init__(self) -> None:
        self.identity = ModelIdentity(
            "ckpt",
            "cfg",
            "stats",
            {
                "repository": "OpenWAM-Official/OpenWAM",
                "revision": "c171854a18a2acbeaadf46a746c4118c0e0c00f7",
                "patch_sha256": None,
                "files_sha256": {},
            },
            dict(MODEL_CONTRACT),
        )
        self.reset_count = 0
        self.images = []

    def reset(self) -> None:
        self.reset_count += 1

    def generate(self, *, image, state, prompt) -> ModelOutput:
        self.images.append(image)
        return ModelOutput(
            [[float(i) / 100.0 for i in range(ACTION_DIM)] for _ in range(ACTION_HORIZON)], step=7, latency_ms=1.5
        )


def _jpeg(size=(32, 24), color=(12, 34, 56)) -> str:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _schema_request() -> dict:
    return {
        "wire_format": "openwam_native",
        "action_dim": 20,
        "action_horizon": 32,
        "action_semantics": "absolute_flange_rot6d_20d",
        "dt_sec": 1 / 60,
        "supports_rtc": False,
        "return_hidden": False,
        "image_roles": list(CLIENT_IMAGE_ROLES),
        "io_schema": json.loads(json.dumps(OPENWAM_IO_SCHEMA)),
    }


def _infer_request() -> dict:
    return {
        "wire_format": "openwam_native",
        "action_dim": 20,
        "action_semantics": "absolute_flange_rot6d_20d",
        "prompt": PROMPT,
        "state": [0.0] * 20,
        "images": {"head_camera": _jpeg(), "left_wrist_camera": _jpeg(), "right_wrist_camera": _jpeg()},
    }


def test_handshake_and_single_generate_returns_complete_chunk():
    model = FakeModel()
    front = OpenWAMFront(model)
    reply = front.handshake(_schema_request())
    assert reply["io_schema"] == OPENWAM_IO_SCHEMA
    assert reply["image_roles"] == list(CLIENT_IMAGE_ROLES)
    result = front.infer(_infer_request())
    assert len(result["actions"]) == ACTION_HORIZON
    assert all(len(row) == ACTION_DIM for row in result["actions"])
    assert len(model.images) == 1
    assert model.images[0].size == (320, 384)


def test_delta_identity_keeps_absolute_wire_semantics():
    model = FakeModel()
    model.identity = ModelIdentity(
        model.identity.checkpoint,
        model.identity.config,
        model.identity.stats,
        model.identity.source,
        dict(MODEL_CONTRACTS["delta"]),
        action_mode="delta",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        stats_sha256="c" * 64,
    )
    front = OpenWAMFront(model)
    handshake = front.handshake(_schema_request())
    result = front.infer(_infer_request())
    assert handshake["action_mode"] == result["action_mode"] == "delta"
    assert handshake["model_contract"] == MODEL_CONTRACTS["delta"]
    assert result["action_semantics"] == "absolute_flange_rot6d_20d"


def test_missing_or_mismatched_schema_is_rejected():
    front = OpenWAMFront(FakeModel())
    request = _schema_request()
    del request["io_schema"]["groups"][0]
    with pytest.raises(ProtocolError, match="io_schema"):
        front.handshake(request)


def test_infer_requires_handshake_and_rejects_shape_reason():
    front = OpenWAMFront(FakeModel())
    with pytest.raises(ProtocolError, match="successful /handshake"):
        front.infer(_infer_request())
    front.handshake(_schema_request())
    request = _infer_request()
    request["state"] = [0.0] * 19
    with pytest.raises(ProtocolError, match=r"state.*width 20"):
        front.infer(request)


def test_infer_rejects_missing_camera_rtc_and_nonfinite_output():
    front = OpenWAMFront(FakeModel())
    front.handshake(_schema_request())
    request = _infer_request()
    del request["images"]["right_wrist_camera"]
    with pytest.raises(ProtocolError, match="exactly"):
        front.infer(request)
    request = _infer_request()
    request["rtc"] = True
    with pytest.raises(ProtocolError, match="RTC"):
        front.infer(request)


def test_finite_native_gripper_values_are_clipped_with_a_recorded_intervention():
    class Overshoot(FakeModel):
        def generate(self, *, image, state, prompt) -> ModelOutput:
            actions = [[0.0] * ACTION_DIM for _ in range(ACTION_HORIZON)]
            actions[0][9] = 1.2
            actions[0][19] = -1.2
            return ModelOutput(actions, step=1, latency_ms=0.1)

    front = OpenWAMFront(Overshoot())
    front.handshake(_schema_request())
    payload = front.infer(_infer_request())
    assert payload["actions"][0][9] == 1.0
    assert payload["actions"][0][19] == -1.0
    events = payload["safety"]["interventions"]
    assert payload["gripper_clips"] == 2
    assert events[0]["requested"] == 1.2 and events[0]["effective"] == 1.0
    assert events[1]["requested"] == -1.2 and events[1]["effective"] == -1.0


def test_canvas_uses_native_slot_geometry():
    images = {
        "head_camera": Image.new("RGB", (640, 480), (1, 2, 3)),
        "left_wrist_camera": Image.new("RGB", (640, 480), (4, 5, 6)),
        "right_wrist_camera": Image.new("RGB", (640, 480), (7, 8, 9)),
    }
    canvas = make_multiview_canvas(images)
    assert canvas.size == (320, 384)
    assert canvas.getpixel((0, 0)) == (1, 2, 3)
    assert canvas.getpixel((0, 256)) == (4, 5, 6)
    assert canvas.getpixel((160, 256)) == (7, 8, 9)


def test_config_requires_fixed_identity_and_contract(tmp_path):
    path = tmp_path / "front.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": "c",
                "config": "config.yaml",
                "stats": "normalization_stats.npy",
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "stats_sha256": "c" * 64,
                "action_mode": "eef",
                "source": {
                    "repository": "OpenWAM-Official/OpenWAM",
                    "revision": "c171854a18a2acbeaadf46a746c4118c0e0c00f7",
                    "patch_sha256": None,
                    "files_sha256": {},
                },
                "model_contract": MODEL_CONTRACT,
                "io_schema": OPENWAM_IO_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    _, identity = load_config(path)
    assert identity.source["revision"].startswith("c171")


def test_config_accepts_delta_identity_and_patch_fingerprint(tmp_path):
    path = tmp_path / "delta-front.json"
    raw = {
        "checkpoint": "delta.safetensors",
        "config": "config.yaml",
        "stats": "normalization_stats.npy",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "stats_sha256": "c" * 64,
        "action_mode": "delta",
        "source": {
            "repository": "OpenWAM-Official/OpenWAM",
            "revision": "c171854a18a2acbeaadf46a746c4118c0e0c00f7",
            "patch_sha256": "d" * 64,
            "files_sha256": {"openwam/deploy/engine.py": "e" * 64},
        },
        "model_contract": MODEL_CONTRACTS["delta"],
        "io_schema": OPENWAM_IO_SCHEMA,
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    _, identity = load_config(path)
    assert identity.action_mode == "delta"
    assert identity.model_contract["delta_xyz_addback"] == "engine_once_xyz_state0"


def test_http_fixture_exposes_real_json_boundary():
    front = OpenWAMFront(FakeModel())
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.front = front
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        body = json.dumps(_schema_request()).encode()
        client.request("POST", "/handshake", body=body, headers={"Content-Type": "application/json"})
        response = client.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["io_schema"] == OPENWAM_IO_SCHEMA
        client.close()

        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        body = json.dumps(_infer_request()).encode()
        client.request("POST", "/infer", body=body, headers={"Content-Type": "application/json"})
        response = client.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200 and len(payload["actions"]) == 32
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
