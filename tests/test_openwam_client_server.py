"""Unified OpenWAM wire contract through the real client and temporary HTTP.

The model is a CPU stub; PNG encoding, transport, negotiation, sessions,
preprocessing and action decoding use the actual implementations.
"""

import copy
import io
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from openwam.deploy.nero_chunk_server import CONTRACT, ROLES, SERVER_ROLES, make_http_server
from openwam.deploy.nero_contract import conditions_from_observation
from tests.dataloader.test_nero_multitask import bucket as bucket
from tests.test_nero_chunk_server import schema, setup_service


@pytest.fixture
def client(monkeypatch):
    root = Path(os.environ.get("VELABOT_CLIENT_ROOT", Path(__file__).resolve().parents[2] / "velabot_agentic_os"))
    if not (root / "packages/velabot-autonomy/src/velabot/inference/protocol/openwam.py").is_file():
        pytest.skip("Set VELABOT_CLIENT_ROOT to exercise the real client")
    for package in ("velabot-contracts", "velabot-autonomy"):
        monkeypatch.syspath_prepend(str(root / "packages" / package / "src"))
    from velabot.contracts.io_schema import GroupIOSpec, IOSchema
    from velabot.inference.client import negotiate
    from velabot.inference.protocol import ProtocolTransport, create_codec
    from velabot.inference.transport import HttpJsonTransport

    groups = []
    for group in schema()["groups"]:
        group = dict(group)
        for key in ("effector_names", "effector_position_limits", "effector_position_velocity_max"):
            group[key] = tuple(group[key])
        groups.append(GroupIOSpec(**group))
    io_schema = IOSchema(tuple(groups))
    return io_schema, create_codec, ProtocolTransport, HttpJsonTransport, negotiate


@pytest.mark.parametrize("mode", ["eef", "eef_relative"])
@pytest.mark.parametrize("close_offset", [None, 0.002])
def test_new_name_preserves_png_physical_actions_prompt_and_session(bucket, client, mode, close_offset):
    io_schema, create_codec, protocol_transport, http_transport, negotiate = client
    service, request, images, outputs = setup_service(bucket, gripper_close_offset_m=close_offset)
    service.identity.update(
        model_representation=mode,
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        stats_sha256="c" * 64,
    )
    encoded = {}
    for role, server_role in zip(ROLES, SERVER_ROLES, strict=True):
        stream = io.BytesIO()
        images[server_role].save(stream, format="PNG")
        encoded[role] = stream.getvalue()
    observation = dict(images=encoded, state=request["state"], prompt="Task A")
    expected = outputs.copy()
    if close_offset is not None:
        expected[:, [9, 19]] = np.clip(expected[:, [9, 19]] - np.float32(close_offset), 0, np.float32(0.1))
    codec = create_codec("openwam", schema=io_schema)
    assert codec.wire_format == "openwam/v1"
    assert codec.encode_request("/handshake", {})["contract"] == CONTRACT
    server = make_http_server(service, port=0)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    transport = protocol_transport(http_transport(f"http://127.0.0.1:{server.server_port}"), codec)
    try:
        info = negotiate(transport, io_schema)
        assert info.name == "openwam" and info.action_horizon == 32 and info.dt_sec == 1 / 60
        assert info.raw["wire_format"] == "openwam/v1"
        assert info.raw.get("gripper_close_offset_m") == close_offset
        session = codec.session_id
        for prompt in ("  Pick up the yellow bowl.\n", "把物体交给左手。", "  Pick up the yellow bowl.\n"):
            observation["prompt"] = prompt
            reply = transport.request("/infer", observation)
            assert codec.session_id == session and service.engine.conditions["prompt"] == prompt
            np.testing.assert_array_equal(reply["actions"], expected)
            np.testing.assert_array_equal(reply["anchor_state"], observation["state"])
            direct = conditions_from_observation(
                service.preprocessor, dict(images=images, state=observation["state"], prompt=prompt), 42
            )
            np.testing.assert_array_equal(direct["proprio"], service.engine.conditions["proprio"])
            np.testing.assert_array_equal(
                np.asarray(direct["first_frame_image"]), np.asarray(service.engine.conditions["first_frame_image"])
            )
            if close_offset is not None:
                np.testing.assert_array_equal(reply["raw_gripper_widths_m"], outputs[:, [9, 19]])
        stale = codec.encode_request("/infer", observation)
        transport.request("/reset", {})
        assert codec.session_id != session
        with pytest.raises(ValueError, match="current handshake"):
            service.request("/infer", stale)
        np.testing.assert_array_equal(transport.request("/infer", observation)["actions"], expected)
        assert service.engine.calls == 4
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("wire", ["openwam_native", "openwam_nero/v1", None])
def test_retired_wire_never_enters_model_or_replaces_session(bucket, wire):
    service, request, _, _ = setup_service(bucket)
    original_session = request["session_id"]
    handshake = dict(wire_format=wire, contract=copy.deepcopy(CONTRACT), io_schema=schema())
    with pytest.raises(ValueError, match="Expected wire_format=openwam/v1"):
        service.request("/handshake", handshake)
    with pytest.raises(ValueError, match="Expected wire_format=openwam/v1"):
        service.request("/infer", dict(request, wire_format=wire))
    assert service.engine.calls == 0 and service._session == original_session
