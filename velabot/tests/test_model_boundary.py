"""Small CPU checks for the OpenWAM model and HTTP boundaries."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import sys
import threading
import types
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from openwam_front import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_MODES,
    CLIENT_IMAGE_ROLES,
    MODEL_CONTRACT,
    MODEL_CONTRACTS,
    OPENWAM_IO_SCHEMA,
    PROMPT,
    SOURCE_REVISION,
    BusyError,
    ModelError,
    ModelIdentity,
    ModelOutput,
    OpenWAMFront,
    _Handler,
    _verify_artifacts,
    _verify_stats_file,
)
from PIL import Image

from velabot.inference.transport import HttpJsonTransport, TransportError


def _identity() -> ModelIdentity:
    return ModelIdentity(
        checkpoint="checkpoint_step_3000.safetensors",
        config="config.yaml",
        stats="normalization_stats.npy",
        source={
            "repository": "OpenWAM-Official/OpenWAM",
            "revision": "c171854a18a2acbeaadf46a746c4118c0e0c00f7",
            "patch_sha256": None,
            "files_sha256": {},
        },
        model_contract=dict(MODEL_CONTRACT),
        action_mode="eef",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        stats_sha256="c" * 64,
    )


class _Model:
    identity = _identity()

    def reset(self) -> None:
        pass

    def generate(self, *, image, state, prompt) -> ModelOutput:
        del image, state, prompt
        return ModelOutput([[0.0] * ACTION_DIM for _ in range(ACTION_HORIZON)])


def _jpeg() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), (1, 2, 3)).save(stream, format="JPEG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _handshake_request() -> dict:
    return {
        "wire_format": "openwam_native",
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_semantics": "absolute_flange_rot6d_20d",
        "dt_sec": 1 / 60,
        "supports_rtc": False,
        "return_hidden": False,
        "image_roles": list(CLIENT_IMAGE_ROLES),
        "io_schema": OPENWAM_IO_SCHEMA,
    }


def _infer_request() -> dict:
    return {
        **_handshake_request(),
        "prompt": PROMPT,
        "state": [0.0] * ACTION_DIM,
        "images": {"head_camera": _jpeg(), "left_wrist_camera": _jpeg(), "right_wrist_camera": _jpeg()},
    }


def _serve(front: OpenWAMFront) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.front = front  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http500_preserves_model_failure_reason():
    class BadModel(_Model):
        def generate(self, *, image, state, prompt) -> ModelOutput:
            del image, state, prompt
            raise ValueError("normalizer input is invalid")

    front = OpenWAMFront(BadModel())
    front.handshake(_handshake_request())
    server, thread = _serve(front)
    transport = HttpJsonTransport(f"http://127.0.0.1:{server.server_port}", timeout_sec=2.0)
    try:
        with pytest.raises(TransportError, match=r"500:.*normalizer input is invalid"):
            transport.request("/infer", _infer_request())
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_handshake_and_infer_are_single_flight():
    entered = threading.Event()
    release = threading.Event()

    class BlockingModel(_Model):
        def generate(self, *, image, state, prompt) -> ModelOutput:
            entered.set()
            assert release.wait(2.0)
            return super().generate(image=image, state=state, prompt=prompt)

    front = OpenWAMFront(BlockingModel())
    front.handshake(_handshake_request())
    worker = threading.Thread(target=front.infer, args=(_infer_request(),))
    worker.start()
    assert entered.wait(2.0)
    with pytest.raises(BusyError, match="already in flight"):
        front.handshake(_handshake_request())
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()


def _official_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, fail_at: int | None = None):
    import openwam_front

    root = tmp_path / "OpenWAM"
    root.mkdir()
    ckpt_dir = tmp_path / "artifacts"
    ckpt_dir.mkdir()
    calls: dict[str, object] = {}

    class Policy:
        def __init__(self) -> None:
            self._executor = types.SimpleNamespace(_action_buffer=[])
            self.calls = 0

        def reset(self) -> None:
            calls["policy_reset"] = int(calls.get("policy_reset", 0)) + 1

        def predict_action(self, _obs):
            self.calls += 1
            if self.calls == 1:
                count = {"short": 2, "long": 32, "normal": 31, "exception": 31}[mode]
                self._executor._action_buffer.extend([np.zeros(ACTION_DIM)] * count)
                return np.ones(ACTION_DIM)
            if fail_at == self.calls:
                raise ValueError("predict failed at step 5")
            self._executor._action_buffer.pop(0)
            return np.ones(ACTION_DIM)

    policy = Policy()

    class Server:
        _policy = policy

        def _init_policy(self) -> None:
            calls["init"] = int(calls.get("init", 0)) + 1

    server = Server()
    server_module = types.ModuleType("openwam.deploy.server")
    server_module.__file__ = str(root / "openwam" / "deploy" / "server.py")
    server_module._load_deploy_yaml = lambda _path: {"inference": {}, "optimization": {}}

    def build_server_from_config(**kwargs):
        calls["build"] = kwargs
        return server

    server_module.build_server_from_config = build_server_from_config
    omega_module = types.ModuleType("omegaconf")
    omega_module.OmegaConf = types.SimpleNamespace(update=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "omegaconf", omega_module)
    monkeypatch.setitem(sys.modules, "openwam.deploy.server", server_module)
    monkeypatch.setattr(openwam_front, "_verify_artifacts", lambda *_args: None)
    monkeypatch.setattr(openwam_front, "_verify_loaded_config", lambda *_args: None)
    args = argparse.Namespace(
        openwam_root=str(root),
        ckpt_dir=str(ckpt_dir),
        ckpt_name="checkpoint_step_3000.safetensors",
        device="cpu",
        denoise_steps=10,
        compile=False,
        dit_cache=False,
        patch=None,
    )
    adapter = openwam_front._official_adapter(_identity(), args)
    return adapter, policy, calls, ckpt_dir


@pytest.mark.parametrize("mode, expected", [("short", "generated 3 steps"), ("long", "generated 33 steps")])
def test_official_adapter_rejects_non_exact_chunks_and_clears_queue(monkeypatch, tmp_path, mode, expected):
    adapter, policy, calls, ckpt_dir = _official_adapter(monkeypatch, tmp_path, mode)
    with pytest.raises(ModelError, match=expected):
        adapter.generate(image=None, state=[0.0] * ACTION_DIM, prompt=PROMPT)
    assert policy._executor._action_buffer == []
    assert calls["build"]["ckpt_dir"] == str(ckpt_dir)  # type: ignore[index]
    assert calls["build"]["ckpt_name"] == "checkpoint_step_3000.safetensors"  # type: ignore[index]


def test_official_adapter_clears_queue_after_mid_generation_failure(monkeypatch, tmp_path):
    adapter, policy, _calls, _ckpt_dir = _official_adapter(monkeypatch, tmp_path, "exception", fail_at=5)
    with pytest.raises(ValueError, match="predict failed at step 5"):
        adapter.generate(image=None, state=[0.0] * ACTION_DIM, prompt=PROMPT)
    assert policy._executor._action_buffer == []


def test_official_adapter_produces_one_exact_chunk_and_reset_reinitializes_policy(monkeypatch, tmp_path):
    adapter, policy, calls, _ckpt_dir = _official_adapter(monkeypatch, tmp_path, "normal")
    output = adapter.generate(image=None, state=[0.0] * ACTION_DIM, prompt=PROMPT)
    assert len(output.actions) == ACTION_HORIZON
    assert all(len(row) == ACTION_DIM for row in output.actions)
    assert policy._executor._action_buffer == []
    adapter.reset()
    assert calls["policy_reset"] == 1


def test_delta_source_fingerprint_allows_only_registered_patched_files(monkeypatch, tmp_path):
    import openwam_front

    root = tmp_path / "OpenWAM"
    root.mkdir()
    patch_path = tmp_path / "openwam-moon-cake-training.patch"
    patch_path.write_bytes(b"patch outside checkout")
    source_files = {
        "openwam/deploy/engine.py": b"patched engine",
        "openwam/deploy/delta_addback.py": b"new helper",
    }
    files_sha256: dict[str, str] = {}
    for relative, content in source_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files_sha256[relative] = hashlib.sha256(content).hexdigest()

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    artifact_hashes: dict[str, str] = {}
    for name, content in {
        "checkpoint.safetensors": b"checkpoint",
        "config.yaml": b"config",
        "normalization_stats.npy": b"stats",
    }.items():
        (artifacts / name).write_bytes(content)
        artifact_hashes[name] = hashlib.sha256(content).hexdigest()
    patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    identity = ModelIdentity(
        checkpoint="checkpoint.safetensors",
        config="config.yaml",
        stats="normalization_stats.npy",
        source={
            "repository": "OpenWAM-Official/OpenWAM",
            "revision": SOURCE_REVISION,
            "patch_sha256": patch_sha256,
            "files_sha256": files_sha256,
        },
        model_contract=dict(MODEL_CONTRACTS["delta"]),
        action_mode="delta",
        checkpoint_sha256=artifact_hashes["checkpoint.safetensors"],
        config_sha256=artifact_hashes["config.yaml"],
        stats_sha256=artifact_hashes["normalization_stats.npy"],
    )

    def fake_run(command, **_kwargs):
        if command[-1] == "HEAD":
            return types.SimpleNamespace(stdout=SOURCE_REVISION)
        if "apply" in command:
            return types.SimpleNamespace(stdout="")
        return types.SimpleNamespace(stdout=" M openwam/deploy/engine.py\n?? openwam/deploy/delta_addback.py\n")

    monkeypatch.setattr(openwam_front.subprocess, "run", fake_run)
    _verify_artifacts(identity, artifacts, identity.checkpoint, root, patch_path)
    assert set(identity.source["files_sha256"]) == set(source_files)
    assert identity.action_mode in ACTION_MODES

    def fake_run_with_unregistered(command, **_kwargs):
        if command[-1] == "HEAD":
            return types.SimpleNamespace(stdout=SOURCE_REVISION)
        if "apply" in command:
            return types.SimpleNamespace(stdout="")
        return types.SimpleNamespace(stdout=" M openwam/deploy/engine.py\n M unrelated.py\n")

    monkeypatch.setattr(openwam_front.subprocess, "run", fake_run_with_unregistered)
    with pytest.raises(ModelError, match="unregistered changes"):
        _verify_artifacts(identity, artifacts, identity.checkpoint, root, patch_path)


def test_loaded_stats_require_delta_state_to_equal_eef(tmp_path):
    stats = {
        block: {key: np.full(20, float(index)) for index, key in enumerate(("min", "max", "mean", "std", "q01", "q99"))}
        for block in ("eef", "delta", "delta_state")
    }
    stats["delta_state"] = {key: value.copy() for key, value in stats["eef"].items()}
    path = tmp_path / "normalization_stats.npy"
    np.save(path, stats, allow_pickle=True)
    _verify_stats_file(path, "delta")
    stats["delta_state"]["mean"][0] += 1.0
    np.save(path, stats, allow_pickle=True)
    with pytest.raises(ModelError, match=r"delta_state\.mean"):
        _verify_stats_file(path, "delta")


def test_loaded_eef_stats_keep_legacy_single_block_compatibility(tmp_path):
    stats = {
        "eef": {key: np.full(20, float(index)) for index, key in enumerate(("min", "max", "mean", "std", "q01", "q99"))}
    }
    path = tmp_path / "normalization_stats.npy"
    np.save(path, stats, allow_pickle=True)
    _verify_stats_file(path, "eef")


@pytest.mark.parametrize(
    "field, value",
    [("type", "other_loader"), ("normalize_mode", "z-score")],
)
def test_loaded_config_rejects_non_mooncake_or_non_minmax_pipeline(monkeypatch, tmp_path, field, value):
    import openwam_front

    dataloader = {
        "type": "moon_cake",
        "action_mode": "eef",
        "normalize_mode": "min-max",
        "num_frames": 33,
        "video_stride": 4,
        "height": 384,
        "width": 320,
        "multiview": True,
        "unify_action": True,
        "unify_action_map": ["0-9", "34-43"],
    }
    dataloader[field] = value
    action_map = np.asarray(list(range(10)) + list(range(34, 44)))
    normalizer = types.SimpleNamespace(_action_dst_index=action_map, _state_dst_index=action_map, _unify_dim=80)
    policy_server = types.SimpleNamespace(
        cfg={"dataloader": dataloader},
        _obs_preprocessor=types.SimpleNamespace(
            multiview=True,
            img_width=320,
            img_height=384,
            camera_layout=["head_camera", "left_camera", "right_camera"],
        ),
        engine=types.SimpleNamespace(
            architecture=types.SimpleNamespace(binary_command_dims=(), normalizer=normalizer),
        ),
        _init_policy=lambda: None,
    )
    monkeypatch.setattr(openwam_front, "_verify_stats_file", lambda *_args: None)
    with pytest.raises(ModelError, match=field):
        openwam_front._verify_loaded_config(policy_server, "eef", tmp_path / "unused.npy")
