"""Nero YAML/CLI precedence must reach the model loader without changing semantics."""

import json
import logging
import shutil
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from openwam.deploy import nero_chunk_server
from openwam.deploy.nero_contract import DEFAULT_INFERENCE_CONFIG, inference_config
from tests.dataloader.test_nero_multitask import bucket as bucket


def write_config(path, **changes):
    cfg = OmegaConf.load(DEFAULT_INFERENCE_CONFIG)
    cfg.update(changes)
    OmegaConf.save(cfg, path)
    return path


def test_default_yaml_loads_shared_nero_recipe_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert OmegaConf.to_container(inference_config()) == {
        "inference": {"denoise_steps": 4, "denoise_mode": "sync", "cfg_scale": 1.0},
        "optimization": {
            "inference_fast_path": {"enabled": True},
            "cuda_graph": {"enabled": True},
            "request_context_cache": {"enabled": True},
            "decode_video": False,
            "compile": {"enabled": False},
            "dit_cache": {"enabled": False},
            "prompt_embed_cache": {"enabled": True, "maxsize": 32},
        },
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"denoise_steps": 0},
        {"denoise_steps": -1},
        {"denoise_steps": True},
        {"denoise_steps": 3.5},
        {"denoise_steps": "8"},
        {"cuda_graph": "false"},
        {"inference_fast_path": 1},
        {"request_context_cache": None},
        {"action_mode": "eef"},
    ],
)
def test_invalid_config_fails_before_checkpoint_access(tmp_path, changes):
    path = write_config(tmp_path / "deploy.yaml", **changes)
    with pytest.raises(ValueError):
        nero_chunk_server.load_service(tmp_path / "missing-checkpoint", "fake.safetensors", config_path=path)


@pytest.mark.parametrize("steps", [0, -2, True, 1.5])
def test_invalid_step_override_is_rejected(steps):
    with pytest.raises(ValueError, match="positive integer"):
        inference_config(steps)


@pytest.mark.parametrize("override", [False, True])
def test_server_cli_resolves_yaml_and_overrides_before_loading(bucket, tmp_path, monkeypatch, caplog, override):
    from openwam.deploy import server as official_server

    _, dl, _, _ = bucket
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    training = OmegaConf.create({"dataloader": OmegaConf.to_container(dl)})
    OmegaConf.save(training, checkpoint / "config.yaml")
    shutil.copyfile(dl.normalization_stats_path, checkpoint / "normalization_stats.npy")
    (checkpoint / "fake.safetensors").write_bytes(b"test-only: loader is replaced")
    config_path = write_config(tmp_path / "deploy.yaml", denoise_steps=6, cuda_graph=False)
    seen = []

    def build(deploy, ckpt_dir, device, ckpt_name):
        seen.append((deploy, ckpt_dir, device, ckpt_name))
        return SimpleNamespace(engine=object(), cfg=official_server.merge_deploy_cfg(training, deploy))

    monkeypatch.setattr(official_server, "build_server_from_config", build)
    lifecycle = []
    monkeypatch.setattr(
        nero_chunk_server,
        "make_http_server",
        lambda *args: SimpleNamespace(
            server_address=("127.0.0.1", 8849),
            serve_forever=lambda: lifecycle.append("serve"),
            server_close=lambda: lifecycle.append("close"),
        ),
    )
    argv = ["nero", "--ckpt-dir", str(checkpoint), "--ckpt-name", "fake.safetensors", "--config", str(config_path)]
    if override:
        argv += ["--denoise-steps", "8", "--disable-inference-fast-path"]
    monkeypatch.setattr(sys, "argv", argv)
    with caplog.at_level(logging.INFO):
        nero_chunk_server.main()
    assert len(seen) == 1
    cfg, path, device, name = seen[0]
    assert cfg.inference.denoise_steps == (8 if override else 6)
    assert cfg.optimization.inference_fast_path.enabled is (not override)
    assert cfg.optimization.cuda_graph.enabled is False
    assert cfg.optimization.request_context_cache.enabled is True
    assert (path, device, name) == (str(checkpoint), "cuda:0", "fake.safetensors")
    assert lifecycle == ["serve", "close"]
    assert f"denoise_steps={8 if override else 6} fast_path={not override}" in caplog.text


@pytest.mark.parametrize("steps, expected", [(None, 6), (8, 8)])
def test_openloop_manifest_records_effective_yaml_config(bucket, tmp_path, steps, expected):
    from openwam.evaluation.nero_openloop import run

    root, dl, _, _ = bucket
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    OmegaConf.save(OmegaConf.create({"dataloader": OmegaConf.to_container(dl)}), checkpoint / "config.yaml")
    shutil.copyfile(dl.normalization_stats_path, checkpoint / "normalization_stats.npy")
    (checkpoint / "fake.safetensors").write_bytes(b"audit-only: no model loading")
    config_path = write_config(tmp_path / "deploy.yaml", denoise_steps=6, request_context_cache=False)
    output = tmp_path / "evaluation"
    run(
        SimpleNamespace(
            dataset=root,
            ckpt_dir=checkpoint,
            ckpt_name="fake.safetensors",
            output=output,
            episodes=[0],
            starts=[0],
            stride=32,
            max_windows=1,
            seed=42,
            denoise_steps=steps,
            config=config_path,
            audit_only=True,
            device="cpu",
            verify_parity=False,
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["denoise_steps"] == expected
    assert manifest["inference"]["inference"]["denoise_steps"] == expected
    assert manifest["inference"]["optimization"]["request_context_cache"]["enabled"] is False
