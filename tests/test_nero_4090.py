"""Low-memory placement must not change Nero model or wire semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.deploy.model_loader import _place_deploy_model
from openwam.model.video_backbone.wan.encode import encode_text, encode_text_for_inference
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from tests.test_frozen_text_cache import Encoder, Tokenizer


def tiny_architecture(*, fail=False):
    # A real Wan module type, without allocating the actual 5B/UMT5 models.
    backbone = object.__new__(Wan22Ti2v)
    torch.nn.Module.__init__(backbone)
    backbone.text_encoder = torch.nn.Linear(3, 4)
    backbone.dit = torch.nn.Linear(3, 4)
    observations = []

    def place(dtype, device):
        observations.append(dict(backbone.named_parameters()))
        if fail:
            raise RuntimeError("placement failed")
        backbone.to(dtype=dtype, device=device)

    return SimpleNamespace(video_backbone=backbone, set_dtype_device=place), observations


def test_cpu_t5_never_enters_model_device_move_and_is_restored():
    arch, seen = tiny_architecture()
    original = arch.video_backbone.text_encoder
    weights = original.weight.detach().clone()
    # Meta represents a second device in a CPU-only test. Moving T5 there
    # would destroy its data and make restoration to CPU fail.
    _place_deploy_model(arch, torch.bfloat16, torch.device("meta"), text_encoder_cpu=True)
    assert seen and all(not name.startswith("text_encoder.") for name in seen[0])
    assert arch.video_backbone.dit.weight.device.type == "meta"
    assert arch.video_backbone.text_encoder is original
    assert original.weight.device.type == "cpu" and original.weight.dtype == torch.bfloat16
    torch.testing.assert_close(original.weight, weights.bfloat16(), rtol=0, atol=0)


def test_failed_placement_restores_registered_encoder():
    arch, _ = tiny_architecture(fail=True)
    original = arch.video_backbone.text_encoder
    with pytest.raises(RuntimeError, match="placement failed"):
        _place_deploy_model(arch, torch.bfloat16, torch.device("meta"), text_encoder_cpu=True)
    assert arch.video_backbone.text_encoder is original
    assert "text_encoder.weight" in dict(arch.video_backbone.named_parameters())


def test_default_placement_keeps_original_all_module_route():
    arch, seen = tiny_architecture()
    _place_deploy_model(arch, torch.bfloat16, torch.device("cpu"))
    assert "text_encoder.weight" in seen[0]


def test_cpu_t5_rejects_other_backbones():
    with pytest.raises(ValueError, match="Wan backbone"):
        _place_deploy_model(SimpleNamespace(), torch.bfloat16, torch.device("cpu"), text_encoder_cpu=True)


@pytest.mark.parametrize("cpu", [False, True])
def test_server_builder_passes_cpu_placement_to_loader(monkeypatch, cpu):
    from omegaconf import OmegaConf

    import openwam.deploy
    from openwam.deploy import model_loader, server
    from openwam.deploy.nero_contract import inference_config

    seen = []
    architecture = object()

    def load(path, **kw):
        seen.append((path, kw))
        return OmegaConf.create({}), architecture

    monkeypatch.setattr(model_loader, "load_from_checkpoint_dir", load)
    monkeypatch.setattr(openwam.deploy, "JointInferenceEngine", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(server, "PolicyServer", lambda **kw: SimpleNamespace(**kw))
    result = server.build_server_from_config(inference_config(), "model", "cuda:0", "x.safetensors", text_encoder_cpu=cpu)
    assert result.engine.architecture is architecture
    expected = {"device": "cuda:0", "ckpt_name": "x.safetensors"}
    if cpu:
        expected["text_encoder_cpu"] = True
    assert seen == [("model", expected)]


def test_inference_text_keeps_padding_values_and_cache_behavior():
    encoder, tokenizer = Encoder().eval(), Tokenizer()
    expected, lengths = encode_text(["duck"], tokenizer=tokenizer, text_encoder=encoder, device="cpu")
    cache = {}
    kw = dict(vace_cache=None, prompt_embed_cache=cache, tokenizer=tokenizer, text_encoder=encoder, device="cpu")
    with torch.no_grad():
        context, actual_lengths = encode_text_for_inference("duck", **kw)
        calls = encoder.calls
        repeated = encode_text_for_inference("duck", **kw)
    torch.testing.assert_close(context, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, lengths, rtol=0, atol=0)
    assert repeated[0] is context and encoder.calls == calls
    assert not context.requires_grad


def test_inference_text_runs_on_encoder_device_and_caches_on_model_device():
    encoder = Encoder().eval()
    cache = {}
    kw = dict(vace_cache=None, prompt_embed_cache=cache, tokenizer=Tokenizer(), text_encoder=encoder, device="meta")
    context, lengths = encode_text_for_inference("duck", **kw)
    assert context.device.type == lengths.device.type == "meta"
    assert next(encoder.parameters()).device.type == "cpu"
    encode_text_for_inference("duck", **kw)
    assert encoder.calls == 1


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/check_nero_4090.py"
    spec = importlib.util.spec_from_file_location("check_nero_4090", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_check_rejects_changed_and_extra_code(checker, tmp_path):
    folder = tmp_path / "openwam"
    folder.mkdir()
    path = folder / "x.py"
    path.write_text("x=1\n")
    manifest = {"files": {"openwam/x.py": {"bytes": path.stat().st_size, "sha256": checker.digest(path)}}}
    assert checker.verify_files(tmp_path, manifest)["status"] == "passed"
    path.write_text("x=2\n")
    with pytest.raises(ValueError, match="changed"):
        checker.verify_files(tmp_path, manifest)
    path.write_text("x=1\n")
    (folder / "retired.py").write_text("")
    with pytest.raises(ValueError, match="extra Python"):
        checker.verify_files(tmp_path, manifest)


def test_action_comparison_checks_each_arm_in_physical_units(checker):
    actions = np.zeros((32, 20), dtype=np.float32)
    actions[:, [3, 7, 13, 17]] = 1
    limits = {"position_max_mm": 2, "rotation_max_deg": 1, "gripper_max_mm": 1}
    assert checker.compare_actions(actions, actions, limits)["passed"]
    shifted = actions.copy()
    shifted[12, 10] += .003
    result = checker.compare_actions(shifted, actions, limits)
    assert not result["passed"] and result["arms"]["left"]["passed"]
    assert result["arms"]["right"]["position_max_mm"] == pytest.approx(3)
    shifted = actions.copy()
    shifted[7, 9] += .002
    assert not checker.compare_actions(shifted, actions, limits)["passed"]
