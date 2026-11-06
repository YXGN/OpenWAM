"""PR58 additions retain the local Nero cache and inference contracts, without weights."""

from types import SimpleNamespace

import pytest
import torch

from openwam.dataloader.utils.nero_video_cache import GEOMETRY, SHAPE, prepare_cached_ti2v
from openwam.model.video_backbone.wan import encode
from tests.test_text_offload_lifecycle import _Encoder, _RecordingOffload, _Tokenizer
from tests.test_vace_context_truncation import _make_backbone


def _backbone():
    return SimpleNamespace(
        text_encoder=_Encoder().eval().requires_grad_(False),
        _tokenizer=_Tokenizer(),
        device="cpu",
        _training_text_cache=None,
    )


@pytest.mark.parametrize("change", ["dtype", "autocast"])
def test_bounded_cache_refreshes_execution_context(change):
    bb = _backbone()
    cache = encode.TrainTextEmbedCache()

    def run():
        return encode.encode_text(
            ["same"], tokenizer=bb._tokenizer, text_encoder=bb.text_encoder, device=bb.device, cache=cache
        )[0]

    run()
    if change == "dtype":
        bb.text_encoder.double()
        actual = run()
        assert actual.dtype == torch.float64
    else:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = run()
            reference = encode.encode_text(
                ["same"], tokenizer=bb._tokenizer, text_encoder=bb.text_encoder, device="cpu"
            )[0]
            assert torch.equal(actual, reference)
    assert bb.text_encoder.calls == (2 if change == "dtype" else 3)


def test_raw_and_preencoded_training_share_rba_cache_without_running_vae(monkeypatch):
    monkeypatch.setenv("OPENWAM_TEXT_CACHE", "1")
    monkeypatch.setenv("OPENWAM_TEXT_MISS_FAST", "0")
    bb = _backbone()
    bb._training_text_cache = SimpleNamespace(encode=lambda *args, **kwargs: pytest.fail("RBA cache selected"))
    expected, _ = encode.encode_text_for_training(["same"], backbone=bb)
    bb._nero_preencoded_video_enabled = True
    bb.dtype = torch.bfloat16
    bb.vae = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    monkeypatch.setattr(bb.vae, "forward", lambda *args: pytest.fail("preencoded cache must bypass VAE"))
    result = prepare_cached_ti2v(bb, torch.zeros(1, *SHAPE, dtype=torch.bfloat16), GEOMETRY, ["same"])
    assert torch.equal(result["context"], expected)
    assert bb.text_encoder.calls == 1
    assert bb._train_text_cache.hits == 1


def test_legacy_cache_keeps_its_full_batch_miss_behavior(monkeypatch):
    monkeypatch.delenv("OPENWAM_TEXT_CACHE", raising=False)
    bb = _backbone()
    bb._training_text_cache = encode.FrozenTextCache(tokenizer=bb._tokenizer, text_encoder=bb.text_encoder)
    first, _ = encode.encode_text_for_training(["same"], backbone=bb)
    second, _ = encode.encode_text_for_training(["same"], backbone=bb)
    assert torch.equal(first, second)
    assert bb.text_encoder.calls == 1
    assert "_train_text_cache" not in bb.__dict__


def test_backbone_migration_detaches_offload_snapshot_and_clears_both_caches(monkeypatch):
    monkeypatch.setenv("OPENWAM_TEXT_CACHE", "1")
    bb = _make_backbone("plain")
    bb.text_encoder = _Encoder().eval().requires_grad_(False)
    bb._tokenizer = _Tokenizer()
    encode.encode_text_for_training(["same"], backbone=bb)
    cache = bb._train_text_cache
    offload = _RecordingOffload(bb.text_encoder, cache)
    bb._training_text_cache = encode.FrozenTextCache(tokenizer=bb._tokenizer, text_encoder=bb.text_encoder)
    bb._training_text_cache.encode(["same"], device="cpu")
    bb.double()
    assert offload.events[-1] == "disable"
    assert not hasattr(bb.text_encoder, "_openwam_text_offload")
    assert not hasattr(cache, "offload") and not cache.rows
    assert not bb._training_text_cache.entries
    actual, _ = encode.encode_text_for_training(["same"], backbone=bb)
    assert actual.dtype == torch.float64


def test_cache_cannot_reuse_rows_for_a_replacement_encoder():
    bb = _backbone()
    cache = encode.TrainTextEmbedCache()
    before, _ = encode.encode_text(
        ["same"], tokenizer=bb._tokenizer, text_encoder=bb.text_encoder, device="cpu", cache=cache
    )
    replacement = _Encoder().eval().requires_grad_(False)
    after, _ = encode.encode_text(
        ["same"], tokenizer=bb._tokenizer, text_encoder=replacement, device="cpu", cache=cache
    )
    reference, _ = encode.encode_text(["same"], tokenizer=bb._tokenizer, text_encoder=replacement, device="cpu")
    assert torch.equal(after, reference)
    assert not torch.equal(before, after)
    assert cache.hits == 0 and cache.misses == 2
