"""Cache coverage, bf16 integrity and unchanged absolute labels/state contracts."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.utils.nero import sha256
from openwam.dataloader.utils.nero_video_cache import (
    FORMAT,
    GEOMETRY,
    SHAPE,
    NeroVideoCache,
    atomic_json,
    prepare_cached_ti2v,
    source_contract,
)
from tests.dataloader import test_nero_multitask_absolute as absolute_tests

absolute_bucket = absolute_tests.absolute_bucket
bucket = absolute_tests.bucket


def make_cache(directory, dataset):
    directory.mkdir()
    values = torch.arange(len(dataset), dtype=torch.float32).to(torch.bfloat16)
    tensor = values[:, None, None, None, None].expand(len(dataset), *SHAPE).contiguous()
    shards = []
    for sid, start in enumerate(range(0, len(dataset), 5)):
        part = tensor[start : start + 5]
        path = directory / f"shard_{sid:05d}.safetensors"
        save_file({"latents": part}, path)
        shards.append(
            dict(file=path.name, start=start, count=len(part), bytes=path.stat().st_size, sha256=sha256(path))
        )
    manifest = dict(
        format=FORMAT,
        complete=True,
        source=source_contract(dataset),
        shape=list(SHAPE),
        dtype="bfloat16",
        shard_size=5,
        shards=shards,
        vae_sha256="test-only",
    )
    atomic_json(directory / "manifest.json", manifest)
    return manifest, tensor


def test_cached_reader_keeps_labels_and_skips_rgb(absolute_bucket, tmp_path, monkeypatch):
    cfg = OmegaConf.create(OmegaConf.to_container(absolute_bucket[0]))
    raw = build_dataset(cfg)
    expected = [raw[i] for i in range(len(raw))]
    _, latents = make_cache(tmp_path / "cache", raw)
    cfg.video_latent_cache_dir = str(tmp_path / "cache")
    cached = build_dataset(cfg)

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached reader must not decode any camera")

    monkeypatch.setattr(cached, "_decode_one_camera", forbidden)
    for i in (0, 4, 5, 9, 10, 11):
        actual = cached[i]
        assert torch.equal(actual["video_latents"], latents[i])
        assert actual["video_geometry"] == GEOMETRY
        assert actual["video"] == [] and actual["first_frame_image"] is None
        for key in ("action", "action_mask", "proprio", "proprio_mask", "video_mask"):
            assert torch.equal(actual[key], expected[i][key]), key
        assert actual["prompt"] == expected[i]["prompt"]
        actual["video_latents"].zero_()
        assert torch.equal(cached[i]["video_latents"], latents[i])


@pytest.mark.parametrize("failure", ["partial", "source", "gap", "shape", "truncate", "augmentation"])
def test_rejects_incompatible_cache(absolute_bucket, tmp_path, failure):
    dataset = build_dataset(absolute_bucket[0])
    manifest, _ = make_cache(tmp_path / "cache", dataset)
    if failure == "partial":
        manifest["complete"] = False
    elif failure == "source":
        manifest["source"]["dataset_fingerprint"] = "wrong"
    elif failure == "gap":
        manifest["shards"][1]["start"] += 1
    elif failure == "shape":
        manifest["shape"][0] = 16
    elif failure == "truncate":
        (tmp_path / "cache" / manifest["shards"][0]["file"]).write_bytes(b"bad")
    elif failure == "augmentation":
        dataset._color_jitter = object()
    atomic_json(tmp_path / "cache" / "manifest.json", manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        NeroVideoCache(tmp_path / "cache", dataset)


def test_cached_conditioning_is_clean_first_latent_and_owns_storage():
    vae = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    context = torch.randn(2, 8, 4)
    lengths = torch.tensor([5, 7])
    cache = SimpleNamespace(encode=lambda text, device: (context.clone(), lengths.clone()))
    backbone = SimpleNamespace(
        _nero_preencoded_video_enabled=True, vae=vae, dtype=torch.bfloat16, device="cpu", _training_text_cache=cache
    )
    latent = torch.randn(2, *SHAPE).to(torch.bfloat16)
    result = prepare_cached_ti2v(backbone, latent, GEOMETRY, ["a", "b"])
    assert result["fuse_vae_embedding_in_latents"] and result["num_clean_prefix_frames"] == 0
    assert torch.equal(result["first_frame_latents"], latent[:, :, 0:1])
    result["first_frame_latents"].zero_()
    assert latent.abs().sum() > 0
    assert torch.equal(result["context"], context)
    backbone._nero_preencoded_video_enabled = False
    with pytest.raises(ValueError, match="validated"):
        prepare_cached_ti2v(backbone, latent, GEOMETRY, ["a", "b"])
