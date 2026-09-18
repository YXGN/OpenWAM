"""Bounded VAE cache audit on real Nero windows; does not change training."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.model.video_backbone.wan.encode import preprocess_video  # noqa: E402
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38  # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1, "Select one diagnostic GPU"
    free, total = torch.cuda.mem_get_info()
    assert free > 12 * 2**30, "Need at least 12 GiB free"
    torch.cuda.set_per_process_memory_fraction(12 * 2**30 / total)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train_nero_6gpu")
    dataset = build_dataset(cfg.dataloader, split="train")
    representatives = {}
    for i, ep in enumerate(dataset._episodes):
        representatives.setdefault(tuple(ep["tasks"]), int(dataset._cum_n_starts[i]))
    starts = list(representatives.values())
    indices = [starts[i % len(starts)] + i for i in range(16)]
    samples = [dataset[i] for i in indices]
    vae = WanVideoVAE38()
    prefix = "video_backbone.vae."
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        weights = {k[len(prefix) :]: handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix)}
    vae.load_state_dict(weights, strict=True)
    del weights
    vae.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)

    def pixels(frames):
        return preprocess_video(frames, dtype=torch.bfloat16, device="cuda")

    batch = torch.cat([pixels(s["video"]) for s in samples])
    timings = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        latent = vae.batch_encode(batch, device="cuda")
        torch.cuda.synchronize()
        if i:
            timings.append(time.perf_counter() - start)
    expected = latent.cpu().contiguous()
    path = output / "sample_latents.safetensors"
    save_file({"latents": expected}, str(path))
    cached = load_file(str(path))["latents"]
    assert torch.equal(expected, cached), "Cache serialization must be lossless"
    repeated = vae.batch_encode(batch, device="cuda").cpu()
    repeat_diff = (expected.float() - repeated.float()).abs()
    # An aligned temporal crop of a longer encode still carries earlier history.
    a, b = dataset[0]["video"], dataset[16]["video"]
    longer = vae.batch_encode(pixels(a + b[5:]), device="cuda").cpu()
    standalone = vae.batch_encode(pixels(b), device="cuda").cpu()
    difference = (longer[:, :, 1:4].float() - standalone.float()).abs()
    read_times = []
    for _ in range(10):
        start = time.perf_counter()
        loaded = load_file(str(path))["latents"].to("cuda")
        torch.cuda.synchronize()
        read_times.append(time.perf_counter() - start)
        del loaded
    per_window = expected[0].numel() * expected.element_size()
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "indices": indices,
        "prompts": sorted({s["prompt"] for s in samples}),
        "input_shape": list(batch.shape),
        "latent_shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "cache_roundtrip_equal": torch.equal(expected, cached),
        "repeat_encode_max_abs_difference": repeat_diff.max().item(),
        "whole_clip_slice_max_abs_difference": difference.max().item(),
        "whole_clip_slice_mean_abs_difference": difference.mean().item(),
        "vae_batch16_seconds": timings,
        "vae_batch16_seconds_mean": statistics.mean(timings),
        "warm_page_cache_read_and_h2d_seconds_mean": statistics.mean(read_times),
        "num_windows": len(dataset),
        "raw_cache_bytes": len(dataset) * per_window,
        "raw_cache_gib": len(dataset) * per_window / 2**30,
        "projected_encode_only_gpu_hours": len(dataset) / 16 * statistics.mean(timings) / 3600,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "limitations": "Small sample; warm page-cache read is not full shuffled-cache throughput. Encoding projection excludes video decoding and disk writes. No model update or full-dataset cache generation.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
