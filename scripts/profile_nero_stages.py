"""Isolated Nero stage timing; never attaches to or updates an existing run.

Matches the per-rank model/data shapes. Backward is local (no ZeRO/NCCL),
and no model optimizer step or checkpoint write is performed. A separate
synthetic FP32 AdamW shard benchmark measures only local update kernels.
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.model.video_backbone.wan import encode as wan_encode  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from openwam.train.utils.seeding import per_step_seed, seed_everything  # noqa: E402


class Timings:
    def __init__(self):
        self.row = {}
        self.detailed = True

    def call(self, name, fn, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function(name):
            result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        self.row[name] = self.row.get(name, 0.0) + time.perf_counter() - start
        return result

    def wrap(self, owner, attr, name):
        original = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            if not self.detailed:
                return original(*args, **kwargs)
            return self.call(name, original, *args, **kwargs)

        setattr(owner, attr, wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1, "Select exactly one diagnostic GPU"
    free, total = torch.cuda.mem_get_info()
    assert free >= 50 * 2**30, f"Need 50 GiB free, have {free / 2**30:.1f} GiB"
    # Leave headroom for the existing GPU user; fail instead of consuming the card.
    torch.cuda.set_per_process_memory_fraction(min(0.60, (free - 4 * 2**30) / total))
    seed_everything(42, rank=0)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train_nero_6gpu",
            overrides=[
                f"training.finetune_ckpt_path={Path(args.checkpoint).resolve()}",
                f"training.batch_size={args.batch_size}",
            ],
        )
    dataset = build_dataset(cfg.dataloader, split="train")
    trainer = OpenWAMTrainer(cfg, dataset=dataset)
    arch = trainer.architecture
    arch.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
    arch.move_frozen_to_device(torch.device("cuda:0"))
    trainable_count = sum(p.numel() for p in arch.parameters() if p.requires_grad)
    loader = trainer.build_dataloader(args.batch_size)
    iterator = iter(loader)
    timer = Timings()
    timer.wrap(wan_encode, "encode_text", "text_encoding")
    timer.wrap(wan_encode, "preprocess_video", "pixels_to_gpu")
    timer.wrap(wan_encode, "encode_video", "vae_encoding")
    timer.wrap(arch, "forward", "joint_forward")
    timer.wrap(arch, "_compute_video_loss", "video_loss")
    timer.wrap(arch, "_compute_action_loss", "action_loss")

    def step(index, *, detailed=True):
        timer.row = {}
        timer.detailed = detailed
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = timer.call("data_wait", next, iterator)
        timer.call("seed", torch.manual_seed, per_step_seed(42, rank=0, step=4000 + index))
        inputs = timer.call("prepare_inputs", arch.prepare_inputs, batch)
        result = timer.call("compute_loss", arch.compute_loss, **inputs)
        timer.call("backward_local", result["loss"].backward)
        timer.call("zero_grad", arch.zero_grad, set_to_none=True)
        row = dict(timer.row)
        row["total"] = time.perf_counter() - start
        row["loss"] = float(result["loss"].detach())
        assert math.isfinite(row["loss"])
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        row["index"] = index
        row["detailed"] = detailed
        return row

    rows = []
    for index in range(args.warmup + args.steps):
        row = step(index)
        row["warmup"] = index < args.warmup
        rows.append(row)
        (output / "steps.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    baseline = [step(args.warmup + args.steps + i, detailed=False) for i in range(2)]
    measured = [r for r in rows if not r["warmup"]]
    keys = [k for k in measured[0] if k not in ("index", "detailed", "warmup", "loss")]
    summary = {
        "scope": "Isolated single GPU; same batch/model/data. No distributed communication or real optimizer update.",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "trainable_parameters": trainable_count,
        "mean": {k: statistics.mean(r[k] for r in measured) for k in keys},
        "min": {k: min(r[k] for r in measured) for k in keys},
        "max": {k: max(r[k] for r in measured) for k in keys},
        "coarse_only_baseline": baseline,
        "timing_method": "Wall time with CUDA synchronization at stage boundaries; nested durations overlap.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.trace:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            step(args.warmup + args.steps + 2)
        prof.export_chrome_trace(str(output / "trace.json"))
        (output / "operators.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
        (output / "operators_cpu.txt").write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
        )

    # Release the actual model before measuring a six-way optimizer shard.
    # This is a kernel microbenchmark, not end-to-end ZeRO optimizer timing.
    del step, trainer, loader, dataset
    arch = iterator = timer = None
    gc.collect()
    torch.cuda.empty_cache()
    shard_count = math.ceil(trainable_count / 6)
    parameter = torch.nn.Parameter(torch.full((shard_count,), 0.01, device="cuda", dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    parameter.grad = torch.full_like(parameter, 0.001)
    optimizer_rows = []
    for index in range(8):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        optimizer_rows.append(time.perf_counter() - start)
    summary["synthetic_adamw_shard"] = {
        "fp32_elements": shard_count,
        "mean_seconds_excluding_first": statistics.mean(optimizer_rows[1:]),
        "steps_seconds": optimizer_rows,
        "excludes": "gradient reduction, clipping, bf16/fp32 copies, parameter all-gather; synthetic values",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
