"""GPU0 diagnostic: full compute graph, one approximate six-way optimizer shard.

This is NOT a reproduction of six-rank training. Frozen inputs are prepared
first; full-network gradients are computed, but only an approximately 1/6-sized
parameter subset is updated by single-rank ZeRO-2. No checkpoints are written.
"""

import argparse
import functools
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import deepspeed
import torch
from accelerate import DeepSpeedPlugin
from deepspeed.runtime.zero import stage_1_and_2 as zero
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.model.video_backbone.wan import encode as wan_encode  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from openwam.train.utils.seeding import seed_everything  # noqa: E402


class Timer:
    def __init__(self):
        self.row = {}
        self.enabled = False

    def call(self, name, fn, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function(name):
            result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        self.row[name] = self.row.get(name, 0.0) + time.perf_counter() - start
        return result

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self.enabled:
                return self.call(label, original, *args, **kwargs)
            return original(*args, **kwargs)

        setattr(owner, name, wrapped)


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1
    free, total = torch.cuda.mem_get_info()
    assert free >= 50 * 2**30, "Need 50 GiB free"
    torch.cuda.set_per_process_memory_fraction(min(0.60, (free - 4 * 2**30) / total))
    seed_everything(42)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train_nero_6gpu",
            overrides=[f"training.finetune_ckpt_path={Path(args.checkpoint).resolve()}"],
        )
    dataset = build_dataset(cfg.dataloader, split="train")
    trainer = OpenWAMTrainer(cfg, dataset=dataset)
    arch = trainer.architecture
    arch.set_dtype_device(torch.bfloat16, torch.device("cuda"))
    arch.move_frozen_to_device(torch.device("cuda"))
    timer = Timer()
    timer.wrap(wan_encode, "encode_text", "text_encoding")
    timer.wrap(wan_encode, "preprocess_video", "pixels_to_gpu")
    timer.wrap(wan_encode, "encode_video", "vae_encoding")
    loader = trainer.build_dataloader(16)
    iterator = iter(loader)
    prepared = []
    for i in range(8):
        timer.row = {}
        timer.enabled = True
        batch = timer.call("data_wait", next, iterator)
        inputs = timer.call("prepare_inputs", arch.prepare_inputs, batch)
        prepared.append(move(inputs, "cpu"))
        row = dict(timer.row, phase="prepare", iteration=i, warmup=i < 2)
        with (output / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    del inputs, batch, iterator, loader
    arch.move_frozen_to_device(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    timer.wrap(arch, "forward", "joint_forward")
    engine = None

    def step(phase, i, *, detailed=True, traced=False):
        timer.row = {}
        timer.enabled = detailed
        torch.manual_seed(4400 + i)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        inputs = timer.call("cached_inputs_h2d", move, prepared[i % len(prepared)], "cuda")
        result = timer.call("forward_loss", arch.compute_loss, **inputs)
        if engine is None:
            timer.call("backward", result["loss"].backward)
        else:
            timer.call("backward", engine.backward, result["loss"])
            timer.call("engine_step", engine.step)
        timer.call("full_graph_zero_grad", arch.zero_grad, set_to_none=True)
        row = dict(
            timer.row,
            phase=phase,
            iteration=i,
            warmup=i < 2,
            detailed=detailed,
            traced=traced,
            total=time.perf_counter() - started,
            loss=float(result["loss"].detach()),
            allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        assert torch.isfinite(result["loss"])
        with (output / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return row

    for i in range(5):
        step("native_cached", i)
    optimizer = trainer.build_optimizer()
    assert len(optimizer.param_groups) == 1
    all_params = optimizer.param_groups[0]["params"]
    numel = sum(p.numel() for p in all_params)
    selected, selected_numel = [], 0
    for p in all_params:
        if selected_numel >= numel // 6:
            break
        selected.append(p)
        selected_numel += p.numel()
    optimizer.param_groups[0]["params"] = selected
    plugin = DeepSpeedPlugin(
        zero_stage=2, gradient_accumulation_steps=1, gradient_clipping=1.0, offload_optimizer_device="none"
    )
    config = plugin.deepspeed_config
    config.update(
        train_micro_batch_size_per_gpu=16,
        train_batch_size=16,
        bf16={"enabled": True},
        fp16={"enabled": False},
        zero_allow_untested_optimizer=True,
        steps_per_print=1000000,
    )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    scope = {
        "trainable_numel": numel,
        "optimizer_numel": selected_numel,
        "target_six_way_shard_numel": (numel + 5) // 6,
        "scope": __doc__,
    }
    (output / "scope.json").write_text(json.dumps(scope, indent=2))
    engine, _, _, _ = deepspeed.initialize(model=arch, optimizer=optimizer, config=config)
    # Engine initialization moves frozen modules too; keep them off-device now.
    arch.move_frozen_to_device(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    for name in (
        "scaled_global_norm",
        "get_grad_norm_direct",
        "_optimizer_step",
        "unscale_and_clip_grads",
        "flatten_dense_tensors_aligned",
        "reduce_independent_p_g_buckets_and_remove_grads",
        "independent_gradient_partition_epilogue",
    ):
        timer.wrap(zero.DeepSpeedZeroOptimizer, name, name)
    timer.wrap(zero, "all_gather_dp_groups", "parameter_all_gather")
    for i in range(8):
        step("deepspeed_partial_optimizer", i, detailed=i < 6)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        step("deepspeed_partial_optimizer", 8, traced=True)
    rows = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
    summary = {"scope": scope}
    for phase in ("prepare", "native_cached", "deepspeed_partial_optimizer"):
        measured = [
            r
            for r in rows
            if r["phase"] == phase and not r["warmup"] and not r.get("traced") and r.get("detailed", True)
        ]
        keys = [k for k, v in measured[0].items() if isinstance(v, float)]
        summary[phase] = {k: statistics.mean(r[k] for r in measured) for k in keys}
    coarse = [r for r in rows if r.get("detailed") is False]
    summary["deepspeed_partial_optimizer_coarse"] = {
        k: statistics.mean(r[k] for r in coarse) for k in ("forward_loss", "backward", "engine_step", "total")
    }
    summary["observer_effect"] = (
        "Fine-grained reduction-hook CUDA fences add overhead: use coarse measurements "
        "for wall-time comparison; trace step is excluded."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    profiler.export_chrome_trace(str(output / "trace.json"))
    (output / "operators_cpu.txt").write_text(
        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
