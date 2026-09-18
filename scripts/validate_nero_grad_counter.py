"""Full-model gradient/count A/B validation; no parameter updates or saves."""

import argparse
import functools
import gc
import json
import sys
import time
from pathlib import Path

import torch
from deepspeed.runtime.utils import count_used_parameters_in_backward
from deepspeed.utils.torch import register_grad_hook
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from openwam.train.utils.deepspeed_grad_counter import GraphParameterCounter  # noqa: E402
from openwam.train.utils.seeding import seed_everything  # noqa: E402


@torch.no_grad()
def compare_gradients(params, reference):
    diff2 = torch.zeros((), device="cuda", dtype=torch.float64)
    ref2 = diff2.clone()
    actual2 = diff2.clone()
    dot = diff2.clone()
    maximum = diff2.clone()
    ref_max = diff2.clone()
    mismatches = torch.zeros((), device="cuda", dtype=torch.int64)
    for p, ref in zip(params, reference):
        assert (p.grad is None) == (ref is None), "Gradient presence changed"
        if ref is None:
            continue
        a = p.grad.detach().float()
        b = ref.to(device="cuda", dtype=torch.float32)
        d = a - b
        diff2 += d.square().sum(dtype=torch.float64)
        ref2 += b.square().sum(dtype=torch.float64)
        actual2 += a.square().sum(dtype=torch.float64)
        dot += (a * b).sum(dtype=torch.float64)
        maximum = torch.maximum(maximum, d.abs().max())
        ref_max = torch.maximum(ref_max, b.abs().max())
        mismatches += torch.any(d != 0)
    return {
        "max_abs": maximum.item(),
        "reference_max_abs": ref_max.item(),
        "relative_l2": (diff2 / ref2.clamp_min(1e-60)).sqrt().item(),
        "cosine": (dot / (ref2 * actual2).sqrt().clamp_min(1e-60)).item(),
        "reference_l2": ref2.sqrt().item(),
        "actual_l2": actual2.sqrt().item(),
        "different_tensors": mismatches.item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deterministic", action="store_true", help="Diagnostic only: deterministic PyTorch/FA2 backward"
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1
    free, total = torch.cuda.mem_get_info()
    assert free >= 50 * 2**30
    torch.cuda.set_per_process_memory_fraction(min(0.60, (free - 4 * 2**30) / total))
    seed_everything(42)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        # FA2 is external to torch and needs its own switch. No model source or
        # production attention setting is changed by this validation process.
        try:
            import flash_attn
        except ImportError:
            pass
        else:
            flash_attn.flash_attn_func = functools.partial(flash_attn.flash_attn_func, deterministic=True)
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
    inputs = arch.prepare_inputs([dataset[i * 50000] for i in range(16)])
    # This reuses one batch only for A/B; it does not enable dataset VAE caching.
    arch.move_frozen_to_device(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    optimizer = trainer.build_optimizer()
    params = optimizer.param_groups[0]["params"]
    del optimizer
    rows, reference, original_loss = [], None, None
    for mode in ("warmup", "original_a", "original_b", "cached_a", "cached_b", "audit"):
        counter = GraphParameterCounter(params, audit=mode == "audit")
        counters = {"calls": 0, "cpu_seconds": 0.0}

        def hook(*unused):
            start = time.perf_counter()
            if mode.startswith("original") or mode == "warmup":
                count_used_parameters_in_backward(params)
            else:
                counter()
            counters["cpu_seconds"] += time.perf_counter() - start
            counters["calls"] += 1

        handles = [register_grad_hook(p, hook) for p in params]
        torch.manual_seed(4444)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = arch.compute_loss(**inputs)
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        result["loss"].backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        row = {
            "mode": mode,
            "forward_seconds": forward_end - start,
            "backward_seconds": backward_end - forward_end,
            "loss": result["loss"].item(),
            "count": counters,
            "cache_rebuilds": counter.rebuilds,
            "audit_checks": counter.audit_checks,
        }
        if mode == "original_a":
            reference = [None if p.grad is None else p.grad.detach().cpu() for p in params]
            original_loss = row["loss"]
        elif reference is not None:
            row["gradient_comparison"] = compare_gradients(params, reference)
            assert row["loss"] == original_loss
        rows.append(row)
        (output / "steps.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
        for handle in handles:
            handle.remove()
        arch.zero_grad(set_to_none=True)
    by_mode = {r["mode"]: r for r in rows}
    baseline = by_mode["original_b"]["gradient_comparison"]
    # Compare against repeated original execution, not bitwise identity claims.
    criteria = {
        "relative_l2": 1.25 * baseline["relative_l2"] + 1e-6,
        "max_abs": 1.25 * baseline["max_abs"] + 1e-7,
    }
    summary = {
        "deterministic_diagnostic": args.deterministic,
        "parameter_tensors": len(params),
        "parameter_numel": sum(p.numel() for p in params),
        "criteria": criteria,
        "baseline_repeat": baseline,
        "cached_within_repeat_envelope": all(
            by_mode[m]["gradient_comparison"][k] <= criteria[k] for m in ("cached_a", "cached_b") for k in criteria
        ),
        "all_compared_gradients_bitwise_equal": all(
            r["gradient_comparison"]["different_tensors"] == 0 for r in rows if "gradient_comparison" in r
        ),
        "audited_counts": by_mode["audit"]["audit_checks"],
        "original_backward_mean": (
            by_mode["original_a"]["backward_seconds"] + by_mode["original_b"]["backward_seconds"]
        )
        / 2,
        "cached_backward_mean": (by_mode["cached_a"]["backward_seconds"] + by_mode["cached_b"]["backward_seconds"]) / 2,
        "limitations": "One fixed batch covering three tasks, full gradients; not a multi-GPU optimizer or policy-success evaluation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
