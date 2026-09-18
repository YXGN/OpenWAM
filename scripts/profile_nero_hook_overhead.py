"""Isolate DeepSpeed's per-parameter graph counting on the full Nero graph.

No optimizer, communication, checkpoint writes, or installed-package changes.
The cached-node variant is a diagnostic only, not a production ZeRO patch.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from deepspeed.runtime.utils import count_used_parameters_in_backward
from deepspeed.utils.torch import register_grad_hook
from hydra import compose, initialize_config_dir
from torch.autograd.graph import _get_grad_fn_or_grad_acc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from openwam.train.utils.seeding import seed_everything  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1
    free, total = torch.cuda.mem_get_info()
    assert free >= 50 * 2**30
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
    batch = [dataset[i * 50000] for i in range(16)]
    inputs = arch.prepare_inputs(batch)
    del batch
    arch.move_frozen_to_device(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    optimizer = trainer.build_optimizer()
    params = optimizer.param_groups[0]["params"]
    del optimizer
    nodes = list({_get_grad_fn_or_grad_acc(p) for p in params})
    rows = []
    reference = None
    reference_loss = None
    stats = {}

    def hook(*_):
        started = time.perf_counter()
        if mode == "original_count":
            count = count_used_parameters_in_backward(params)
        else:
            count = sum(map(torch._C._will_engine_execute_node, nodes))
        stats["count_cpu_seconds"] += time.perf_counter() - started
        stats["count_calls"] += 1
        stats["counts_seen"].add(count)
        stats["graph_tasks_seen"].add(torch._C._current_graph_task_id())

    for mode in ("native", "original_count", "cached_nodes"):
        handles = [] if mode == "native" else [register_grad_hook(p, hook) for p in params]
        for i in range(3):
            stats = {"count_cpu_seconds": 0.0, "count_calls": 0, "counts_seen": set(), "graph_tasks_seen": set()}
            torch.manual_seed(4444)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = arch.compute_loss(**inputs)
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            result["loss"].backward()
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            row = {
                "mode": mode,
                "iteration": i,
                "warmup": i == 0,
                "forward_seconds": forward_end - started,
                "backward_seconds": backward_end - forward_end,
                "total_seconds": backward_end - started,
                "loss": float(result["loss"]),
                **{k: sorted(v) if isinstance(v, set) else v for k, v in stats.items()},
            }
            if i == 1:
                if mode == "native":
                    reference = [None if p.grad is None else p.grad.detach().cpu().clone() for p in params]
                    reference_loss = result["loss"].detach().cpu().clone()
                else:
                    equal = True
                    max_diff = 0.0
                    mismatches = 0
                    for p, expected in zip(params, reference):
                        actual = None if p.grad is None else p.grad.detach().cpu()
                        if expected is None or actual is None:
                            match = expected is None and actual is None
                        else:
                            match = torch.equal(actual, expected)
                            if not match:
                                max_diff = max(max_diff, (actual.float() - expected.float()).abs().max().item())
                        equal = equal and match
                        mismatches += not match
                    row.update(
                        gradients_equal_native=equal,
                        gradient_tensor_mismatches=mismatches,
                        max_gradient_abs_difference=max_diff,
                        loss_equal_native=torch.equal(result["loss"].detach().cpu(), reference_loss),
                    )
            arch.zero_grad(set_to_none=True)
            rows.append(row)
            (output / "steps.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        for handle in handles:
            handle.remove()
    summary = {
        "scope": __doc__,
        "parameter_tensors": len(params),
        "parameter_numel": sum(p.numel() for p in params),
        "mean": {},
    }
    for mode in ("native", "original_count", "cached_nodes"):
        measured = [r for r in rows if r["mode"] == mode and not r["warmup"]]
        summary["mean"][mode] = {
            k: statistics.mean(r[k] for r in measured)
            for k in ("forward_seconds", "backward_seconds", "total_seconds", "count_cpu_seconds", "count_calls")
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
