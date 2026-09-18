"""Measure all-rank memory and throughput through the official training loop.

Use scripts/train_nero_8gpu.sh --benchmark. The launcher caps the run at 100
updates, disables checkpoint writes, and uses constant LR for this diagnostic.
No model, dataset, loss or optimizer implementation is replaced.
"""

import csv
import json
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402

_compute_loss = OpenWAMTrainer.compute_loss
_log_step = OpenWAMTrainer.log_step
_rows = []


def _measured_compute(self, batch):
    torch.cuda.reset_peak_memory_stats()
    return _compute_loss(self, batch)


def _measured_log(self, **kwargs):
    torch.cuda.synchronize()
    accelerator = self.accelerator
    if not all(torch.isfinite(torch.tensor(value)) for value in kwargs["metrics"].values()):
        raise RuntimeError("Non-finite loss or gradient norm during benchmark")
    local = torch.tensor(
        [
            [
                1.0 / kwargs["steps_per_sec"],
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ]
        ],
        device=accelerator.device,
        dtype=torch.float64,
    )
    measured = accelerator.gather(local)
    if not torch.isfinite(measured).all():
        raise RuntimeError("Non-finite benchmark measurements")
    _log_step(self, **kwargs)
    if not accelerator.is_main_process:
        return
    output = Path(kwargs["output_path"])
    row = {
        "step": kwargs["global_step"],
        "slowest_rank_step_seconds": float(measured[:, 0].max()),
        "peak_allocated_gib_max_rank": float(measured[:, 1].max()),
        "peak_reserved_gib_max_rank": float(measured[:, 2].max()),
        "loss": kwargs["metrics"]["loss_total"],
        "grad_norm": kwargs["metrics"]["grad_norm"],
    }
    _rows.append(row)
    with (output / "benchmark_steps.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if len(_rows) == 1:
            writer.writeheader()
        writer.writerow(row)
    steady = _rows[10:]  # First ten updates include allocator/kernel warmup.
    if steady:
        total_seconds = sum(r["slowest_rank_step_seconds"] for r in steady)
        global_batch = int(self.cfg.training.batch_size) * accelerator.num_processes
        result = {
            "world_size": accelerator.num_processes,
            "per_gpu_batch": int(self.cfg.training.batch_size),
            "dataset_workers_per_rank": int(self.cfg.training.dataset_num_workers),
            "gradient_accumulation_steps": int(self.cfg.training.gradient_accumulation_steps),
            "measured_steps_after_warmup": len(steady),
            "mean_step_seconds": total_seconds / len(steady),
            "global_windows_per_second": global_batch * len(steady) / total_seconds,
            "peak_allocated_gib_max_rank": max(r["peak_allocated_gib_max_rank"] for r in _rows),
            "peak_reserved_gib_max_rank": max(r["peak_reserved_gib_max_rank"] for r in _rows),
            "scope": "Diagnostic constant-LR run; includes data loading and per-step logging; no policy quality claim.",
        }
        (output / "benchmark_summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    OpenWAMTrainer.compute_loss = _measured_compute
    OpenWAMTrainer.log_step = _measured_log
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
