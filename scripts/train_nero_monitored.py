"""Run the normal trainer with local per-rank telemetry, without extra CUDA fences."""

import json
import math
import os
import runpy
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402

original_log = OpenWAMTrainer.log_step
previous = None
handle = None


def measured_log(self, **kwargs):
    global previous, handle
    now, cpu = time.monotonic(), time.process_time()
    if not all(math.isfinite(float(v)) for v in kwargs["metrics"].values()):
        raise RuntimeError("Non-finite training metrics")
    row = {
        "timestamp": time.time(),
        "rank": self.accelerator.process_index,
        "pid": os.getpid(),
        "microstep": kwargs["global_step"],
        "optimizer_step": kwargs["opt_step"],
        "microstep_seconds": 1 / kwargs["steps_per_sec"],
        "cpu_percent_one_core": None if previous is None else 100 * (cpu - previous[1]) / (now - previous[0]),
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "lr": kwargs["lr"],
        **kwargs["metrics"],
    }
    previous = now, cpu
    if handle is None:
        handle = (Path(kwargs["output_path"]) / f"telemetry_rank{row['rank']}.jsonl").open("a", buffering=1)
    handle.write(json.dumps(row) + "\n")
    original_log(self, **kwargs)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    free, total = torch.cuda.mem_get_info()
    # Leave 3 GiB outside our allocator on a shared GPU; do not touch other jobs.
    torch.cuda.set_per_process_memory_fraction(min(0.95, (free - 3 * 2**30) / total), local_rank)
    OpenWAMTrainer.log_step = measured_log
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
