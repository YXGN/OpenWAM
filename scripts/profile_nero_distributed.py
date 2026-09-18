"""Run the official training loop with bounded, per-rank stage instrumentation.

Use a separate output directory, a finetune checkpoint, save_steps=0 and
max_steps=12. Steps 1-3 warm up; 4-9 measure; 10-11 use coarse timing only;
step 12 also records a CPU/CUDA trace on rank 0. Never attach to a live run.
"""

import functools
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero import stage_1_and_2 as zero

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openwam.model.architectures.base import BaseWAMArchitecture  # noqa: E402
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture  # noqa: E402
from openwam.model.video_backbone.wan import encode as wan_encode  # noqa: E402
from openwam.train import openwam_trainer as trainer_module  # noqa: E402


class Recorder:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.output = Path(os.environ["OPENWAM_PROFILE_DIR"]).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.armed = False
        self.index = 0
        self.row = {}
        self.started = None
        self.profiler = None
        self.coarse = False

    def call(self, name, fn, *args, detail=True, **kwargs):
        if not self.armed or (self.coarse and detail):
            return fn(*args, **kwargs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function(name):
            result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        self.row[name] = self.row.get(name, 0.0) + time.perf_counter() - started
        return result

    def wrap(self, owner, attr, name, *, detail=True):
        original = getattr(owner, attr)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return self.call(name, original, *args, detail=detail, **kwargs)

        setattr(owner, attr, wrapped)

    def begin(self):
        self.coarse = self.index in (9, 10)
        if self.index == 11 and self.rank == 0:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            )
            self.profiler.start()
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def finish(self, kwargs):
        torch.cuda.synchronize()
        self.row.update(
            rank=self.rank,
            global_step=kwargs["global_step"],
            iteration=self.index,
            warmup=self.index < 3,
            coarse_only=self.coarse,
            traced=self.index == 11,
            total=time.perf_counter() - self.started,
            loss=kwargs["metrics"]["loss_total"],
            grad_norm=kwargs["metrics"]["grad_norm"],
            allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        with (self.output / f"rank{self.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(self.row) + "\n")
        self.row = {}
        self.index += 1
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.output / "rank0_trace.json"))
            (self.output / "rank0_operators_cpu.txt").write_text(
                self.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
            )
            self.profiler = None
        if self.index >= 12:
            self.armed = False


recorder = Recorder()
Trainer = trainer_module.OpenWAMTrainer
original_prepare = Trainer.prepare_accelerate


def prepare(self, *args, **kwargs):
    result = original_prepare(self, *args, **kwargs)
    if self.cfg.training.resume_ckpt_path:
        raise ValueError("Use an isolated finetune diagnostic; do not resume the production output directory")
    if self.cfg.training.save_steps or self.cfg.training.max_steps != 12:
        raise ValueError("Diagnostic requires save_steps=0 and max_steps=12")
    if self.accelerator.num_processes != 6:
        raise ValueError("This diagnostic expects exactly six ranks")
    (recorder.output / f"deepspeed_rank{recorder.rank}.json").write_text(
        json.dumps(self.accelerator.state.deepspeed_plugin.deepspeed_config, indent=2)
    )
    recorder.armed = True
    return result


original_iter = DataLoaderShard.__iter__


def iterate(self):
    iterator = original_iter(self)
    while True:
        if recorder.armed:
            recorder.begin()
        try:
            batch = recorder.call("data_wait", next, iterator, detail=False)
        except StopIteration:
            return
        yield batch


original_log = Trainer.log_step


def log(self, **kwargs):
    result = recorder.call("logging", original_log, self, detail=False, **kwargs)
    if recorder.armed:
        recorder.finish(kwargs)
    return result


Trainer.prepare_accelerate = prepare
Trainer.log_step = log
DataLoaderShard.__iter__ = iterate
recorder.wrap(Trainer, "compute_loss", "prepare_forward_loss", detail=False)
recorder.wrap(BaseWAMArchitecture, "prepare_inputs", "prepare_inputs")
recorder.wrap(BaseWAMArchitecture, "compute_loss", "forward_and_loss")
recorder.wrap(DualSystemSelfAttnArchitecture, "forward", "joint_forward")
recorder.wrap(wan_encode, "encode_text", "text_encoding")
recorder.wrap(wan_encode, "preprocess_video", "pixels_to_gpu")
recorder.wrap(wan_encode, "encode_video", "vae_encoding")
recorder.wrap(BaseWAMArchitecture, "_compute_video_loss", "video_loss")
recorder.wrap(BaseWAMArchitecture, "_compute_action_loss", "action_loss")
recorder.wrap(Accelerator, "backward", "accelerate_backward_and_step", detail=False)
recorder.wrap(DeepSpeedEngine, "backward", "ds_backward")
recorder.wrap(DeepSpeedEngine, "step", "ds_step")
recorder.wrap(zero.DeepSpeedZeroOptimizer, "scaled_global_norm", "gradient_norm")
recorder.wrap(zero.DeepSpeedZeroOptimizer, "_optimizer_step", "local_adamw")
recorder.wrap(zero.DeepSpeedZeroOptimizer, "check_overflow", "overflow_check")
recorder.wrap(zero, "all_gather_dp_groups", "parameter_all_gather")
recorder.wrap(trainer_module, "reduce_step_metrics", "metrics_reduce", detail=False)

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
