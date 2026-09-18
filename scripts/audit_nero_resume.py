"""Bounded full-state resume diagnostic; use an independent checkpoint copy."""

import json
import math
import os
import runpy
from pathlib import Path

from openwam.train.openwam_trainer import OpenWAMTrainer

original_resume = OpenWAMTrainer.resume_if_configured


def audited_resume(self, *args, **kwargs):
    result = original_resume(self, *args, **kwargs)
    engine = self.architecture
    zero = engine.optimizer
    base = zero.optimizer
    states = list(base.state.values())
    assert len(states) == 1
    state = states[0]
    row = {
        "rank": self.accelerator.process_index,
        "resume_position": list(result),
        "engine_step": engine.global_steps,
        "engine_samples": engine.global_samples,
        "accelerate_step": self.accelerator.step,
        "adam_step": int(state["step"].item()),
        "lr": base.param_groups[0]["lr"],
        "betas": base.param_groups[0]["betas"],
        "fp32_numel": zero.single_partition_of_fp32_groups[0].numel(),
        "exp_avg_numel": state["exp_avg"].numel(),
        "exp_avg_sq_numel": state["exp_avg_sq"].numel(),
        "optimizer_device": str(state["exp_avg"].device),
    }
    assert row["engine_step"] == row["adam_step"] == row["accelerate_step"] == 4000
    assert row["engine_samples"] == 384000
    assert row["fp32_numel"] == row["exp_avg_numel"] == row["exp_avg_sq_numel"] == 1505288772
    assert row["betas"] == (0.9, 0.95)
    assert math.isclose(row["lr"], 9.948868348456858e-5, rel_tol=1e-12)
    assert row["resume_position"] == [4000, 4000, 0, 3000]
    assert row["optimizer_device"].startswith("cuda")
    output = Path(os.environ["OPENWAM_RESUME_AUDIT_DIR"])
    (output / f"resume_audit_rank{row['rank']}.json").write_text(json.dumps(row, indent=2))
    return result


if __name__ == "__main__":
    OpenWAMTrainer.resume_if_configured = audited_resume
    runpy.run_path(str(Path(__file__).with_name("train_nero_monitored.py")), run_name="__main__")
