"""Wait for idle GPUs2/3/4/7, validate batch32, then resume with telemetry.

No GPU allocation while waiting; no signal is sent to unrelated processes.
One supervisor per launch directory (flock). Errors stop the queue rather than
silently changing the recipe or repeatedly running an OOM configuration.
"""

import argparse
import fcntl
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

import pynvml as nvml

ROOT = Path(__file__).resolve().parents[1]
GPUS = (2, 3, 4, 7)
RUN = ROOT / "outputs/nero_multitask_4gpu/resume_step4000"
SOURCE = ROOT / "outputs/nero_multitask_6gpu/2026-09-17_18-22-36"


def idle_snapshot():
    rows = []
    for index in GPUS:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
        rows.append(
            {
                "gpu": index,
                "free_gib": memory.free / 2**30,
                "utilization": utilization,
                "pids": [p.pid for p in processes],
            }
        )
    return rows


def all_idle(rows):
    return {r["gpu"] for r in rows} == set(GPUS) and all(
        not r["pids"] and r["free_gib"] >= 75 and r["utilization"] <= 5 for r in rows
    )


def read_tail(path, limit=256000):
    if not path.exists():
        return []
    with path.open("rb") as f:
        size = f.seek(0, 2)
        f.seek(max(0, size - limit))
        lines = f.read().splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            pass
    return rows


def summarize(run):
    ranks = [{r["microstep"]: r for r in read_tail(run / f"telemetry_rank{i}.jsonl")} for i in range(4)]
    common = sorted(set.intersection(*(set(r) for r in ranks)))
    steps = [s for s in common if s > 4010][-50:]
    if len(steps) < 20:
        return None
    times = [max(r[s]["microstep_seconds"] for r in ranks) for s in steps]
    if not all(math.isfinite(r[s]["loss_total"]) for r in ranks for s in steps):
        raise RuntimeError("Non-finite loss in telemetry")
    mean = statistics.mean(times)
    cv = statistics.pstdev(times) / mean
    return {
        "sample_count": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "global_batch": 128,
        "mean_update_seconds": mean,
        "min_max_seconds": [min(times), max(times)],
        "cv": cv,
        "stable_window": cv <= 0.20,
        "samples_per_second": 128 / mean,
        "speedup_vs_old_6gpu_15s_update": 15 / mean,
        "throughput_ratio_vs_old_6gpu": (128 / mean) / (96 / 15),
        "peak_allocated_gib": [max(r[s]["peak_allocated_gib"] for s in steps) for r in ranks],
        "peak_reserved_gib": [max(r[s]["peak_reserved_gib"] for s in steps) for r in ranks],
        "timestamp": time.time(),
        "scope": "Four GPUs, batch32 each, native GPU AdamW. Old six-GPU baseline used global batch96; compare sample throughput as well as step time.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.launch_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / "queue.pid").write_text(str(os.getpid()))
    nvml.nvmlInit()
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="2,3,4,7",
        WANDB_MODE="online",
        OPENWAM_REQUIRE_IDLE="1",
    )
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_RESUME", None)
    children = []

    def status(name, **details):
        (folder / "status").write_text(name + "\n")
        data = {"status": name, "timestamp": time.time(), **details}
        (folder / "queue_state.json").write_text(json.dumps(data, indent=2))
        print(json.dumps(data), flush=True)

    def start(command, logfile, child_env=env):
        log = (folder / logfile).open("a", buffering=1)
        process = subprocess.Popen(
            command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        log.close()
        children.append(process)
        return process

    def wait_idle():
        consecutive = 0
        while consecutive < 3:
            rows = idle_snapshot()
            consecutive = consecutive + 1 if all_idle(rows) else 0
            status("waiting_for_idle_gpus", consecutive_idle_checks=consecutive, gpus=rows)
            if consecutive < 3:
                time.sleep(30)

    def stop_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        marker = RUN / "accel_state_step_4000/trainer_state.json"
        if not marker.exists():
            status("preparing_cpu_checkpoint", gpus=idle_snapshot())
            convert = start(
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/convert_nero_zero2_6to4.py",
                    "--source-run",
                    str(SOURCE),
                    "--target-run",
                    str(RUN),
                    "--micro-batch",
                    "32",
                ],
                "conversion.log",
                dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8"),
            )
            if convert.wait() != 0:
                raise RuntimeError("Checkpoint conversion failed; see conversion.log")
        wait_idle()
        status("validating_batch32")
        smoke = start(
            [
                "bash",
                "scripts/train_nero_4gpu.sh",
                "training.resume_ckpt_path=null",
                f"training.finetune_ckpt_path={SOURCE}/checkpoint_step_4000.safetensors",
                "training.num_epochs=null",
                "training.max_steps=4",
                "training.save_steps=0",
                "training.save_full_states_for_resume=false",
                "training.lr_scheduler=null",
                f"training.output_path={folder}/batch32_smoke",
            ],
            "batch32_validation.log",
            dict(env, WANDB_MODE="disabled"),
        )
        if smoke.wait() != 0:
            raise RuntimeError("Batch32 validation failed; formal training was not started")
        smoke_roots = sorted((folder / "batch32_smoke").glob("*/telemetry_rank0.jsonl"))
        if not smoke_roots:
            raise RuntimeError("Batch32 validation has no complete four-step telemetry")
        for rank in range(4):
            rows = read_tail(smoke_roots[-1].parent / f"telemetry_rank{rank}.jsonl")
            if len(rows) < 4 or not all(math.isfinite(r["loss_total"]) for r in rows):
                raise RuntimeError(f"Batch32 validation incomplete/non-finite on rank {rank}")
        wait_idle()  # Recheck after smoke released memory; other jobs may have started.
        status("starting_training")
        train = start(["bash", "scripts/train_nero_4gpu.sh"], "train.log")
        (folder / "train.pid").write_text(str(train.pid))
        monitor = start(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/monitor_nero_resources.py",
                "--gpus",
                "2,3,4,7",
                "--output",
                str(folder / "resources.jsonl"),
                "--seconds",
                "3600",
            ],
            "monitor.log",
        )
        stable_reported = False
        while train.poll() is None:
            summary = summarize(RUN)
            if summary:
                (folder / "stable_summary.json").write_text(json.dumps(summary, indent=2))
                if summary["stable_window"] and not stable_reported:
                    status("training_stable", **summary)
                    stable_reported = True
            time.sleep(30)
        if monitor.poll() is None:
            monitor.terminate()
        status("completed" if train.returncode == 0 else "training_failed", exit_code=train.returncode)
    except KeyboardInterrupt:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        status("paused_by_user")
    except Exception as exc:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        status("needs_attention", error=str(exc))
        raise


if __name__ == "__main__":
    main()
