"""Sample selected GPU board metrics and this training's process tree every 2 seconds."""

import argparse
import json
import time

import psutil
import pynvml as nvml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--gpus", default="6,7")
    args = parser.parse_args()
    nvml.nvmlInit()
    handles = {i: nvml.nvmlDeviceGetHandleByIndex(i) for i in map(int, args.gpus.split(","))}
    processes = {}
    deadline = time.monotonic() + args.seconds
    with open(args.output, "a", buffering=1) as output:
        while time.monotonic() < deadline:
            roots = []
            for p in psutil.process_iter(["pid", "cmdline"]):
                if "scripts/train_nero_monitored.py" in (p.info["cmdline"] or []):
                    roots.append(p)
            active = {}
            for root in roots:
                try:
                    for p in [root, *root.children(recursive=True)]:
                        active[p.pid] = processes.get(p.pid, p)
                except psutil.NoSuchProcess:
                    pass
            cpu_rows = []
            for pid, p in active.items():
                try:
                    cpu_rows.append(
                        {"pid": pid, "cpu_percent_one_core": p.cpu_percent(), "rss_gib": p.memory_info().rss / 2**30}
                    )
                except psutil.NoSuchProcess:
                    pass
            processes = active
            gpu_rows = []
            for index, handle in handles.items():
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_rows.append(
                    {
                        "index": index,
                        "gpu_percent": util.gpu,
                        "memory_busy_percent": util.memory,
                        "used_gib": memory.used / 2**30,
                        "free_gib": memory.free / 2**30,
                        "power_w": nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                        "processes": [
                            {"pid": p.pid, "used_gib": p.usedGpuMemory / 2**30, "training": p.pid in active}
                            for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        ],
                    }
                )
            output.write(
                json.dumps(
                    {"timestamp": time.time(), "logical_cpus": psutil.cpu_count(), "cpu": cpu_rows, "gpu": gpu_rows}
                )
                + "\n"
            )
            time.sleep(2)


if __name__ == "__main__":
    main()
