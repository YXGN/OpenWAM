#!/usr/bin/env bash
# Fresh target-server training; run inside tmux after validation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_openwam.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE=online
export WANDB_ENTITY=xxxwwwzzz77-shenzhen
unset WANDB_RUN_ID WANDB_RESUME
python - <<'PY'
import shutil
import pynvml as nvml
import wandb

assert wandb.Api(timeout=20).viewer.username == "xxxwwwzzz77", "W&B account changed; verify login"
nvml.nvmlInit()
try:
    assert nvml.nvmlDeviceGetCount() >= 8, "Eight GPUs are required"
    for index in range(8):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        assert not nvml.nvmlDeviceGetComputeRunningProcesses(handle), f"GPU {index} is occupied"
        assert nvml.nvmlDeviceGetMemoryInfo(handle).free >= 78 * 2**30, f"GPU {index} has insufficient free memory"
finally:
    nvml.nvmlShutdown()
assert shutil.disk_usage("/data3").free >= 270 * 2**30, "Need space for old and new full checkpoints"
assert shutil.disk_usage("/data2").free >= 30 * 2**30, "Need space for the step-4000 weights archive"
PY
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_nero_monitored.py --config-name=train_nero_target_8gpu \
    project.wandb.entity=xxxwwwzzz77-shenzhen "$@"
