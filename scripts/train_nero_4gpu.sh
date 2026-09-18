#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_openwam.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_DEBUG=WARN
if [[ "${1:-}" == --print-config ]]; then
    shift
    exec "$OPENWAM_PYTHON" scripts/train.py --config-name=train_nero_4gpu --cfg job --resolve "$@"
fi
"$OPENWAM_PYTHON" - <<'PY'
import os
if os.environ.get('OPENWAM_REQUIRE_IDLE') == '1':
    import pynvml as nvml
    nvml.nvmlInit()
    for index in map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        assert not nvml.nvmlDeviceGetComputeRunningProcesses(handle), f'Physical GPU {index} is occupied'
    nvml.nvmlShutdown()
import torch
assert torch.cuda.device_count() == 4, 'Exactly four visible GPUs required'
for i in range(4):
    with torch.cuda.device(i):
        free, _ = torch.cuda.mem_get_info()
        assert free >= 75 * 2**30, f'GPU {i} needs at least 75 GiB free'
PY
exec "$OPENWAM_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_nero_monitored.py --config-name=train_nero_4gpu "$@"
