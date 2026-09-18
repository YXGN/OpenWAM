#!/usr/bin/env bash
# Usage: bash scripts/train_nero_8gpu.sh
# Uses the project environment and local official Foundation by default.
# --print-config resolves the recipe without torchrun or GPU allocation.
# --benchmark runs 100 diagnostic updates and writes throughput/memory metrics.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${OPENWAM_PYTHON:-}" ]]; then
    source scripts/activate_openwam.sh
fi
nero_python="$OPENWAM_PYTHON"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
nero_entry=scripts/train.py
nero_overrides=()

if [[ "${1:-}" == "--print-config" ]]; then
    shift
    exec "$nero_python" scripts/train.py --config-name=train_nero_8gpu --cfg job --resolve "$@"
fi

if [[ "${1:-}" == "--benchmark" ]]; then
    shift
    nero_entry=scripts/benchmark_nero_8gpu.py
    nero_overrides=(training.debug=false training.num_epochs=null training.max_steps=100
        training.lr_scheduler=null training.save_steps=0 training.save_full_states_for_resume=false
        training.output_path=outputs/nero_multitask_8gpu_benchmark)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
"$nero_python" - <<'PY'
import torch

if torch.cuda.device_count() != 8:
    raise SystemExit("This recipe requires exactly 8 visible CUDA devices.")
for index in range(8):
    with torch.cuda.device(index):
        free, total = torch.cuda.mem_get_info()
        if total < 79 * 2**30 or free < 70 * 2**30:
            raise SystemExit(
                f"CUDA device {index}: {free / 2**30:.1f}/{total / 2**30:.1f} GiB free; "
                "the 8 x 80GB recipe needs at least 70 GiB free on every device. "
                "Existing processes were not modified."
            )
print("Eight 80GB-class devices passed the free-memory check.")
PY

# Python module launch avoids stale torchrun console-script shebangs.
exec "$nero_python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 \
    "$nero_entry" --config-name=train_nero_8gpu "$@" "${nero_overrides[@]}"
