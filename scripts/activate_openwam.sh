#!/usr/bin/env bash
# Source this file: source scripts/activate_openwam.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source scripts/activate_openwam.sh" >&2
    exit 1
fi

_openwam_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$_openwam_root/.venv/bin/python" ]]; then
    echo "Missing project environment; run bash scripts/setup_openwam_env.sh first." >&2
    return 1
fi
source "$_openwam_root/.venv/bin/activate"
export OPENWAM_PYTHON="$_openwam_root/.venv/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$_openwam_root"
export OPENWAM_FOUNDATION_DIR="${OPENWAM_FOUNDATION_DIR:-$_openwam_root/models/OpenWAM-Alpha-Pretrain-Foundation-Model}"
export UV_CACHE_DIR="$_openwam_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$_openwam_root/.cache/python"
export TORCH_EXTENSIONS_DIR="$_openwam_root/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$_openwam_root/.cache/torchinductor"
export TRITON_CACHE_DIR="$_openwam_root/.cache/triton"
export CUDA_CACHE_PATH="$_openwam_root/.cache/cuda"
export HF_HOME="$_openwam_root/.cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
# CUDA runtime is installed with PyTorch; this toolkit is for optional JIT ops.
if [[ -d "${OPENWAM_CUDA_HOME:-/usr/local/cuda-12.4}" ]]; then
    export CUDA_HOME="${OPENWAM_CUDA_HOME:-/usr/local/cuda-12.4}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi
unset _openwam_root
