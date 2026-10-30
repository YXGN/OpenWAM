#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# This process only serves action chunks; the existing robot client owns motion.
export CUDA_VISIBLE_DEVICES="${NERO_ATOM_GPU:-2}"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

exec .venv/bin/python -m openwam.deploy.nero_chunk_server \
    --ckpt-dir outputs/nero_atom_4gpu/2026-10-04_15-00-03 \
    --ckpt-name checkpoint_step_6248.safetensors \
    --config configs/deploy_nero_atom_8017.yaml \
    --gripper-close-offset-m 0.002 \
    --device cuda:0 --host 0.0.0.0 --port 8017
