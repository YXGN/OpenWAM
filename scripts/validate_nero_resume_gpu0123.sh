#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_openwam.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled NCCL_DEBUG=WARN
export OPENWAM_RESUME_AUDIT_DIR="$PWD/outputs/nero_gpu0123_resume_validation"
"$OPENWAM_PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count() == 4
assert all(torch.cuda.mem_get_info(i)[0] >= 75 * 2**30 for i in range(4))
PY
set +e
"$OPENWAM_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/audit_nero_resume.py --config-name=train_nero_4gpu \
    training.resume_ckpt_path=outputs/nero_gpu0123_resume_validation/run \
    training.max_steps=4004 training.save_steps=0 training.save_full_states_for_resume=false \
    > "$OPENWAM_RESUME_AUDIT_DIR/train.log" 2>&1
result=$?
printf '%s\n' "$result" > "$OPENWAM_RESUME_AUDIT_DIR/exit_code"
exit "$result"
