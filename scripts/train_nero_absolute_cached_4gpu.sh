#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate_openwam.sh
if [[ "${1:-}" == --print-config ]]; then
    shift
    exec python scripts/train.py --config-name=train_nero_absolute_cached_4gpu --cfg job --resolve "$@"
fi
: "${CUDA_VISIBLE_DEVICES:?Set the four physical GPU indices explicitly before launch}"
# This is a fresh absolute experiment, not a continuation of a relative W&B run.
unset WANDB_RUN_ID WANDB_RESUME
python - <<'PY'
import torch
assert torch.cuda.device_count() == 4, 'Exactly four visible GPUs required'
for index in range(4):
    with torch.cuda.device(index):
        free, _ = torch.cuda.mem_get_info()
        assert free >= 77 * 2**30, f'GPU {index} needs at least 77 GiB free for B64'
PY
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_nero_monitored.py --config-name=train_nero_absolute_cached_4gpu "$@"
