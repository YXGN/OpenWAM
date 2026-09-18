#!/usr/bin/env bash
# Bounded diagnostic; never resumes/writes production checkpoints.
set -euo pipefail
if (( $# > 1 )) || [[ ! "${1:-32}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'Usage: bash scripts/validate_nero_gpu01.sh [batch_size]; GPU AdamW only' >&2
    exit 2
fi
cd "$(dirname "$0")/.."
source scripts/activate_openwam.sh
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 WANDB_MODE=disabled NCCL_DEBUG=WARN
batch="${1:-32}"
folder="outputs/nero_gpu01_validation/batch${batch}"
mkdir -p "$folder"
"$OPENWAM_PYTHON" scripts/monitor_nero_resources.py --gpus 0,1 --seconds 1800 \
    --output "$folder/resources.jsonl" > "$folder/monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
set +e
"$OPENWAM_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_nero_monitored.py --config-name=train_nero_4gpu \
    training.resume_ckpt_path=null ++training.resume_data_step_offset=0 \
    training.finetune_ckpt_path=outputs/nero_multitask_6gpu/2026-09-17_18-22-36/checkpoint_step_4000.safetensors \
    training.output_path="$folder/runs" training.batch_size="$batch" \
    training.offload_optimizer_device=none training.gradient_accumulation_steps=1 \
    training.num_epochs=null training.max_steps=16 training.lr_scheduler=null \
    training.save_steps=0 training.save_full_states_for_resume=false \
    > "$folder/train.log" 2>&1
result=$?
printf '%s\n' "$result" > "$folder/exit_code"
exit "$result"
