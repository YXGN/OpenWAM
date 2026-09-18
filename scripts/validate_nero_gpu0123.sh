#!/usr/bin/env bash
# Four-GPU timing validation; production checkpoints remain unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_openwam.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 WANDB_MODE=disabled NCCL_DEBUG=WARN
# Existing small processes are allowed; the launcher still checks free memory.
unset OPENWAM_REQUIRE_IDLE
folder="outputs/nero_gpu0123_validation/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$folder"
printf '%s\n' "$folder" > outputs/nero_gpu0123_validation/latest_run
"$OPENWAM_PYTHON" scripts/monitor_nero_resources.py --gpus 0,1,2,3 --seconds 1800 \
    --output "$folder/resources.jsonl" > "$folder/monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
set +e
bash scripts/train_nero_4gpu.sh \
    training.resume_ckpt_path=null training.resume_data_step_offset=0 \
    training.finetune_ckpt_path=outputs/nero_multitask_6gpu/2026-09-17_18-22-36/checkpoint_step_4000.safetensors \
    training.output_path="$folder/runs" training.batch_size=32 \
    training.num_epochs=null training.max_steps=24 training.lr_scheduler=null \
    training.save_steps=0 training.save_full_states_for_resume=false \
    > "$folder/train.log" 2>&1
result=$?
printf '%s\n' "$result" > "$folder/exit_code"
exit "$result"
