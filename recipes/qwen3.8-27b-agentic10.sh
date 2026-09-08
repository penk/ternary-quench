#!/usr/bin/env bash
set -euo pipefail

# Triton compiles a CUDA helper; managed Python includes its development headers.
export UV_MANAGED_PYTHON=1

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
CALIB="${CALIB:?set CALIB to a mixed 512x2048 calibration .npy}"
OUT="${OUT:-out/qwen3.8-27b/ternary.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-out/qwen3.8-27b/checkpoints}"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then
  RESUME_ARGS=(--resume)
fi

# Check the real architecture and batch before committing to a long run.
# A failed toolchain, numerical, or speed gate prevents training.
timeout 900 uv run ternary-quench-profile \
  --model "$MODEL" --calib "$CALIB" --batch-size 3 --seed 2 \
  --report "$(dirname "$OUT")/cuda-profile.json"

uv run ternary-quench \
  --model "$MODEL" --calib "$CALIB" --out "$OUT" \
  --checkpoint-dir "$CHECKPOINT_DIR" "${RESUME_ARGS[@]}" \
  --epochs 60 --nsamples 512 --seqlen 2048 --batch-size 3 \
  --num-layer 4 --sliding-layer 2 --fill-window-size 4 \
  --group-size 128 --lora-r 64 \
  --lora-lr 0.003 --factor-lr 0.015 \
  --lora-wd 0 --factor-wd 0 --lr-schedule linear \
  --huber-loss-max 1.0 --ste tanh --s0 30 \
  --progressive-ratio 0.8 --hard-gradient-mode boundary \
  --seed 2 --device cuda --base-dtype bfloat16 \
  --amp-dtype bfloat16 --cpu-offload
