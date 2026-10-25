#!/usr/bin/env bash
# Export a trained ternary checkpoint to base-3 packed MLX format, then expand
# to standard 2-bit for inference.
#
# This is the same as the standard MLX export (ternary-quench-export-mlx) with
# --base3 added, followed by ternary-quench-expand-base3. Base-3 is a storage-
# only optimisation (1.6 bpw vs 2.0 bpw, ~20% smaller weight tensors); the
# model cannot be run directly from the base-3 directory and must be expanded
# back to standard 2-bit before inference.
#
# Note: the default paths below use Qwen3-1.7B trained with reduced calibration
# params (nsamples=128, seqlen=1024) to fit a 24 GB unified memory machine. The
# goal was to demonstrate and verify base-3 compaction; for production quality
# use the full paper params (nsamples=512, seqlen=2048) from qwen3.8-27b recipe.
#
# Inputs (must exist before running):
#   $TERNARY_PREFIX  — ternary.pt from the training recipe
#   $BASE_MLX        — 4-bit affine-quantized MLX base (mlx_lm.convert --q-bits 4)
set -euo pipefail

TERNARY_PREFIX="${TERNARY_PREFIX:-out/qwen3-1.7b/ternary.pt}"
BASE_MLX="${BASE_MLX:-out/qwen3-1.7b/mlx-4bit}"
BASE3_DIR="${BASE3_DIR:-out/qwen3-1.7b/mlx-ternary-base3}"
EXPANDED_DIR="${EXPANDED_DIR:-out/qwen3-1.7b/mlx-ternary-base3-expanded}"

# Step 1: export ternary weights with base-3 packing
uv run ternary-quench-export-mlx \
  --ternary-prefix "$TERNARY_PREFIX" \
  --base "$BASE_MLX" \
  --out "$BASE3_DIR" \
  --base3

# Step 2: expand base-3 → standard 2-bit (CPU, ~1 sec)
uv run ternary-quench-expand-base3 \
  --model "$BASE3_DIR" \
  --out "$EXPANDED_DIR"
