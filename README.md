# ternary-quench

A CAT-Q-inspired ternary quantization trainer for Qwen3, with MLX export for
Apple silicon.

The trainer implements learnable modulation, the softened-ternarization relay,
two-stream sliding-window reconstruction, and hard ternary export. Training stops
at the soft-stage boundary. This replaces CAT-Q's unreleased hard-training stage.

This is not official CAT-Q code.

## Install

```sh
uv sync
```

MLX export:

```sh
uv sync --extra mlx
```

## Train

Qwen3-1.7B requires an NVIDIA GPU with BF16 support:

```sh
recipes/qwen3-1.7b.sh
```

The output is `out/qwen3-1.7b/ternary.pt`.

## Export to MLX

```sh
uv run ternary-quench-export-mlx \
  --ternary out/qwen3-1.7b/ternary.pt \
  --base /path/to/Qwen3-1.7B \
  --out out/Qwen3-1.7B-ternary-mlx
```

The export stores ternary `{-1, 0, +1}` codes in MLX affine 2-bit form with
`scale=d` and `bias=-d` for each group of 128 weights.

## Qwen3-1.7B result

| Artifact | PPL | KL(fp‖q) | Top-1 agreement |
| --- | ---: | ---: | ---: |
| ternary-quench boundary export | 38.7528 | 1.2035 | 0.5415 |
| CAT-Q parameters through the same exporter | 34.51 | 1.0157 | 0.5786 |

Training: 512 C4 samples, 2048 tokens, 60-epoch LR plan, boundary at epoch 48,
19 sliding-window rounds, 9,090 seconds on one H200. The exported artifact has
196 quantized linear modules and weighted nonzero density 0.5201.

Artifact metadata and the complete training trace are under
`results/qwen3-1.7b/`.

## References

- [CAT-Q paper](https://arxiv.org/abs/2606.26650)
- [Published CAT-Q artifacts](https://huggingface.co/IntelLabsChina/CAT-Q)
- [BitTern](https://github.com/IntelChina-AI/BitTern) — CAT-Q's released
  inference and export code, Apache-2.0
- [SliderQuant](https://github.com/deep-optimization/SliderQuant) — the
  sliding-layer reconstruction framework CAT-Q is built on, Apache-2.0

## Attribution

Copyright 2026 Penk Chen <penkia@gmail.com>. Licensed under the Apache License,
Version 2.0; see `LICENSE`.

This project is Apache-2.0, and so are both upstream projects. Files containing
derived code carry an Apache-2.0 section 4(b) notice of modification at the top:

- `src/ternary_quench/quantizer.py` — derived from BitTern's
  `projects/cat-q/quantize/quantizer.py`
- `src/ternary_quench/train.py` — `build_window_scheduler` and `huber_delta_at`
  are derived from SliderQuant's `quantize/sliderquant.py`

Neither upstream repository ships a `NOTICE` file or per-file copyright headers,
so there are no further notices to reproduce.
