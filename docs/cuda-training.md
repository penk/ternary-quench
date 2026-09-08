# CUDA training

Fast DeltaNet kernels are the default for CUDA BF16-AMP training on Qwen3.5-family
models, as used by the shipped recipe. Linux installations include pinned
`fla-core==0.5.2`. FP32 forwards and CPU/MPS retain the PyTorch path, and
Qwen3's ordinary attention path is unchanged.

The Qwen3.8 recipe selects uv-managed Python (including development headers)
and runs a real-window profile before training. Linux needs a C compiler and
the standard `timeout` command. Missing headers, unavailable CUDA kernels, or
failed numerical/speed checks stop the recipe instead of silently running the
slow path. No data, epochs, batch sizes, learning rates or loss terms change.

## Check another model or machine

```sh
UV_MANAGED_PYTHON=1 uv run ternary-quench-profile \
  --model /path/to/pinned-upstream-model \
  --calib out/calibration/mixed-agentic10-512x2048.npy \
  --batch-size 3 --report results/cuda-profile.json
```

Use a local, revision-pinned model directory for reproducibility. The profiler
compares the first four decoder layers and their actual ternary wrappers at
early and boundary soft stages. It requires finite outputs and gradients,
output relative L2 difference <=2%, trained-gradient relative L2 <=5% and
cosine >=0.99, reconstruction-loss difference <=3%, and warmed forward/backward
speedup >=1.15. Failure exits nonzero; the profiler never starts training itself.
The recipe gives the whole profile command a 15-minute timeout.

These checks allow BF16 rounding differences. They do not establish identical
optimizer trajectories or behavioral quality. Timing excludes optimizer updates,
diagnostics, full-corpus cache allocation, checkpoints and uploads. Use actual
training epochs and later windows to estimate an entire run.

## Memory and explicit overrides

GPU caching is automatic when free memory covers both arrays plus a 16 GiB
workspace reserve. It keeps the same FP32 values and CPU-seeded row ordering.
The reserve is a guard, not proof that every model/window will fit. Otherwise,
training logs the reason and retains the CPU cache without changing batches.

For 512 rows of 2,048 tokens, both caches occupy 32 GiB at hidden size 4,096
(9B), or 40 GiB at hidden size 5,120 (27B). Active weights and working memory
are additional. The profiler uses one batch; it does not validate allocation
of these full-corpus caches.

For diagnostics, the trainer still accepts `--delta-kernel torch` to force the
reference path, `--delta-kernel fla` to require FLA explicitly, and
`--no-window-cache-gpu` to retain CPU arrays. `--window-cache-gpu` explicitly
requires the cache and fails if the memory guard cannot be met. These are
override controls, not flags required to obtain the faster default.

Kernel policy is part of new checkpoint recipes. A legacy checkpoint cannot
silently resume under FLA; use the original reference backend explicitly or
start a fresh experiment. Cache placement is operational and is not part of
the optimizer identity. No existing weights are rewritten by this patch.

## Measured evidence, 2026-09-08

Qwen3.5-9B on 1×H200, Torch 2.7.1, pinned Transformers commit
`a353632607c59463e6ced86a44c2de3c2cd62d5e`, FLA-core 0.5.2:

| Measurement | PyTorch fallback | FLA + GPU cache |
|---|---:|---:|
| Four-layer forward/backward, batch 3×2,048 | 0.418660 s | 0.192246 s |
| One-layer warmed training epoch | ~30 s | 8.7 s |
| Four-layer training epoch | ~80 s | ~34 s |

The four-layer probe was 2.18× faster; maximum output relative L2 difference
was 0.476%, gradient difference 0.402%, and loss difference 0.0091%. The training
interval was measured between epochs 1 and 11 (26 s → 113 s cumulative).
The optimized full-depth 9B job completed in 9h05m, including setup and final
upload; the trainer reported 8h59m. The unoptimized full run was cancelled,
so there is no completed end-to-end baseline for a whole-run speedup ratio.
These are 9B measurements, not a verified 27B speedup or a behavioral result.
Source run: https://huggingface.co/jobs/penkia/6a9f578ee686246ca69a9971
