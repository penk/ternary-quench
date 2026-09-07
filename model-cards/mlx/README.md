---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - qwen3.8
  - ternary
  - quantization
---

# TernaryQuench Qwen3.8-27B — MLX

[TernaryQuench](https://github.com/penk/ternary-quench) gives you the complete
recipe for building your own ternary language models, from generating
calibration data to exporting the file you run.

This release compresses [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for local inference. The calibration tools, resumable trainer, exporters,
and evaluation code are open source, so you can inspect the model's recipe
and build your own.

## Get started

For native MLX inference on macOS:

```sh
pip install mlx-lm
python -m mlx_lm.generate \
  --model penkia/TernaryQuench-Qwen3.8-27B-MLX \
  --prompt "What is 17 + 25?" --max-tokens 128
```

To download the files for another MLX application:

```sh
hf download penkia/TernaryQuench-Qwen3.8-27B-MLX \
  --local-dir models/TernaryQuench-Qwen3.8-27B-MLX
```

The download is **10.53 GB**. Allow additional memory for the context cache
and runtime. For Ollama or llama.cpp, use the
[GGUF release](https://huggingface.co/penkia/TernaryQuench-Qwen3.8-27B-GGUF):

```sh
ollama run hf.co/penkia/TernaryQuench-Qwen3.8-27B-GGUF:Q2_K
```

## Evaluation

### Language-model quality

WikiText-2 test-sample results, measured with MLX:

| Model | Perplexity ↓ | Second-half perplexity ↓ |
|---|---:|---:|
| Qwen3.8-27B 4-bit | 7.7471 | 6.2370 |
| **TernaryQuench 27B** | **12.8492** | **10.0849** |
| Ternary-Bonsai-27B | 13.2475 | 10.1704 |

Protocol: the first 20,480 tokens of WikiText-2 test, split into 40 independent
512-token chunks. “Second half” scores the last 256 tokens of each chunk.
Bonsai uses Qwen3.6 as its base.

### Five-task zero-shot evaluation

Recorded results for the full-ternary training checkpoint, measured with the
LM Evaluation Harness. These are not scores for this MLX artifact with its
higher-precision final two layers.

| Model / checkpoint | Mean accuracy ↑ | PIQA | ARC-Easy | ARC-Challenge | HellaSwag | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B BF16 | 74.35 | 81.61 | 72.98 | 58.87 | 82.93 | 75.37 |
| **TernaryQuench 27B** | **71.64** | **78.51** | **79.29** | **54.01** | **73.48** | **72.93** |
| Ternary-Bonsai-27B | 71.91 | 79.27 | 76.22 | 55.63 | 75.98 | 72.45 |

Scores are percentages. The mean weights the five tasks equally, using
length-normalized accuracy where available and standard accuracy for
WinoGrande. Protocol: lm-eval 0.4.7, zero-shot, no chat template, full task
sets, native PyTorch.
[Recorded scores and provenance](evaluation/five-task-training-checkpoint.json).

### Tool use

In internal tests, this model completed 5/5 repository-inspection runs and
8/24 tasks in a broader task set. The latter used 600-second and 16-record
limits; five failures reached the time cap. An earlier 11-task attempt had
no passes but included timeouts and an environment-setup failure. These
results do not establish reliable general-purpose agentic coding.

The repository-inspection tests used low reasoning effort, a repetition
penalty of 1.05 over 256 tokens, and up to three malformed-tool-call retries
provided by the internal harness. The basic MLX command above does not
include those retry controls.

## Model

Ternary weights use three values per group: **−scale, 0, +scale**.
This release retains higher precision in the final two decoder layers,
embeddings, and output head:

| Component | Storage |
|---|---|
| Decoder layers 0–61 | 481 trained ternary matrices, 2-bit affine packing, group size 128 |
| Decoder layers 62–63 | 15 BF16 matrices |
| Embeddings and output head | 4-bit affine, group size 64 |
| Input | Text |

The tensor files total 10,504,400,856 bytes. [The artifact manifest](artifact.json)
records the module counts, precision settings, and SHA-256 of each shard.

## How it was trained

Starting from Qwen3.8-27B, we trained ternary weights with
[CAT-Q-style reconstruction](https://arxiv.org/abs/2606.26650) and an
[AYOT-inspired](https://arxiv.org/abs/2608.01078) calibration set: 512 sequences
of 2,048 tokens, with 10% of rows drawn from agentic traces. The run took
**55 hours 38 minutes on 1× NVIDIA H200 (141 GB)**.

The [TernaryQuench build guide](https://github.com/penk/ternary-quench#build-from-source)
contains the recipe and commands for generating calibration traces, training,
resuming a run, and exporting your own models.

## License

Apache-2.0. This model is derived from
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), developed by
[Alibaba's Qwen team](https://github.com/QwenLM), under its
[Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE).
