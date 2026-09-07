---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - ollama
  - llama.cpp
  - qwen3.8
  - ternary
  - quantization
---

# TernaryQuench Qwen3.8-27B — GGUF

[TernaryQuench](https://github.com/penk/ternary-quench) gives you the complete
recipe for building your own ternary language models, from generating
calibration data to exporting the file you run.

This release compresses [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for local inference. The calibration tools, resumable trainer, exporters,
and evaluation code are open source, so you can inspect the model's recipe
and build your own.

## Get started

With [Ollama](https://ollama.com/download):

```sh
ollama run hf.co/penkia/TernaryQuench-Qwen3.8-27B-GGUF:Q2_K
```

Or with [llama.cpp](https://github.com/ggml-org/llama.cpp):

```sh
llama-cli -hf penkia/TernaryQuench-Qwen3.8-27B-GGUF:Q2_K -cnv -c 32768
```

The download is **10.86 GB**. Allow additional memory for the context cache
and runtime. Tested with Ollama 0.33.3.

For native MLX inference on Apple silicon, see the
[MLX release](https://huggingface.co/penkia/TernaryQuench-Qwen3.8-27B-MLX).

## Evaluation

### Language-model quality

WikiText-2 test-sample results:

| Model | Perplexity ↓ | Second-half perplexity ↓ |
|---|---:|---:|
| Qwen3.8-27B 4-bit | 7.7471 | 6.2370 |
| **TernaryQuench 27B** | **12.8569** | **10.0972** |
| Ternary-Bonsai-27B | 13.2475 | 10.1704 |

Protocol: the first 20,480 tokens of WikiText-2 test, split into 40 independent
512-token chunks. “Second half” scores the last 256 tokens of each chunk.
TernaryQuench was measured directly from the Q2_K release file in llama.cpp;
the reference rows used MLX. [Raw Q2_K results](evaluation/q2k-llama-metal/ppl.json)
include the file hash and per-chunk scores. Bonsai uses Qwen3.6 as its base.

### Five-task zero-shot evaluation

Results measured directly from this Q2_K release file with the LM Evaluation
Harness on the full task sets.

| Model | Mean accuracy ↑ | PIQA | ARC-Easy | ARC-Challenge | HellaSwag | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B BF16 | 74.35 | 81.61 | 72.98 | 58.87 | 82.93 | 75.37 |
| **TernaryQuench 27B** | **72.28** | **78.78** | **78.96** | **55.29** | **73.96** | **74.43** |
| Ternary-Bonsai-27B | 71.91 | 79.27 | 76.22 | 55.63 | 75.98 | 72.45 |

Scores are percentages. The mean weights the five tasks equally, using
length-normalized accuracy where available and standard accuracy for
WinoGrande. Protocol: lm-eval 0.4.7, zero-shot, no chat template, full task
sets. TernaryQuench uses llama.cpp CUDA with FP32 accumulation; reference
rows use native PyTorch.
[Q2_K results and runtime provenance](evaluation/q2k-llama-cuda-f32acc/five-task.json)
· [Reference scores](evaluation/five-task-training-checkpoint.json).

### Runtime checks

The Q2_K file passed direct generation, JSON output without a grammar, and
a complete tool-call round trip in Ollama 0.33.3 and llama.cpp. Its answers
matched the original Q2_0 export on those short tests.

These checks verify installation and basic tool use. They do not establish
reliability on complex, multi-turn agentic work.

## Model

Ternary weights use three values per group: **−scale, 0, +scale**.
This release retains higher precision in the final two decoder layers,
embeddings, and output head:

| Component | Storage |
|---|---|
| Decoder layers 0–61 | 481 trained ternary matrices, packed as Q2_K |
| Decoder layers 62–63 | 15 BF16 matrices |
| Embeddings and output head | Q4_1 |
| Input | Text |

Q2_K stores the ternary weights in a format supported by standard Ollama and
llama.cpp. Packing preserves every sign and zero but rounds group scales,
introducing **1.55% relative RMS weight error** across the converted matrices.
All other tensors are unchanged. The
[conversion manifest](TernaryQuench-Qwen3.8-27B-Q2_K.gguf.json) records the
error, tensor layout, file size, and SHA-256.

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
