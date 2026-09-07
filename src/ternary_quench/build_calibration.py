#!/usr/bin/env python3
"""Build deterministic row-level mixtures of C4 and agentic calibration rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def c4_rows(tokenizer, *, n_samples: int, seqlen: int, seed: int) -> np.ndarray:
    """Reproduce ``tools.catq.train.calibration_batches(..., dataset='c4')``."""
    from datasets import load_dataset

    shard = (
        "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
        "c4-train.00000-of-01024.json.gz"
    )
    dataset = load_dataset("json", data_files={"train": shard}, split="train")
    rng = random.Random(seed)
    rows: list[list[int]] = []
    while len(rows) < n_samples:
        record = dataset[rng.randint(0, len(dataset) - 1)]
        ids = tokenizer(record["text"])["input_ids"]
        if len(ids) < seqlen:
            continue
        start = rng.randint(0, len(ids) - seqlen)
        rows.append(ids[start : start + seqlen])
    return np.asarray(rows, dtype=np.int64)


def trailing_padding(rows: np.ndarray, eos_token_id: int) -> int:
    """Count packed-trace EOS padding, excluding the final trace's own EOS."""
    total = 0
    for row in rows:
        count = 0
        for token in row[::-1]:
            if int(token) != eos_token_id:
                break
            count += 1
        total += max(count - 1, 0)
    return total


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--agentic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=(0.10, 0.30))
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    agentic = np.load(args.agentic, allow_pickle=False)
    if agentic.ndim != 2 or agentic.dtype.kind not in "iu":
        raise ValueError(
            f"bad agentic array: shape={agentic.shape}, dtype={agentic.dtype}"
        )
    n_samples, seqlen = agentic.shape
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    c4 = c4_rows(tokenizer, n_samples=n_samples, seqlen=seqlen, seed=args.seed)

    selector = random.Random(args.seed)
    agentic_order = list(range(n_samples))
    c4_order = list(range(n_samples))
    selector.shuffle(agentic_order)
    selector.shuffle(c4_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ratio in args.ratios:
        if not 0 < ratio < 1:
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
        agentic_count = round(n_samples * ratio)
        c4_count = n_samples - agentic_count
        selected_agentic = agentic[agentic_order[:agentic_count]]
        selected_c4 = c4[c4_order[:c4_count]]
        sources = ["agentic"] * agentic_count + ["c4"] * c4_count
        rows = np.concatenate((selected_agentic, selected_c4))

        final_order = list(range(n_samples))
        random.Random(args.seed + 1).shuffle(final_order)
        rows = rows[final_order]
        sources = [sources[index] for index in final_order]

        percent = round(ratio * 100)
        output = args.output_dir / f"mixed-agentic{percent}-{n_samples}x{seqlen}.npy"
        np.save(output, rows.astype(np.int64, copy=False))
        padding = trailing_padding(selected_agentic, tokenizer.eos_token_id)
        agentic_positions = agentic_count * seqlen
        general_positions = c4_count * seqlen
        agentic_content_tokens = agentic_positions - padding
        manifest = {
            "model": args.model,
            "shape": list(rows.shape),
            "dtype": str(rows.dtype),
            "seed": args.seed,
            "agentic_source": str(args.agentic),
            "agentic_source_sha256": sha256(args.agentic),
            "agentic_rows": agentic_count,
            "c4_rows": c4_count,
            "row_sources": sources,
            "padding_tokens": padding,
            "padding_fraction": padding / rows.size,
            # The reconstruction objective does not mask trailing EOS
            # padding, so every fixed-shape position has equal nominal loss
            # weight.  r_loss is an objective coefficient, not a claim about
            # the resulting gradient magnitude; train.py measures that
            # separately for mixed-domain artifacts.
            "r_row": agentic_count / n_samples,
            "r_content": agentic_content_tokens
            / (agentic_content_tokens + general_positions),
            "r_loss": agentic_positions / rows.size,
            "sha256": sha256(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"{output}: agentic={agentic_count} C4={c4_count} "
            f"padding={manifest['padding_fraction']:.4%} sha256={manifest['sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
