"""Validation and coherent packing for self-generated AYOT traces."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def accepted_trace_texts(rows: Iterable[dict[str, Any]]) -> list[str]:
    texts = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if row.get("accepted") is not True:
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"accepted trace {index} has no text")
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    if not texts:
        raise ValueError("no accepted traces")
    return texts


def pack_sequences(
    sequences: list[list[int]],
    *,
    n_samples: int,
    seqlen: int,
    pad_token_id: int,
    seed: int = 2,
    lookahead: int = 64,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Pack complete traces into fixed rows; never splice one trace across rows."""
    if n_samples <= 0 or seqlen <= 0:
        raise ValueError("n_samples and seqlen must be positive")
    if not sequences:
        raise ValueError("no token sequences to pack")
    oversized = [len(seq) for seq in sequences if len(seq) > seqlen]
    if oversized:
        raise ValueError(
            f"{len(oversized)} traces exceed seqlen={seqlen}; longest={max(oversized)}"
        )

    rng = random.Random(seed)
    pending = [list(seq) for seq in sequences]
    rng.shuffle(pending)
    rows: list[list[int]] = []
    used_traces = used_tokens = pad_tokens = 0
    while pending and len(rows) < n_samples:
        row: list[int] = []
        while pending:
            remaining = seqlen - len(row)
            candidates = [
                (len(seq), index)
                for index, seq in enumerate(pending[:lookahead])
                if len(seq) <= remaining
            ]
            if not candidates:
                break
            _, index = max(candidates)
            seq = pending.pop(index)
            row.extend(seq)
            used_traces += 1
            used_tokens += len(seq)
        padding = seqlen - len(row)
        row.extend([pad_token_id] * padding)
        pad_tokens += padding
        rows.append(row)

    if len(rows) < n_samples:
        available = len(rows) * seqlen - pad_tokens
        required = n_samples * seqlen
        raise ValueError(
            f"accepted traces provide {available} packed tokens, insufficient for "
            f"{n_samples}x{seqlen}={required}; generation must produce more unique data"
        )
    total = n_samples * seqlen
    metrics = {
        "samples": n_samples,
        "seqlen": seqlen,
        "input_traces": len(sequences),
        "used_traces": used_traces,
        "unused_traces": len(pending),
        "content_tokens": used_tokens,
        "padding_tokens": pad_tokens,
        "padding_fraction": pad_tokens / total,
        "seed": seed,
        "lookahead": lookahead,
    }
    return rows, metrics
