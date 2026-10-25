# Copyright 2026 Penk Chen <penkia@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np

QK2_0 = 128
MLX_CODES_PER_WORD = 16
MLX_BASE3_CODES_PER_WORD = 20  # 3^20 = 3_486_784_401 < 2^32


def pack_mlx_base3(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack ternary codes {0,1,2} into uint32 words using base-3 encoding.

    Packs flat across the full weight matrix (not per group) to avoid per-group
    padding waste. Returns (packed_1d_uint32, original_n_codes).
    """
    flat = codes.ravel()
    n = len(flat)
    pad = (MLX_BASE3_CODES_PER_WORD - n % MLX_BASE3_CODES_PER_WORD) % MLX_BASE3_CODES_PER_WORD
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    if flat.size and (flat.min() < 0 or flat.max() > 2):
        raise ValueError("base-3 codes must be in {0, 1, 2}")
    c = flat.astype(np.uint32).reshape(-1, MLX_BASE3_CODES_PER_WORD)
    powers = np.array([3 ** i for i in range(MLX_BASE3_CODES_PER_WORD)], dtype=np.uint32)
    packed = (c * powers).sum(axis=1)
    return packed, n


def unpack_mlx_base3(packed: np.ndarray, n_weights: int, rows: int) -> np.ndarray:
    """Unpack base-3 uint32 words back to standard MLX 2-bit format.

    Reverses pack_mlx_base3, then re-packs with pack_mlx_2bit so the result
    is a drop-in replacement for the weight tensor in a standard QuantizedLinear.

    Args:
        packed:    1-D uint32 array produced by pack_mlx_base3.
        n_weights: original number of codes (before padding), equals rows * cols.
        rows:      out_features dimension of the weight matrix.

    Returns:
        2-D uint32 array of shape (rows, cols // 16) — standard MLX 2-bit layout.
    """
    words = packed.copy()
    digits = []
    for _ in range(MLX_BASE3_CODES_PER_WORD):
        digits.append(words % 3)
        words //= 3
    # digits[i] shape (n_words,): the i-th code from every word.
    # Interleave so flat layout matches original: [w0c0, w0c1, ..., w0c19, w1c0, ...]
    codes = np.stack(digits, axis=1).ravel()[:n_weights].astype(np.uint8)
    cols = n_weights // rows
    return pack_mlx_2bit(codes.reshape(rows, cols))


def pack_mlx_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 2-bit codes into MLX uint32 words."""
    rows, cols = codes.shape
    if cols % MLX_CODES_PER_WORD:
        raise ValueError(f"{cols} columns is not a multiple of {MLX_CODES_PER_WORD}")
    if codes.size and (codes.min() < 0 or codes.max() > 3):
        raise ValueError("2-bit codes must be in [0, 3]")
    words = codes.astype(np.uint32).reshape(
        rows, cols // MLX_CODES_PER_WORD, MLX_CODES_PER_WORD
    )
    shifts = np.arange(MLX_CODES_PER_WORD, dtype=np.uint32) * 2
    return np.bitwise_or.reduce(words << shifts, axis=2).astype(np.uint32)
