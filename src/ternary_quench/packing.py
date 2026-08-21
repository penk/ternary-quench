# Copyright 2026 Penk Chen <penkia@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import numpy as np

QK2_0 = 128
MLX_CODES_PER_WORD = 16


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
