"""Roundtrip test: pack to base-3, unpack to 2-bit, verify bit-exact match."""

from __future__ import annotations

import numpy as np
import pytest

from ternary_quench.packing import pack_mlx_2bit, pack_mlx_base3, unpack_mlx_base3


def _reference_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack {0,1,2} codes directly to 2-bit MLX format (the ground truth)."""
    return pack_mlx_2bit(codes.astype(np.uint8))


@pytest.mark.parametrize("rows,cols", [
    (2, 128),    # smallest valid group-aligned matrix
    (4, 256),    # 1024 weights — 1024 % 20 != 0, exercises padding
    (8, 512),    # larger, also unaligned (4096 % 20 != 0)
    (16, 128),   # many rows, one group per row
])
def test_base3_roundtrip(rows, cols):
    rng = np.random.default_rng(42)
    codes = rng.integers(0, 3, size=(rows, cols), dtype=np.uint8)

    packed, n = pack_mlx_base3(codes)
    assert n == rows * cols

    result = unpack_mlx_base3(packed, n, rows)
    expected = _reference_2bit(codes)

    assert result.shape == expected.shape, f"shape mismatch: {result.shape} vs {expected.shape}"
    np.testing.assert_array_equal(result, expected)


def test_all_zeros():
    codes = np.zeros((4, 128), dtype=np.uint8)
    packed, n = pack_mlx_base3(codes)
    result = unpack_mlx_base3(packed, n, 4)
    np.testing.assert_array_equal(result, _reference_2bit(codes))


def test_all_twos():
    codes = np.full((4, 128), 2, dtype=np.uint8)
    packed, n = pack_mlx_base3(codes)
    result = unpack_mlx_base3(packed, n, 4)
    np.testing.assert_array_equal(result, _reference_2bit(codes))
