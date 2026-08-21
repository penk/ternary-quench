import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ternary_quench.packing import pack_mlx_2bit  # noqa: E402
from ternary_quench.train import (  # noqa: E402
    QuantLinear,
    build_window_scheduler,
    progress_at,
    validate_exported_state,
)


def test_qwen3_1_7b_schedule():
    rounds = build_window_scheduler(28, 4, 2, 4)
    assert len(rounds) == 19
    assert sum(map(len, rounds)) == 64
    assert set().union(*map(set, rounds)) == set(range(28))


def test_boundary_epoch():
    assert progress_at(47, 60) == 0.8
    assert progress_at(48, 60) > 0.8


def test_export_is_hard_ternary():
    base = torch.nn.Linear(128, 2, bias=False)
    quant = QuantLinear(base, group_size=128, r=0, s0=30, gamma=0.8, ste="tanh")
    quant.progress = 0.8
    codes, _ = quant.export()
    assert set(torch.unique(codes).tolist()) <= {-1, 0, 1}


def test_artifact_validation():
    validate_exported_state({
        "model.layers.0.mlp.up_proj": {
            "codes": torch.tensor([[-1, 0, 1]], dtype=torch.int8),
            "scales": torch.ones(1, 1),
        }
    })


def test_mlx_packing():
    codes = np.array([[0, 1, 2, 3] * 4], dtype=np.uint8)
    packed = pack_mlx_2bit(codes)
    expected = sum(int(code) << (2 * i) for i, code in enumerate(codes[0]))
    assert packed.shape == (1, 1)
    assert int(packed[0, 0]) == expected
