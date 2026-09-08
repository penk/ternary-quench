from types import SimpleNamespace

import pytest
import torch

from ternary_quench.performance import (
    compare_vectors,
    resolve_delta_backend,
    select_delta_kernel,
    use_gpu_window_cache,
)
from ternary_quench.train import checkpoint_recipe


def test_fast_cuda_default_and_unchanged_other_architectures():
    assert resolve_delta_backend("qwen3_5", "cuda", "auto", "bfloat16") == "fla"
    assert resolve_delta_backend("qwen3_5", "cuda", "auto") == "torch"
    with pytest.raises(ValueError, match="amp-dtype bfloat16"):
        resolve_delta_backend("qwen3_5", "cuda", "fla")
    assert resolve_delta_backend("qwen3_5", "cuda", "fla", "bfloat16") == "fla"
    for device in ("cpu", "mps"):
        assert resolve_delta_backend("qwen3_5", device, "auto") == "torch"
        with pytest.raises(ValueError):
            resolve_delta_backend("qwen3_5", device, "fla")
    assert resolve_delta_backend("qwen3", "cuda", "auto") == "auto"
    assert resolve_delta_backend("qwen3_5", "cuda", "torch") == "torch"
    with pytest.raises(ValueError):
        resolve_delta_backend("qwen3", "cuda", "fla")


def test_gpu_cache_auto_and_explicit_controls():
    assert use_gpu_window_cache("cuda", None, 32, 48, reserve_bytes=16)
    assert not use_gpu_window_cache("cuda", None, 32, 47, reserve_bytes=16)
    assert not use_gpu_window_cache("mps", None, 0, 100, reserve_bytes=16)
    assert not use_gpu_window_cache("cuda", False, 32, 100, reserve_bytes=16)
    with pytest.raises(RuntimeError):
        use_gpu_window_cache("cuda", True, 32, 47, reserve_bytes=16)


def test_legacy_checkpoint_compatibility_requires_reference_backend():
    old = checkpoint_recipe(SimpleNamespace(seed=2), [[0]])
    reference = checkpoint_recipe(SimpleNamespace(seed=2, delta_kernel="torch",
                                                  window_cache_gpu=None), [[0]])
    fast = checkpoint_recipe(SimpleNamespace(seed=2, delta_kernel="fla",
                                             window_cache_gpu=None), [[0]])
    assert old == reference
    assert fast != reference
    assert fast["delta_kernel"] == "fla"


def test_vector_parity_and_nonfinite():
    ref = torch.tensor([1.0, 2.0, 3.0])
    assert compare_vectors(ref, ref)["relative_l2"] == 0
    assert compare_vectors(ref * 2, ref)["relative_l2"] == pytest.approx(1)
    assert compare_vectors(torch.zeros(3), torch.zeros(3))["cosine"] == 1
    with pytest.raises(FloatingPointError):
        compare_vectors(ref * float("nan"), ref)


def test_explicit_torch_dispatch_survives_switch():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model
    original = model.torch_chunk_gated_delta_rule
    try:
        ref = select_delta_kernel("torch")
        model.torch_chunk_gated_delta_rule = lambda: None
        assert select_delta_kernel("torch") is ref
    finally:
        model.torch_chunk_gated_delta_rule = original
