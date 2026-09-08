"""Explicit DeltaNet dispatch and bounded, real-window CUDA parity/timing gate."""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import shutil
import statistics
import sysconfig
import time
from pathlib import Path

import torch


@functools.lru_cache(maxsize=1)
def check_cuda_toolchain():
    """Fail before loading weights if the compiler path cannot run Triton."""
    if not torch.cuda.is_available():
        raise RuntimeError("FLA profiling/training requires an NVIDIA CUDA device")
    header = Path(sysconfig.get_path("include")) / "Python.h"
    if not header.is_file():
        raise RuntimeError(f"missing {header}; use UV_MANAGED_PYTHON=1 uv run ...")
    if not (shutil.which("gcc") or shutil.which("clang")):
        raise RuntimeError("Triton requires a C compiler (gcc or clang)")
    import triton
    from triton.runtime import driver
    target = driver.active.get_current_target()
    if target.backend != "cuda":
        raise RuntimeError(f"expected CUDA Triton target, got {target}")
    result = {"python_header": str(header), "triton": triton.__version__,
              "target": str(target), "torch": torch.__version__}
    print("CUDA_TOOLCHAIN " + json.dumps(result), flush=True)
    return result


def resolve_delta_backend(model_type, device_type, requested, amp_dtype="none"):
    if requested not in {"auto", "torch", "fla"}:
        raise ValueError(f"unknown DeltaNet backend: {requested}")
    if model_type != "qwen3_5":
        if requested != "auto":
            raise ValueError("explicit DeltaNet backend requires a Qwen3.5-family model")
        return "auto"
    # The validated FLA path uses BF16 forwards, including activation capture.
    # Preserve full-precision runs rather than silently changing their precision.
    fast = device_type == "cuda" and amp_dtype == "bfloat16"
    backend = ("fla" if fast else "torch") if requested == "auto" else requested
    if backend == "fla" and device_type != "cuda":
        raise ValueError("FLA requires CUDA; use the torch backend on CPU/MPS")
    if backend == "fla" and amp_dtype != "bfloat16":
        raise ValueError("FLA training requires --amp-dtype bfloat16; use torch for FP32 forwards")
    return backend


def use_gpu_window_cache(device_type, requested, needed_bytes, free_bytes, reserve_bytes=16 * 1024**3):
    """Auto skips the cache on smaller devices; explicit requests fail loudly."""
    if requested is False:
        return False
    supported = device_type == "cuda" and free_bytes >= needed_bytes + reserve_bytes
    if requested is True and not supported:
        raise RuntimeError("GPU window cache requires CUDA and enough memory plus 16 GiB workspace")
    return supported


def select_delta_kernel(backend: str):
    if backend == "fla":
        check_cuda_toolchain()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    if backend == "torch":
        function = inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
        # Keep the original fallback across repeated backend switches.
        function = getattr(modeling, "_tq_torch_delta_reference", function)
        modeling._tq_torch_delta_reference = function
    elif backend == "fla":
        if not hasattr(modeling, "_tq_torch_delta_reference"):
            modeling._tq_torch_delta_reference = inspect.unwrap(
                modeling.torch_chunk_gated_delta_rule)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        accepted = tuple(inspect.signature(chunk_gated_delta_rule).parameters)

        @functools.wraps(chunk_gated_delta_rule)
        def function(*args, **kwargs):
            # Same keyword filtering as the pinned Transformers dispatch decorator.
            return chunk_gated_delta_rule(
                *args, **{key: value for key, value in kwargs.items() if key in accepted})
    else:
        raise ValueError(backend)
    modeling.torch_chunk_gated_delta_rule = function
    print(f"DELTA_KERNEL {backend}: {function.__module__}.{function.__name__}", flush=True)
    return function


def compare_vectors(actual, reference):
    a, b = actual.detach().float().flatten(), reference.detach().float().flatten()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise FloatingPointError("non-finite parity vector")
    norm = b.norm()
    return {
        "relative_l2": float((a - b).norm() / norm.clamp_min(1e-20)),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        if float(norm) > 1e-20 else (1.0 if torch.equal(a, b) else 0.0),
        "max_abs": float((a - b).abs().max()),
    }


def main():
    import numpy as np

    from ternary_quench.train import (
        QuantLinear,
        autocast_context,
        capture_layer_inputs,
        kwargs_for_layer,
        load_training_model,
        resolve_decoder_layout,
        window_loss,
        wrap_layer,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    toolchain = check_cuda_toolchain()
    select_delta_kernel("fla")
    calibration = np.load(args.calib, mmap_mode="r")
    if (calibration.ndim != 2 or calibration.shape[0] < args.batch_size
            or not np.issubdtype(calibration.dtype, np.integer)):
        raise ValueError("calibration must be integer [rows, tokens] with at least one full batch")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    select_delta_kernel("torch")
    model = load_training_model(args.model, torch.bfloat16).eval().to(device)
    model.requires_grad_(False)
    layout = resolve_decoder_layout(model)
    ids = torch.from_numpy(calibration[:args.batch_size].copy()).long()
    with torch.no_grad():
        inputs, kwargs = capture_layer_inputs(model, layout.layers, ids, device,
                                              model_type=layout.model_type)
    model.cpu()
    layers = layout.layers[:4].to(device=device, dtype=torch.float32)
    torch.cuda.empty_cache()

    def forward(x):
        with autocast_context(device, torch.bfloat16):
            for layer in layers:
                out = layer(x, **kwargs_for_layer(layer, kwargs))
                x = out[0] if isinstance(out, tuple) else out
        return x

    with torch.no_grad():
        target = forward(inputs.to(device)).float().detach()
    for layer in layers:
        wrap_layer(layer, 128, 64, 30, ste="tanh", target_suffixes=layout.target_suffixes)
    params = [p for layer in layers for p in layer.parameters() if p.requires_grad]
    gpu_inputs = inputs.to(device)
    cpu_target = target.cpu()
    report = {"shape": list(inputs.shape), "modules": sum(isinstance(m, QuantLinear)
              for layer in layers for m in layer.modules()), "device": torch.cuda.get_device_name(),
              "comparisons": [], "timing": {}}
    report.update(model=args.model, calib_sha256=hashlib.sha256(
        Path(args.calib).read_bytes()).hexdigest(), seed=args.seed, toolchain=toolchain,
        scope="first four layers; forward/backward only, not whole-training wall time")

    def step(backend, gpu_cache):
        select_delta_kernel(backend)
        for p in params:
            p.grad = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        x = gpu_inputs if gpu_cache else inputs.to(device)
        y = target if gpu_cache else cpu_target.to(device)
        out = forward(x)
        loss = window_loss(out, y, huber_delta=0.5, center=False)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        # Compact trained-parameter gradients only; no full model copies.
        grads = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])
        return out.detach().float(), grads, float(loss.detach()), elapsed

    # Both early smooth and near-boundary sharp quantization, same weights/data.
    for progress in (1 / 60, 0.8):
        for layer in layers:
            for module in layer.modules():
                if isinstance(module, QuantLinear):
                    module.progress = progress
        old_out, old_grad, old_loss, _ = step("torch", False)
        new_out, new_grad, new_loss, _ = step("fla", True)
        row = {"progress": progress, "output": compare_vectors(new_out, old_out),
               "gradient": compare_vectors(new_grad, old_grad),
               "loss_relative": abs(new_loss - old_loss) / max(abs(old_loss), 1e-20)}
        report["comparisons"].append(row)
        print("PARITY " + json.dumps(row), flush=True)
        del old_out, old_grad, new_out, new_grad
    for backend, cache in (("torch", False), ("fla", True)):
        times = []
        for _ in range(4):
            out, grads, _, elapsed = step(backend, cache)
            times.append(elapsed)
            del out, grads
        report["timing"][backend] = statistics.median(times[1:])
    report["speedup"] = report["timing"]["torch"] / report["timing"]["fla"]
    report["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report["pass"] = all(r["output"]["relative_l2"] <= 0.02
                         and r["gradient"]["relative_l2"] <= 0.05
                         and r["gradient"]["cosine"] >= 0.99
                         and r["loss_relative"] <= 0.03 for r in report["comparisons"])
    report["pass"] = report["pass"] and report["speedup"] >= 1.15
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print("OPTIMIZATION_GATE " + json.dumps(report), flush=True)
    if not report["pass"]:
        raise RuntimeError("optimized backend failed numerical or speed gate; full run blocked")


if __name__ == "__main__":
    main()
