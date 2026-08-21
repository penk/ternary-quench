# Copyright 2026 Penk Chen <penkia@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from SliderQuant by Intel Labs China, which
# is licensed under the Apache License, Version 2.0. CAT-Q is built on
# SliderQuant, so its sliding-window procedure is inherited from that project.
# See https://github.com/deep-optimization/SliderQuant and
# http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE OF MODIFICATION (Apache-2.0 section 4(b)): the following are derived
# from `quantize/sliderquant.py` and have been changed here:
#   * `build_window_scheduler` is a port of its `layer_windows_scheduler`
#     (lines 435-457). Changes: a single `fill_window_size` replaces the separate
#     `fill_start_window_size` / `fill_end_window_size`; `start_len` is clamped
#     at zero; the fill size is clamped to the layer count so truncated smoke
#     runs are well defined; consecutive duplicate windows are removed; and
#     argument validation is added.
#   * `huber_delta_at` reproduces its per-window Huber schedule,
#     `0.1 + r / num_round * huber_loss_max`. The upstream release omits the
#     `huber_loss_max` argparse definition, so it is an explicit argument here.
#
# Other references to SliderQuant in the comments below record *facts* read from
# its source and released YAML -- batch-size learning-rate scaling, zero weight
# decay, FP32 activation caches under BF16 autocast, the two-stream student and
# teacher inputs -- rather than copied code. `calibration_batches` is an
# independent implementation of the standard GPTQ-lineage calibration recipe
# (random document, reject if shorter than the sequence length, random span),
# and `round_ste` in quantizer.py is the conventional BinaryConnect
# straight-through idiom; neither is copied from SliderQuant.
#
# The upstream sources carry no copyright headers, SPDX tags, or NOTICE file,
# so there are no such notices to retain under sections 4(c) and 4(d).

"""Sliding-window ternary calibration for Qwen3."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import TernaryQuantizer


def require_finite_tensor(value: torch.Tensor, name: str) -> None:
    """Fail at the producer of a non-finite tensor, not later as a NaN PPL."""
    finite = torch.isfinite(value)
    if bool(torch.all(finite)):
        return
    bad = int((~finite).sum().detach().cpu())
    total = value.numel()
    raise FloatingPointError(f"{name} contains {bad}/{total} non-finite values")


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    """SliderQuant-style mixed-precision compute with FP32 master parameters.

    The released Qwen3-1.7B config sets ``use_bfloat16: true`` and
    ``fp16_act: false``. SliderQuant moves active windows to FP32, keeps cached
    activations in FP32, and executes forwards under BF16 autocast. That is what
    this context reproduces; no GradScaler is needed for BF16.
    """
    if amp_dtype is None:
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError(f"AMP dtype {amp_dtype} is supported only on CUDA, got {device}")
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def validate_exported_state(exported: dict[str, dict]) -> None:
    """Validate every tensor that will be uploaded in ``ternary.pt``."""
    if not exported:
        raise RuntimeError("refusing to export an empty CAT-Q artifact")
    for module_name, state in exported.items():
        for field, value in state.items():
            if isinstance(value, torch.Tensor):
                require_finite_tensor(value, f"{module_name}.{field}")
            elif isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"{module_name}.{field} is non-finite: {value}")
        codes = state["codes"]
        is_ternary = (codes == -1) | (codes == 0) | (codes == 1)
        if not bool(torch.all(is_ternary)):
            unique = torch.unique(codes).detach().cpu().tolist()
            raise RuntimeError(
                f"{module_name}.codes is not ternary: {unique[:10]}"
            )


class QuantLinear(nn.Module):
    """Linear whose weight is ternarised, with a LoRA update applied first."""

    def __init__(self, base: nn.Linear, group_size: int = 128, r: int = 64,
                 lora_alpha: float = 1.0, s0: float = 30.0, gamma: float = 0.8,
                 init_scale_from_raw_weights: bool = False,
                 ste: str = "round"):
        super().__init__()
        self.register_buffer("weight", base.weight.data.clone())
        self.bias = None if base.bias is None else nn.Parameter(base.bias.data.clone(),
                                                                requires_grad=False)
        out_f, in_f = self.weight.shape
        dev, dt = self.weight.device, self.weight.dtype
        # Build every parameter on the wrapped layer's device: the model is
        # already moved (MPS) by the time we wrap, and freshly created modules
        # would otherwise land on CPU and fail at the first matmul.
        self.quantizer = TernaryQuantizer(
            (out_f, in_f), group_size=group_size, s0=s0, gamma=gamma,
            init_scale_from_raw_weights=init_scale_from_raw_weights, ste=ste).to(dev)
        # When set, forward returns the ORIGINAL fp output: no LoRA, no quantiser.
        # `weight` is the pristine base buffer, so a wrapped window can still
        # produce a full-precision reconstruction target (see fp_target()).
        self.bypass = False
        self.r = r
        if r > 0:
            # A normal-init, B zero: the update starts at zero but has gradient
            # (grad_A is zero only at step 0, then recovers once B moves).
            self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=dt))
            # SliderQuant initializes A exactly like nn.Linear and B at zero.
            # This matters across Qwen's differently shaped projections: a fixed
            # 0.01 std over-initializes the wide MLP matrices relative to fan-in.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
            self.scaling = lora_alpha / r
        self.progress = 0.0

    def effective_weight(self) -> torch.Tensor:
        return self.quantizer(self.merged_weight(), self.progress)

    def merged_weight(self) -> torch.Tensor:
        """Base weight with the LoRA update folded in -- what actually ships."""
        if self.r > 0:
            return self.weight + (self.lora_B @ self.lora_A) * self.scaling
        return self.weight

    def forward(self, x):
        if self.bypass:
            return F.linear(x, self.weight, self.bias)
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def export(self):
        """Final ternary codes + per-group scales, at full hardness."""
        merged = self.merged_weight()
        require_finite_tensor(merged, "merged weight")
        codes, scale = self.quantizer.codes_and_scales(
            merged, progress=1.0, force_hard=True)
        require_finite_tensor(codes, "hard ternary codes")
        require_finite_tensor(scale, "ternary scales")
        is_ternary = (codes == -1) | (codes == 0) | (codes == 1)
        if not bool(torch.all(is_ternary)):
            unique = torch.unique(codes).detach().cpu().tolist()
            raise RuntimeError(f"hard export produced non-ternary codes: {unique[:10]}")
        return codes.to(torch.int8), scale.to(torch.float32)

    @torch.no_grad()
    def export_state(self) -> dict:
        """Export codes, scales, factors, and LoRA parameters."""
        codes, scale = self.export()
        state = {
            "codes": codes,
            "scales": scale,
            "t_scale": self.quantizer.t_scale.detach().cpu(),
            "t_mu": self.quantizer.t_mu.detach().cpu(),
            "t_round": self.quantizer.t_round.detach().cpu(),
        }
        if self.r > 0:
            state["lora_A"] = self.lora_A.detach().cpu()
            state["lora_B"] = self.lora_B.detach().cpu()
            state["lora_scaling"] = float(self.scaling)
        return state


@contextlib.contextmanager
def fp_target(layers):
    """Run `layers` at full precision even where they are already wrapped.

    Needed because windows overlap: with num_layer 4 / sliding_layer 2, layers 2-3
    belong to both window 0 and window 1, so by the time window 1 builds its
    reconstruction target those layers are already ternarised. Regressing onto that
    makes every window after the first fit a target that has absorbed the previous
    window's quantisation error -- the opposite of the intent, and against
    `use_quant_tar_loss: false` in every shipped config.
    """
    touched = [m for layer in layers for m in layer.modules()
               if isinstance(m, QuantLinear)]
    for m in touched:
        m.bypass = True
    try:
        yield
    finally:
        for m in touched:
            m.bypass = False


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def build_window_scheduler(n: int, num_layer: int, sliding_layer: int,
                           fill_window_size: int | None) -> list[list[int]]:
    """SliderQuant's `layer_windows_scheduler` (quantize/sliderquant.py:435-457).

    CAT-Q is built on SliderQuant, and every shipped config sets
    `fill_window_size: 4`. That adds *growing* windows at the start and *shrinking*
    ones at the end, which a bare `range(0, n, stride)` does not:

        n=28, num_layer=4, sliding_layer=2, fill=4  ->  19 rounds
          [0], [0,1], [0,1,2], [0,1,2,3]              (growing fill)
          [2-5], [4-7], ... [22-25]                   (11 middle windows)
          [24-27], [25-27], [26-27], [27]             (shrinking fill)

    Starts are non-decreasing, which is what lets the caller advance the cached
    activations incrementally instead of recomputing a prefix per round.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if num_layer <= 0:
        raise ValueError(f"num_layer must be positive, got {num_layer}")
    if sliding_layer <= 0:
        raise ValueError(f"sliding_layer must be positive, got {sliding_layer}")
    if fill_window_size is not None and fill_window_size < 0:
        raise ValueError(f"fill_window_size must be non-negative, got {fill_window_size}")
    if n == 0:
        return []
    if not fill_window_size:
        rounds = [list(range(s, min(s + num_layer, n))) for s in range(0, n, sliding_layer)]
        return [r for r in rounds if r]

    # SliderQuant assumes a full model (n >> fill). Keeping the same construction
    # while clamping fill to n also makes --max-layers smoke runs well-defined.
    fill_window_size = min(fill_window_size, n)
    start_fill = [list(range(i + 1)) for i in range(fill_window_size)]
    end_fill = [list(range(n - fill_window_size + i, n)) for i in range(fill_window_size)]
    edge_len = max(fill_window_size - sliding_layer, 0)  # their start_len / end_len
    mid_len = n - 2 * edge_len
    mid_round = max(math.ceil((mid_len - num_layer) / sliding_layer) + 1, 0)
    mid = [
        list(range(r * sliding_layer + edge_len,
                   min(r * sliding_layer + num_layer + edge_len, n)))
        for r in range(mid_round)
    ]
    rounds = start_fill + mid + end_fill
    # Small smoke models can make the last growing window equal the first middle
    # or shrinking window. The production n=28 schedule has no duplicates.
    deduplicated = []
    for window in rounds:
        if window and (not deduplicated or window != deduplicated[-1]):
            deduplicated.append(window)
    return deduplicated


def huber_delta_at(round_idx: int, num_rounds: int, huber_loss_max: float) -> float:
    """SliderQuant's per-window Huber schedule.

    Its released trainer uses ``0.1 + r / num_round * huber_loss_max``. The
    ``huber_loss_max`` argparse definition is absent from that partial release,
    so it remains an explicit knob here rather than another hidden assumption.
    """
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    if not 0 <= round_idx < num_rounds:
        raise ValueError(f"round_idx {round_idx} outside [0, {num_rounds})")
    if huber_loss_max < 0:
        raise ValueError(f"huber_loss_max must be non-negative, got {huber_loss_max}")
    return 0.1 + round_idx / num_rounds * huber_loss_max


def wrap_layer(layer: nn.Module, group_size: int, r: int, s0: float,
               gamma: float = 0.8,
               init_scale_from_raw_weights: bool = False,
               ste: str = "tanh") -> dict[str, QuantLinear]:
    """Replace every target nn.Linear in `layer` with a QuantLinear."""
    wrapped: dict[str, QuantLinear] = {}
    for name, module in list(layer.named_modules()):
        if not isinstance(module, nn.Linear) or not name.endswith(TARGET_SUFFIXES):
            continue
        parent = layer
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        ql = QuantLinear(module, group_size=group_size, r=r, s0=s0, gamma=gamma,
                         init_scale_from_raw_weights=init_scale_from_raw_weights,
                         ste=ste)
        setattr(parent, attr, ql)
        wrapped[name] = ql
    return wrapped


def progress_at(epoch: int, total_epochs: int) -> float:
    """CAT-Q's normalized *calibration epoch* state in (0, 1].

    The paper defines a finite sequence of mappings with one sharpness per epoch.
    Holding ``t`` fixed for the whole epoch is material: at 512 samples / batch 9,
    each mapping receives 57 updates.  Advancing it per minibatch instead gives
    every mapping only one update and was the largest remaining mismatch in our
    softened-ternarization port.

    Equation 6 calls ``t=0`` the *pre-trained initialization point* and assigns
    calibration epochs to ``0 < t <= gamma`` and ``gamma < t <= 1``.  Epochs are
    therefore numbered 1..m here, not 0..m-1.  With m=60 and gamma=0.8 this visits
    the boundary exactly at epoch 48, followed by exactly 12 hard epochs.
    """
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}")
    if not 0 <= epoch < total_epochs:
        raise ValueError(f"epoch {epoch} outside [0, {total_epochs})")
    return (epoch + 1) / total_epochs


def batches_per_epoch(n_samples: int, batch_size: int) -> int:
    """SliderQuant/DataLoader semantics: include the final partial batch."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return math.ceil(n_samples / batch_size)


def linear_lr_multiplier(completed_steps: int, total_steps: int) -> float:
    """Transformers/SliderQuant linear schedule with zero warmup."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    return max((total_steps - completed_steps) / total_steps, 0.0)


def _diagnose_nonfinite_window(layers, x: torch.Tensor, *, layer_kwargs: dict,
                               log_prefix: str, step: int) -> None:
    """Re-run a failed batch and name the first bad input, parameter, or layer."""
    require_finite_tensor(x, f"{log_prefix} step {step} input")
    for layer_offset, layer in enumerate(layers):
        for name, parameter in layer.named_parameters():
            require_finite_tensor(
                parameter, f"{log_prefix} step {step} layer {layer_offset} parameter {name}"
            )
        with torch.no_grad():
            out = layer(x, **layer_kwargs)
            x = out[0] if isinstance(out, tuple) else out
        require_finite_tensor(
            x, f"{log_prefix} step {step} layer {layer_offset} output"
        )


@contextlib.contextmanager
def hard_forward(layers):
    """Evaluate the deployed hard ternary forward without mutating parameters."""
    touched = [m for layer in layers for m in layer.modules()
               if isinstance(m, QuantLinear)]
    saved = [(m.quantizer.ste, m.progress) for m in touched]
    for m in touched:
        # round_ste has a hard forward and identity backward. No backward happens
        # in this context; selecting it is simply the cheapest exact deployed path.
        m.quantizer.ste = "round"
        m.progress = 1.0
    try:
        yield
    finally:
        for m, (ste, progress) in zip(touched, saved, strict=True):
            m.quantizer.ste = ste
            m.progress = progress


def centered(residual: torch.Tensor) -> torch.Tensor:
    """Remove the per-channel mean over every non-channel axis."""
    return residual - residual.mean(dim=tuple(range(residual.dim() - 1)), keepdim=True)


def window_loss(out: torch.Tensor, target: torch.Tensor, *, huber_delta: float,
                center: bool) -> torch.Tensor:
    residual = out.float() - target.float()
    if center:
        residual = centered(residual)
    return F.huber_loss(residual, torch.zeros_like(residual), delta=huber_delta)


def logit_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(teacher || student), in nats per scored token."""
    require_finite_tensor(student_logits, "student logits")
    require_finite_tensor(teacher_logits, "teacher logits")
    value = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        F.log_softmax(teacher_logits.float(), dim=-1),
        log_target=True,
        reduction="none",
    ).sum(-1).mean()
    require_finite_tensor(value, "logit KL")
    return value


def gradient_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    """Global L2 norm of an ``autograd.grad`` result without flattening it."""
    return math.sqrt(sum(
        float(grad.detach().float().square().sum())
        for grad in grads if grad is not None
    ))


def snapshot_factor_gradients(params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Copy the globally clipped factor gradients at the soft-stage endpoint.

    CAT-Q section 2.3 says that hard-stage updates of the three modulation factors
    take the gradients computed in the last iteration of the differentiable stage.
    The released paper does not disambiguate a last minibatch from a last epoch.
    This port uses a deterministic, sample-weighted pass over the complete
    calibration set at the final soft parameters, then applies the same global
    clipping rule as ordinary training. That avoids replaying one unusually noisy
    partial minibatch for the entire hard stage.
    """
    frozen = []
    for index, parameter in enumerate(params):
        if parameter.grad is None:
            raise RuntimeError(f"factor parameter {index} has no boundary gradient")
        require_finite_tensor(parameter.grad, f"factor parameter {index} boundary gradient")
        frozen.append(parameter.grad.detach().clone())
    return frozen


def install_factor_gradients(params: list[torch.Tensor],
                             frozen: list[torch.Tensor]) -> None:
    """Install an immutable boundary-gradient snapshot for one optimizer step."""
    if len(params) != len(frozen):
        raise ValueError(f"factor/frozen length mismatch: {len(params)} != {len(frozen)}")
    for index, (parameter, gradient) in enumerate(zip(params, frozen, strict=True)):
        require_finite_tensor(gradient, f"factor parameter {index} replay gradient")
        # clip_grad_norm_ mutates gradients in place. Clone here so no future
        # operation can silently alter the authoritative boundary snapshot.
        parameter.grad = gradient.clone()


def max_parameter_change(params: list[torch.Tensor],
                         reference: list[torch.Tensor]) -> float:
    """Largest absolute parameter change, used to prove LoRA freezes in stage 2."""
    if len(params) != len(reference):
        raise ValueError(f"parameter/reference length mismatch: {len(params)} != {len(reference)}")
    if not params:
        return 0.0
    return max(float((parameter.detach() - before).abs().max())
               for parameter, before in zip(params, reference, strict=True))


def parameter_displacement(params: list[torch.Tensor],
                           reference: list[torch.Tensor]) -> tuple[float, float, float]:
    """Mean absolute, RMS and maximum elementwise displacement from a checkpoint."""
    if len(params) != len(reference):
        raise ValueError(f"parameter/reference length mismatch: {len(params)} != {len(reference)}")
    if not params:
        return 0.0, 0.0, 0.0
    differences = [parameter.detach().float() - before.float()
                   for parameter, before in zip(params, reference, strict=True)]
    count = sum(difference.numel() for difference in differences)
    mean_abs = sum(float(difference.abs().sum()) for difference in differences) / count
    rms = math.sqrt(sum(float(difference.square().sum()) for difference in differences) / count)
    maximum = max(float(difference.abs().max()) for difference in differences)
    return mean_abs, rms, maximum


def normalized_kl_weight(local_norm: float, kl_norm: float, *,
                         target_fraction: float, minimum: float,
                         maximum: float) -> tuple[float, float, float]:
    """Weight KL to a requested fraction of the local gradient magnitude.

    Returns ``(clamped_weight, realized_fraction, candidate_weight)``.  Keeping
    the unclamped candidate visible is essential: the first full-depth KL arm
    silently spent eleven rounds at its 0.1 ceiling and delivered as little as
    7.3% of the requested gradient ratio.  The caller calibrates this at a
    representative endpoint rather than training epoch zero: CAT-Q's soft relay
    is exactly full precision at ``t=0``, where both norms can vanish and their
    ratio is meaningless.
    """
    values = (local_norm, kl_norm, target_fraction, minimum, maximum)
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError(f"non-finite KL normalization input: {values}")
    if target_fraction < 0:
        raise ValueError("target_fraction must be non-negative")
    if minimum < 0 or maximum < minimum:
        raise ValueError(f"invalid KL weight bounds: [{minimum}, {maximum}]")
    if local_norm <= 0 or kl_norm <= 0 or target_fraction == 0:
        return 0.0, 0.0, 0.0
    candidate = target_fraction * local_norm / kl_norm
    weight = min(max(candidate, minimum), maximum)
    return weight, weight * kl_norm / local_norm, candidate


def kl_positions(sequence_length: int, count: int, device) -> torch.Tensor:
    """Deterministic positions spanning the sequence for the vocabulary projection."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if count <= 0:
        raise ValueError("KL position count must be positive")
    count = min(count, sequence_length)
    if count == sequence_length:
        return torch.arange(sequence_length, device=device)
    return torch.linspace(0, sequence_length - 1, steps=count, device=device).long()


def _run_decoder_layers(layers, x: torch.Tensor, layer_kwargs: dict) -> torch.Tensor:
    for layer in layers:
        out = layer(x, **layer_kwargs)
        x = out[0] if isinstance(out, tuple) else out
    return x


def _selected_logits(hidden: torch.Tensor, positions: torch.Tensor,
                     final_norm: nn.Module, lm_head: nn.Module) -> torch.Tensor:
    # Qwen's final RMSNorm is token-local, so selecting before the 151k-vocabulary
    # projection is exact and avoids materialising logits for all 512 positions.
    return lm_head(final_norm(hidden[:, positions])).float()


def calibrate_window_kl_weight(
    layers, suffix, inps, targets, *, layer_kwargs: dict, batch_size: int,
    device, huber_delta: float, center: bool, params: list[torch.Tensor],
    final_norm: nn.Module, lm_head: nn.Module, kl_position_count: int,
    norm_batches: int, target_fraction: float, weight_min: float,
    weight_max: float,
) -> dict[str, float | bool]:
    """Measure a robust per-window KL weight at the final soft relay state.

    Training begins at ``t=0``, an exact fp forward, so calibrating on the literal
    first batch would divide two near-zero gradients.  Probe the same parameters at
    ``t=1`` (the state training is approaching), take the median ratio across a few
    deterministic batches, then restore every module's real progress.
    """
    if norm_batches <= 0:
        raise ValueError("KL normalization needs at least one probe batch")
    mods = [m for layer in layers for m in layer.modules()
            if isinstance(m, QuantLinear)]
    saved_progress = [m.progress for m in mods]
    ratios: list[float] = []
    local_norms: list[float] = []
    kl_norms: list[float] = []
    try:
        for m in mods:
            m.progress = 1.0
        available = min(norm_batches, batches_per_epoch(inps.shape[0], batch_size))
        for probe in range(available):
            start = probe * batch_size
            stop = min(start + batch_size, inps.shape[0])
            x = inps[start:stop].to(device)
            target = targets[start:stop].to(device)
            student_out = _run_decoder_layers(layers, x, layer_kwargs)
            local = window_loss(
                student_out, target, huber_delta=huber_delta, center=center
            )
            local_grads = torch.autograd.grad(
                local, params, retain_graph=True, allow_unused=True
            )

            positions = kl_positions(student_out.shape[1], kl_position_count, device)
            with torch.no_grad(), fp_target(suffix):
                teacher_tail = _run_decoder_layers(suffix, target, layer_kwargs)
                teacher_logits = _selected_logits(
                    teacher_tail, positions, final_norm, lm_head
                )
            with fp_target(suffix):
                student_tail = _run_decoder_layers(suffix, student_out, layer_kwargs)
                student_logits = _selected_logits(
                    student_tail, positions, final_norm, lm_head
                )
            kl = logit_kl(student_logits, teacher_logits)
            kl_grads = torch.autograd.grad(kl, params, allow_unused=True)
            local_norm = gradient_norm(local_grads)
            kl_norm = gradient_norm(kl_grads)
            weight, _, candidate = normalized_kl_weight(
                local_norm, kl_norm, target_fraction=target_fraction,
                minimum=weight_min, maximum=weight_max,
            )
            if weight > 0:
                ratios.append(candidate)
                local_norms.append(local_norm)
                kl_norms.append(kl_norm)
        if not ratios:
            raise RuntimeError(
                "KL normalization produced no non-zero finite gradient pairs"
            )
    finally:
        for m, progress in zip(mods, saved_progress, strict=True):
            m.progress = progress

    # The scale is multiplicative, so use a median in log space. With two probe
    # batches, selecting the ordinary upper median silently chooses the larger
    # weight; the first real smoke produced a 1.86x average contribution that way.
    log_ratios = sorted(math.log(value) for value in ratios)
    mid = len(log_ratios) // 2
    median_log = (log_ratios[mid] if len(log_ratios) % 2
                  else (log_ratios[mid - 1] + log_ratios[mid]) / 2)
    candidate_weight = math.exp(median_log)
    weight = min(max(candidate_weight, weight_min), weight_max)
    weight_clamped = weight != candidate_weight
    realized = [weight * kl_norm / local_norm
                for local_norm, kl_norm in zip(local_norms, kl_norms, strict=True)]
    return {
        "kl_weight": weight,
        "kl_weight_candidate": candidate_weight,
        "kl_weight_clamped": weight_clamped,
        "kl_probe_batches": len(ratios),
        "kl_probe_local_grad": sum(local_norms) / len(local_norms),
        "kl_probe_raw_grad": sum(kl_norms) / len(kl_norms),
        "kl_probe_realized_fraction": sum(realized) / len(realized),
        "kl_probe_realized_min": min(realized),
        "kl_probe_realized_max": max(realized),
        "kl_probe_weight_min": min(ratios),
        "kl_probe_weight_max": max(ratios),
    }


@torch.no_grad()
def reconstruction_loss(layers, inps, targets, *, layer_kwargs: dict,
                        batch_size: int, device, huber_delta: float,
                        center: bool = False, deployed: bool = False,
                        amp_dtype: torch.dtype | None = None) -> dict[str, float]:
    """Window reconstruction plus the error decomposition, at full hardness.

    ``offset_amplitude`` is ``||mean_c d|| * sqrt(N) / ||d||``. Squaring it gives
    the offset's share of residual energy.
    """
    total_objective_loss = 0.0
    total_raw_loss = 0.0
    total_samples = 0
    sum_d: torch.Tensor | None = None
    sq_d = sq_tgt = 0.0
    n_rows = 0
    max_abs = 0.0
    ctx = hard_forward(layers) if deployed else contextlib.nullcontext()
    with ctx:
        for start in range(0, inps.shape[0], batch_size):
            stop = min(start + batch_size, inps.shape[0])
            x = inps[start:stop].to(device)
            tgt = targets[start:stop].to(device)
            with autocast_context(device, amp_dtype):
                for layer in layers:
                    out = layer(x, **layer_kwargs)
                    x = out[0] if isinstance(out, tuple) else out
                objective_loss = window_loss(
                    x, tgt, huber_delta=huber_delta, center=center
                )
                raw_loss = window_loss(x, tgt, huber_delta=huber_delta, center=False)
            require_finite_tensor(objective_loss, "hard objective loss")
            require_finite_tensor(raw_loss, "raw hard reconstruction loss")
            total_objective_loss += float(objective_loss) * (stop - start)
            total_raw_loss += float(raw_loss) * (stop - start)
            total_samples += stop - start
            d = (x.float() - tgt.float()).reshape(-1, x.shape[-1])
            # Accumulate in float64 on the CPU: MPS has no float64 at all, and a
            # float32 running sum over ~250k rows loses the precision this
            # decomposition depends on (offset energy is a small difference of
            # large numbers).
            summed = d.sum(0).cpu().double()
            sum_d = summed if sum_d is None else sum_d + summed
            sq_d += float(d.square().sum().cpu().double())
            sq_tgt += float(tgt.float().square().sum().cpu().double())
            n_rows += d.shape[0]
            max_abs = max(max_abs, float(x.detach().abs().max()))
    mean_c = sum_d / max(n_rows, 1)
    offset_energy = n_rows * float(mean_c.square().sum())
    offset_energy_fraction = offset_energy / sq_d if sq_d else 0.0
    return {
        # Always report the ordinary uncentered Huber loss separately.
        "hard_reconstruction_loss": total_raw_loss / total_samples,
        "hard_objective_loss": total_objective_loss / total_samples,
        "offset_norm": float(mean_c.norm()),
        "offset_amplitude": math.sqrt(offset_energy_fraction),
        "offset_energy_fraction": offset_energy_fraction,
        "noise_rel": math.sqrt(max(sq_d - offset_energy, 0.0) / sq_tgt) if sq_tgt else 0.0,
        "max_abs_activation": max_abs,
    }


def train_window(layers, inps, targets, *, layer_kwargs: dict, epochs: int,
                 batch_size: int, lora_lr: float, factor_lr: float, grad_clip: float,
                 device, log_prefix: str, factor_wd: float = 0.0,
                 lora_wd: float = 0.0, huber_delta: float = 1.0,
                 lr_schedule: str = "linear", center: bool = False,
                 hard_gradient_mode: str = "recompute",
                 hard_stage_abort_ratio: float = 1.5,
                 amp_dtype: torch.dtype | None = None,
                 suffix=(), final_norm: nn.Module | None = None,
                 lm_head: nn.Module | None = None, kl_grad_fraction: float = 0.0,
                 kl_every: int = 4, kl_position_count: int = 128,
                 kl_norm_batches: int = 4, kl_weight_min: float = 0.0,
                 kl_weight_max: float = 0.1) -> dict[str, float]:
    params_lora, params_factor = [], []
    factor_params: dict[str, list[torch.Tensor]] = {
        "scale": [], "mu": [], "round": [],
    }
    mods: list[QuantLinear] = []
    for layer in layers:
        for m in layer.modules():
            if isinstance(m, QuantLinear):
                mods.append(m)
                params_factor += [m.quantizer.t_scale, m.quantizer.t_mu, m.quantizer.t_round]
                factor_params["scale"].append(m.quantizer.t_scale)
                factor_params["mu"].append(m.quantizer.t_mu)
                factor_params["round"].append(m.quantizer.t_round)
                if m.r > 0:
                    params_lora += [m.lora_A, m.lora_B]
    initial_codes = [m.export()[0].cpu() for m in mods]
    all_trainable_params = params_lora + params_factor
    kl_enabled = kl_grad_fraction > 0
    if hard_gradient_mode not in ("recompute", "replay", "boundary"):
        raise ValueError(f"unknown hard-gradient mode: {hard_gradient_mode}")
    if hard_stage_abort_ratio <= 1.0:
        raise ValueError("hard-stage abort ratio must be greater than 1")
    if hard_gradient_mode in ("replay", "boundary"):
        if kl_enabled:
            raise ValueError("paper-mechanism arms must run without KL")
        if not mods or any(m.quantizer.ste != "tanh" for m in mods):
            raise ValueError("paper-mechanism arms require the tanh softened relay")
        gammas = {m.quantizer.gamma for m in mods}
        if len(gammas) != 1:
            raise ValueError(f"window contains inconsistent progressive ratios: {gammas}")
        gamma = gammas.pop()
        soft_epoch_indices = [i for i in range(epochs) if progress_at(i, epochs) <= gamma]
        if not soft_epoch_indices or len(soft_epoch_indices) == epochs:
            raise ValueError(
                f"paper-mechanism arm requires both stages, got epochs={epochs} gamma={gamma}"
            )
        boundary_epoch = soft_epoch_indices[-1]
        boundary_t = progress_at(boundary_epoch, epochs)
        if not math.isclose(boundary_t, gamma, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "epoch schedule does not visit gamma exactly: "
                f"last soft t={boundary_t} gamma={gamma}"
            )
    else:
        gamma = mods[0].quantizer.gamma if mods else 1.0
        boundary_epoch = -1
    if kl_enabled:
        if final_norm is None or lm_head is None:
            raise ValueError("KL training requires the final norm and lm_head")
        if kl_every <= 0:
            raise ValueError("kl_every must be positive")
        if kl_position_count <= 0:
            raise ValueError("kl_position_count must be positive")
        kl_calibration = calibrate_window_kl_weight(
            layers, suffix, inps, targets, layer_kwargs=layer_kwargs,
            batch_size=batch_size, device=device, huber_delta=huber_delta,
            center=center, params=all_trainable_params,
            final_norm=final_norm, lm_head=lm_head,
            kl_position_count=kl_position_count, norm_batches=kl_norm_batches,
            target_fraction=kl_grad_fraction, weight_min=kl_weight_min,
            weight_max=kl_weight_max,
        )
        kl_weight = kl_calibration["kl_weight"]
        clamp_note = " **CLAMPED**" if kl_calibration["kl_weight_clamped"] else ""
        print(
            f"  {log_prefix} KL calibration weight={kl_weight:.4g} "
            f"candidate={kl_calibration['kl_weight_candidate']:.4g}{clamp_note} "
            f"target={kl_grad_fraction:.3g} "
            f"realized={kl_calibration['kl_probe_realized_fraction']:.3g} "
            f"range={kl_calibration['kl_probe_realized_min']:.3g}.."
            f"{kl_calibration['kl_probe_realized_max']:.3g} "
            f"local_grad={kl_calibration['kl_probe_local_grad']:.3g} "
            f"kl_grad={kl_calibration['kl_probe_raw_grad']:.3g} "
            f"probes={int(kl_calibration['kl_probe_batches'])}",
            flush=True,
        )
        if kl_calibration["kl_weight_clamped"]:
            print(
                f"  {log_prefix} WARNING: KL normalization requested "
                f"{kl_calibration['kl_weight_candidate']:.4g}, outside "
                f"[{kl_weight_min:.4g}, {kl_weight_max:.4g}]; applied "
                f"{kl_weight:.4g} and realized only "
                f"{kl_calibration['kl_probe_realized_fraction']:.3g} of target",
                flush=True,
            )
    else:
        kl_weight = 0.0
        kl_calibration = {
            "kl_weight": 0.0,
            "kl_weight_candidate": 0.0,
            "kl_weight_clamped": False,
            "kl_probe_batches": 0,
            "kl_probe_local_grad": 0.0,
            "kl_probe_raw_grad": 0.0,
            "kl_probe_realized_fraction": 0.0,
            "kl_probe_realized_min": 0.0,
            "kl_probe_realized_max": 0.0,
            "kl_probe_weight_min": 0.0,
            "kl_probe_weight_max": 0.0,
        }
    # The released CAT-Q configuration uses zero weight decay for both groups.
    opt = torch.optim.AdamW([
        {"params": params_lora, "lr": lora_lr, "weight_decay": lora_wd},
        {"params": params_factor, "lr": factor_lr, "weight_decay": factor_wd},
    ])
    n = inps.shape[0]
    steps_per_epoch = batches_per_epoch(n, batch_size)
    total = epochs * steps_per_epoch
    if lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda completed: linear_lr_multiplier(completed, total)
        )
    elif lr_schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"unknown LR schedule: {lr_schedule}")
    step = 0
    t0 = time.time()
    grad_sum = {name: 0.0 for name in factor_params}
    grad_first = {name: 0.0 for name in factor_params}
    grad_last = {name: 0.0 for name in factor_params}
    final_soft_loss = math.nan
    final_total_loss = math.nan
    kl_loss_sum = 0.0
    kl_steps = 0
    frozen_factor_grads: list[torch.Tensor] | None = None
    lora_at_boundary: list[torch.Tensor] | None = None
    factors_at_boundary: list[torch.Tensor] | None = None
    boundary_deployed: dict[str, float] = {}
    boundary_full_gradient_norm = 0.0
    replay_steps = 0
    hard_stage_max_loss_ratio = 0.0
    hard_stage_displacement = (0.0, 0.0, 0.0)
    # `boundary` is a diagnostic/training recipe, not gamma=1 under another
    # name. It preserves the paper schedule and LR at gamma, then exports that
    # exact soft-stage solution through the normal hard ternary path. This
    # isolates whether the useful calibration solution exists before the hard
    # stage without letting a known-bad continuation corrupt it.
    epochs_to_run = boundary_epoch + 1 if hard_gradient_mode == "boundary" else epochs
    for epoch in range(epochs_to_run):
        t = progress_at(epoch, epochs)
        replay_hard = hard_gradient_mode == "replay" and epoch > boundary_epoch
        for layer in layers:
            for m in layer.modules():
                if isinstance(m, QuantLinear):
                    m.progress = t
        perm = torch.randperm(n)
        run = 0.0
        run_total = 0.0
        run_kl = 0.0
        epoch_kl_steps = 0
        run_hard_raw = 0.0
        run_hard_samples = 0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            x = inps[idx].to(device)
            with torch.no_grad():
                tgt = targets[idx].to(device)
            forward_context = torch.no_grad() if replay_hard else contextlib.nullcontext()
            with forward_context, autocast_context(device, amp_dtype):
                for layer in layers:
                    out = layer(x, **layer_kwargs)
                    x = out[0] if isinstance(out, tuple) else out
                local_loss = window_loss(x, tgt, huber_delta=huber_delta, center=center)
                raw_hard_loss = (
                    window_loss(x, tgt, huber_delta=huber_delta, center=False)
                    if replay_hard and center else local_loss
                )
                loss = local_loss
                used_kl = kl_enabled and step % kl_every == 0
                if used_kl:
                    positions = kl_positions(x.shape[1], kl_position_count, device)
                    with torch.no_grad(), fp_target(suffix):
                        teacher_tail = _run_decoder_layers(suffix, tgt, layer_kwargs)
                        teacher_logits = _selected_logits(
                            teacher_tail, positions, final_norm, lm_head
                        )
                    with fp_target(suffix):
                        student_tail = _run_decoder_layers(suffix, x, layer_kwargs)
                        student_logits = _selected_logits(
                            student_tail, positions, final_norm, lm_head
                        )
                    kl_loss = logit_kl(student_logits, teacher_logits)
                    loss = local_loss + kl_weight * kl_loss
                    run_kl += float(kl_loss.detach())
                    kl_loss_sum += float(kl_loss.detach())
                    epoch_kl_steps += 1
                    kl_steps += 1
            if not bool(torch.isfinite(loss)):
                require_finite_tensor(tgt, f"{log_prefix} step {step} target")
                _diagnose_nonfinite_window(
                    layers, inps[idx].to(device), layer_kwargs=layer_kwargs,
                    log_prefix=log_prefix, step=step,
                )
                raise FloatingPointError(f"{log_prefix} step {step} loss is non-finite")
            opt.zero_grad(set_to_none=True)
            if replay_hard:
                if frozen_factor_grads is None or lora_at_boundary is None:
                    raise RuntimeError(
                        f"{log_prefix} entered hard stage without boundary gradients"
                    )
                install_factor_gradients(params_factor, frozen_factor_grads)
                if any(parameter.grad is not None for parameter in params_lora):
                    raise RuntimeError(
                        f"{log_prefix} LoRA received a gradient in replay hard stage"
                    )
                replay_steps += 1
            else:
                loss.backward()
                if used_kl:
                    # Do not pin the previous step's full-vocabulary tensors for the
                    # next three local-only steps. At batch 9 x 128 positions they are
                    # ~700 MB each before log-softmax temporaries.
                    del teacher_tail, teacher_logits, student_tail, student_logits, kl_loss
            for name, family_parameters in factor_params.items():
                norm = math.sqrt(sum(
                    float(parameter.grad.float().square().sum())
                    for parameter in family_parameters if parameter.grad is not None
                ))
                if step == 0:
                    grad_first[name] = norm
                grad_last[name] = norm
                grad_sum[name] += norm
            if replay_hard:
                grad_norm = torch.linalg.vector_norm(torch.stack([
                    parameter.grad.detach().float().norm()
                    for parameter in params_factor
                ]))
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"{log_prefix} step {step} gradient norm is non-finite")
            opt.step()
            if scheduler is not None:
                scheduler.step()
            run += float(local_loss.detach())
            run_total += float(loss.detach())
            if replay_hard:
                run_hard_raw += float(raw_hard_loss.detach()) * len(idx)
                run_hard_samples += len(idx)
            final_soft_loss = float(local_loss.detach())
            final_total_loss = float(loss.detach())
            step += 1
        if hard_gradient_mode == "replay" and epoch == boundary_epoch:
            # Interpret the paper's "last iteration" as the full final-soft
            # objective, not the shuffled tail minibatch. Parameters stay fixed:
            # this pass only forms a low-variance, sample-weighted gradient. The
            # ordinary mean loss makes weighting by batch sample count exact even
            # when 512 = 56 * 9 + 8 leaves a partial final batch.
            lora_at_boundary = [parameter.detach().clone() for parameter in params_lora]
            opt.zero_grad(set_to_none=True)
            endpoint_loss_sum = 0.0
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                x = inps[start:stop].to(device)
                with torch.no_grad():
                    tgt = targets[start:stop].to(device)
                with autocast_context(device, amp_dtype):
                    for layer in layers:
                        out = layer(x, **layer_kwargs)
                        x = out[0] if isinstance(out, tuple) else out
                    endpoint_loss = window_loss(
                        x, tgt, huber_delta=huber_delta, center=center
                    )
                require_finite_tensor(endpoint_loss, f"{log_prefix} endpoint loss")
                sample_fraction = (stop - start) / n
                (endpoint_loss * sample_fraction).backward()
                endpoint_loss_sum += float(endpoint_loss.detach()) * sample_fraction
            endpoint_grad_norm = torch.nn.utils.clip_grad_norm_(
                all_trainable_params, grad_clip
            )
            if not bool(torch.isfinite(endpoint_grad_norm)):
                raise FloatingPointError(
                    f"{log_prefix} full endpoint gradient norm is non-finite"
                )
            boundary_full_gradient_norm = float(endpoint_grad_norm)
            frozen_factor_grads = snapshot_factor_gradients(params_factor)
            factors_at_boundary = [parameter.detach().clone() for parameter in params_factor]
            opt.zero_grad(set_to_none=True)
            boundary_deployed = reconstruction_loss(
                layers, inps, targets, layer_kwargs=layer_kwargs,
                batch_size=batch_size, device=device, huber_delta=huber_delta,
                center=center, deployed=True, amp_dtype=amp_dtype,
            )
            frozen_norm = math.sqrt(sum(
                float(gradient.float().square().sum())
                for gradient in frozen_factor_grads
            ))
            print(
                f"  {log_prefix} ST BOUNDARY epoch={epoch + 1}/{epochs} t={t:.2f} "
                f"endpoint_loss={endpoint_loss_sum:.6f} "
                f"full_grad_preclip={boundary_full_gradient_norm:.6g} "
                f"frozen_factor_grad={frozen_norm:.6g} "
                f"hard_loss={boundary_deployed['hard_reconstruction_loss']:.6f}",
                flush=True,
            )
        elif hard_gradient_mode == "boundary" and epoch == boundary_epoch:
            boundary_deployed = reconstruction_loss(
                layers, inps, targets, layer_kwargs=layer_kwargs,
                batch_size=batch_size, device=device, huber_delta=huber_delta,
                center=center, deployed=True, amp_dtype=amp_dtype,
            )
            print(
                f"  {log_prefix} ST BOUNDARY EXPORT epoch={epoch + 1}/{epochs} "
                f"t={t:.2f} hard_loss="
                f"{boundary_deployed['hard_reconstruction_loss']:.6f}",
                flush=True,
            )
        if replay_hard:
            lora_change = max_parameter_change(params_lora, lora_at_boundary or [])
            if lora_change != 0.0:
                raise RuntimeError(
                    f"{log_prefix} LoRA changed by {lora_change:.6g} in replay hard stage"
                )
            if factors_at_boundary is None or not boundary_deployed:
                raise RuntimeError(f"{log_prefix} has no factor boundary for hard-stage guard")
            hard_epoch_loss = run_hard_raw / run_hard_samples
            hard_loss_ratio = (
                hard_epoch_loss / boundary_deployed["hard_reconstruction_loss"]
            )
            hard_stage_max_loss_ratio = max(hard_stage_max_loss_ratio, hard_loss_ratio)
            hard_stage_displacement = parameter_displacement(
                params_factor, factors_at_boundary
            )
            print(
                f"  {log_prefix} ST HARD epoch={epoch + 1}/{epochs} "
                f"hard_loss={hard_epoch_loss:.6f} ratio={hard_loss_ratio:.3f} "
                f"factor_drift(mean/rms/max)={hard_stage_displacement[0]:.4g}/"
                f"{hard_stage_displacement[1]:.4g}/{hard_stage_displacement[2]:.4g}",
                flush=True,
            )
            if hard_loss_ratio > hard_stage_abort_ratio:
                raise RuntimeError(
                    f"{log_prefix} hard-stage abort: loss ratio {hard_loss_ratio:.3f} "
                    f"exceeds {hard_stage_abort_ratio:.3f}"
                )
        should_log = (
            epoch % max(epochs // 6, 1) == 0
            or epoch == epochs_to_run - 1
            or epoch == boundary_epoch
            or (hard_gradient_mode == "replay" and epoch == boundary_epoch + 1)
        )
        if should_log:
            # Loss alone hid run 2's failure: every window converged while the
            # threshold factor sat at its init. `nonzero` is the gate -- the
            # reference checkpoint deploys at ~0.50, plain absmean at ~0.68.
            probe = ""
            mods = [m for layer in layers for m in layer.modules()
                    if isinstance(m, QuantLinear)]
            if mods:
                st = [m.quantizer.deployed_state(m.merged_weight()) for m in mods]
                probe = (" | a={:.3f} mu={:+.3f} D={:.3f} nonzero={:.3f}".format(
                    sum(d["scale_factor"] for d in st) / len(st),
                    sum(d["mu_factor"] for d in st) / len(st),
                    sum(d["round_factor"] for d in st) / len(st),
                    sum(d["nonzero"] for d in st) / len(st)))
            current_lrs = "/".join(f"{group['lr']:.3g}" for group in opt.param_groups)
            kl_log = ""
            if epoch_kl_steps:
                kl_log = (f" kl {run_kl / epoch_kl_steps:.5f} "
                          f"total {run_total / steps_per_epoch:.6f} "
                          f"w {kl_weight:.3g}")
            stage = "hard-replay" if replay_hard else "soft"
            print(f"  {log_prefix} epoch {epoch + 1:3d}/{epochs} [{stage}] "
                  f"loss {run / steps_per_epoch:.6f}{kl_log} t {t:.2f} "
                  f"lr {current_lrs} ({time.time() - t0:.0f}s){probe}", flush=True)

    final_codes = [m.export()[0].cpu() for m in mods]
    code_total = sum(codes.numel() for codes in initial_codes)
    code_flips = sum(int((before != after).sum())
                     for before, after in zip(initial_codes, final_codes, strict=True))
    deployed = reconstruction_loss(
        layers, inps, targets, layer_kwargs=layer_kwargs, batch_size=batch_size,
        device=device, huber_delta=huber_delta, center=center, deployed=True,
        amp_dtype=amp_dtype,
    )
    hard_loss = deployed["hard_reconstruction_loss"]
    metrics = {
        "soft_loss_last_batch": final_soft_loss,
        "soft_total_loss_last_batch": final_total_loss,
        "hard_code_flip_rate": code_flips / code_total,
        "hard_gradient_replay_steps": replay_steps,
        "optimizer_steps_completed": step,
        "optimizer_steps_planned": total,
        "stopped_at_soft_boundary": float(hard_gradient_mode == "boundary"),
        "hard_stage_max_loss_ratio": hard_stage_max_loss_ratio,
        "hard_stage_factor_displacement_mean": hard_stage_displacement[0],
        "hard_stage_factor_displacement_rms": hard_stage_displacement[1],
        "hard_stage_factor_displacement_max": hard_stage_displacement[2],
        "boundary_full_gradient_norm_preclip": boundary_full_gradient_norm,
        "boundary_hard_reconstruction_loss": boundary_deployed.get(
            "hard_reconstruction_loss", 0.0
        ),
        "hard_stage_lora_max_change": (
            max_parameter_change(params_lora, lora_at_boundary)
            if lora_at_boundary is not None else 0.0
        ),
        "kl_steps": kl_steps,
        "kl_loss_mean": kl_loss_sum / kl_steps if kl_steps else 0.0,
        **kl_calibration,
        **deployed,
    }
    for name in factor_params:
        metrics[f"grad_{name}_first"] = grad_first[name]
        metrics[f"grad_{name}_mean"] = grad_sum[name] / max(step, 1)
        metrics[f"grad_{name}_last"] = grad_last[name]
    print(
        f"  {log_prefix} diagnostics hard_loss={hard_loss:.6f} "
        f"objective={metrics['hard_objective_loss']:.6f} "
        f"code_flip={metrics['hard_code_flip_rate']:.6f} "
        # The null-direction watch for a centered objective: offset_amplitude
        # SHOULD rise above the uncentered ~0.24 if the loss change is doing
        # anything at all, while noise_rel falls. Rising offset with flat noise
        # is the failure mode -- error moved somewhere free without buying
        # anything -- and a runaway offset_norm/max_abs is the blow-up.
        f"offset_amp={metrics['offset_amplitude']:.4f} "
        f"offset_energy={metrics['offset_energy_fraction']:.4f} "
        f"(|mu|={metrics['offset_norm']:.4g}) noise={metrics['noise_rel']:.4f} "
        f"max_act={metrics['max_abs_activation']:.4g} | "
        + (f"kl={metrics['kl_loss_mean']:.4g} w={metrics['kl_weight']:.3g} "
           f"steps={kl_steps} | " if kl_enabled else "")
        + " ".join(
            f"grad_{name}={grad_first[name]:.3g}/{grad_sum[name] / max(step, 1):.3g}/"
            f"{grad_last[name]:.3g}"
            for name in ("scale", "mu", "round")
        )
        + " (first/mean/last)",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def calibration_batches(tokenizer, n_samples: int, seqlen: int, dataset: str,
                        seed: int = 2):
    """`n_samples` sequences of `seqlen` tokens.

    `c4` matches the CAT-Q config; a local path concatenates a text file, which keeps the
    trainer runnable offline and is enough for smoke tests.
    """
    if dataset == "c4":
        from datasets import load_dataset

        # Match SliderQuant's released get_c4(): load shard 00000, choose a
        # document uniformly with Python's seeded RNG, reject short documents,
        # then choose a random contiguous span.  Our original implementation
        # concatenated the beginning of the streaming corpus into sequential
        # chunks, which was neither random nor document preserving.
        shard = (
            "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
            "c4-train.00000-of-01024.json.gz"
        )
        ds = load_dataset("json", data_files={"train": shard}, split="train")
        rng = random.Random(seed)
        out = []
        while len(out) < n_samples:
            row = ds[rng.randint(0, len(ds) - 1)]
            ids = tokenizer(row["text"])["input_ids"]
            if len(ids) < seqlen:
                continue
            start = rng.randint(0, len(ids) - seqlen)
            out.append(ids[start:start + seqlen])
        return torch.tensor(out)
    if dataset == "c4-sequential":
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        buf, out = [], []
        for row in ds:
            buf += tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            while len(buf) >= seqlen and len(out) < n_samples:
                out.append(buf[:seqlen])
                buf = buf[seqlen:]
            if len(out) >= n_samples:
                break
        return torch.tensor(out)
    text = Path(dataset).read_text()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i:i + seqlen] for i in range(0, len(ids) - seqlen, seqlen)]
    if not chunks:
        raise ValueError(f"{dataset} tokenises to fewer than {seqlen} tokens")
    picks = [chunks[i % len(chunks)] for i in range(n_samples)]
    return torch.tensor(picks)


@torch.no_grad()
def capture_layer_inputs(model, layers, ids, device, batch: int = 1,
                         amp_dtype: torch.dtype | None = None):
    """Hidden states entering layers[0], plus the kwargs the layers need."""
    captured, kwargs_seen = [], {}

    class Stop(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, **kw):
            # Released config: fp16_act=false. SliderQuant writes captured
            # activations into an FP32 cache even though forwards use BF16 AMP.
            captured.append(hidden_states.detach().float().cpu())
            kwargs_seen.update({
                k: v for k, v in kw.items()
                if k not in ("past_key_value", "past_key_values", "use_cache")
            })
            kwargs_seen["use_cache"] = False
            raise Stop

    layers[0] = Catcher(layers[0])
    for i in range(0, ids.shape[0], batch):
        try:
            with autocast_context(device, amp_dtype):
                model(ids[i:i + batch].to(device))
        except Stop:
            pass
    layers[0] = layers[0].inner
    return torch.cat(captured, dim=0), kwargs_seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="base fp16 HF model dir")
    ap.add_argument("--out", required=True, type=Path, help="where to write ternary weights")
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=9)
    ap.add_argument("--num-layer", type=int, default=4, help="sliding window size")
    ap.add_argument("--sliding-layer", type=int, default=2, help="window stride")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=64)
    # These are actual PEAK optimiser rates, not the pre-scaling YAML values.
    # SliderQuant multiplies both config LRs by batch size: for Qwen3-1.7B,
    # 3e-4/1.5e-3 * batch 9 -> 2.7e-3/1.35e-2, then decays them linearly.
    ap.add_argument("--lora-lr", type=float, default=2.7e-3)
    ap.add_argument("--factor-lr", type=float, default=1.35e-2)
    ap.add_argument("--fill-window-size", type=int, default=4,
                    help="SliderQuant's fill_window_size; every shipped CAT-Q config "
                         "sets 4. Adds growing windows at the start and shrinking ones "
                         "at the end so edge layers are not under-trained. 0 falls back "
                         "to a bare range(0, n, sliding_layer).")
    ap.add_argument("--ste", choices=("round", "tanh"), default="tanh",
                    help="tanh = CAT-Q's softened relay (default); round = the "
                         "hard-forward SliderQuant baseline")
    ap.add_argument("--factor-wd", type=float, default=0.0,
                    help="weight decay on the learned factors; keep 0 (see train_window)")
    ap.add_argument("--lora-wd", type=float, default=0.0,
                    help="weight decay on LoRA; SliderQuant explicitly uses 0")
    ap.add_argument("--lr-schedule", choices=("linear", "constant"), default="linear",
                    help="per-round LR schedule; SliderQuant uses linear with zero warmup")
    ap.add_argument("--loss-center", choices=("none", "channel"), default="none",
                    help="'channel' removes the residual's per-channel mean before "
                         "Huber, making the loss blind to a systematic offset")
    ap.add_argument("--kl-grad-fraction", type=float, default=0.0,
                    help="target downstream-KL gradient magnitude as a fraction of "
                         "the local objective; 0 disables KL")
    ap.add_argument("--kl-every", type=int, default=4,
                    help="apply downstream logit KL every N optimiser steps")
    ap.add_argument("--kl-positions", type=int, default=128,
                    help="evenly spaced sequence positions projected to vocabulary")
    ap.add_argument("--kl-norm-batches", type=int, default=4,
                    help="probe batches used for robust per-window gradient scaling")
    ap.add_argument("--kl-weight-min", type=float, default=0.0)
    ap.add_argument("--kl-weight-max", type=float, default=0.1,
                    help="safety clamp on the auto-normalised KL loss weight")
    ap.add_argument("--huber-loss-max", type=float, default=1.0,
                    help="coefficient in delta=0.1 + round/num_rounds * value")
    ap.add_argument("--init-scale-from-raw-weights", action="store_true",
                    help="use the Qwen3-4B scale initialization")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--progressive-ratio", type=float, default=0.8,
                    help="fraction of epochs in CAT-Q's soft stage (source default 0.8)")
    ap.add_argument(
        "--hard-gradient-mode", choices=("recompute", "replay", "boundary"),
        default="boundary",
        help="boundary stops at gamma and hard-exports the soft-stage solution; "
             "recompute and replay are experimental hard-stage modes",
    )
    ap.add_argument(
        "--hard-stage-abort-ratio", type=float, default=1.5,
        help="abort replay as soon as an epoch's deployed hard loss exceeds this "
             "multiple of the soft-boundary hard loss",
    )
    ap.add_argument("--s0", type=float, default=30.0,
                    help="final soft-relay sharpness (source default 30)")
    ap.add_argument(
        "--calib",
        default="c4",
        help="'c4' (source-faithful random documents), 'c4-sequential' (legacy), "
             "or a path to a text file",
    )
    ap.add_argument("--seed", type=int, default=2,
                    help="calibration/training seed; CAT-Q's Qwen3-1.7B config uses 2")
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--amp-dtype", choices=("none", "bfloat16"), default="none",
        help="forward-compute autocast; CAT-Q's released Qwen3-1.7B config uses "
             "bfloat16 while keeping active parameters/activation caches FP32",
    )
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all (smoke tests)")
    args = ap.parse_args()

    if args.kl_grad_fraction < 0:
        ap.error("--kl-grad-fraction must be non-negative")
    if not 0 < args.progressive_ratio <= 1:
        ap.error("--progressive-ratio must be in (0, 1]")
    if args.hard_gradient_mode in ("replay", "boundary") and args.kl_grad_fraction:
        ap.error("paper-mechanism --hard-gradient-mode arms cannot be combined with KL")
    if args.hard_stage_abort_ratio <= 1.0:
        ap.error("--hard-stage-abort-ratio must be greater than 1")
    if args.kl_grad_fraction > 0:
        if args.kl_every <= 0:
            ap.error("--kl-every must be positive")
        if args.kl_positions <= 0:
            ap.error("--kl-positions must be positive")
        if args.kl_norm_batches <= 0:
            ap.error("--kl-norm-batches must be positive")
        if args.kl_weight_min < 0 or args.kl_weight_max < args.kl_weight_min:
            ap.error("invalid KL weight bounds")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else None
    if amp_dtype is not None and device.type != "cuda":
        ap.error("--amp-dtype bfloat16 requires --device cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    all_layers = model.model.layers
    layers = all_layers
    if args.max_layers:
        layers = layers[: args.max_layers]
    print(f"model {args.model} | {len(layers)} layers | device {device}", flush=True)
    hidden_size = int(model.config.hidden_size)
    cache_gib = args.nsamples * args.seqlen * hidden_size * 4 / (1024 ** 3)
    print(
        f"FP32 host activation cache estimate: {cache_gib:.2f} GiB/tensor, "
        f"~{3 * cache_gib:.2f} GiB for quant/fp/target streams",
        flush=True,
    )

    ids = calibration_batches(tok, args.nsamples, args.seqlen, args.calib, args.seed)
    print(f"calibration: {tuple(ids.shape)} from {args.calib}", flush=True)
    inps, layer_kwargs = capture_layer_inputs(
        model, model.model.layers, ids, device, amp_dtype=amp_dtype
    )
    print(f"captured hidden states {tuple(inps.shape)}", flush=True)

    def run_layers(sub, x):
        with autocast_context(device, amp_dtype):
            for layer in sub:
                out = layer(x, **layer_kwargs)
                x = out[0] if isinstance(out, tuple) else out
        return x

    def run_layers_batched(sub, x):
        """`run_layers` over the whole sample set, in `--batch-size` chunks.

        These two passes -- advancing the cached streams and building the window's
        teacher target -- ran one sample at a time. On the measured depth-20 /
        20-epoch A100 job that was 70% of wall clock: 123 of 175 minutes, against
        50 minutes of actual optimisation, and it was going to miss the 5h timeout
        with three rounds still to run. `train_window` already feeds batches of
        `batch_size` through the same `layer_kwargs`, so the captured position
        embeddings demonstrably broadcast -- there was never a shape reason for
        batch 1.
        """
        return torch.cat(
            [run_layers(sub, x[i:i + args.batch_size].to(device)).float().cpu()
             for i in range(0, x.shape[0], args.batch_size)], dim=0
        )

    def advance_to(x, frm: int, to: int, full_precision: bool):
        """Run cached activations through layers [frm, to).

        ``full_precision`` selects the teacher stream. The quantized stream carries
        accumulated quantization error; the teacher stream does not.
        """
        with torch.no_grad():
            sub = layers[frm:to]
            ctx = fp_target(sub) if full_precision else contextlib.nullcontext()
            with ctx:
                return run_layers_batched(sub, x)

    n = len(layers)
    expected_linears = sum(
        1
        for layer in layers
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES)
    )
    rounds = build_window_scheduler(n, args.num_layer, args.sliding_layer,
                                    args.fill_window_size)
    print(f"scheduler: {len(rounds)} rounds over {n} layers "
          f"(num_layer={args.num_layer} sliding={args.sliding_layer} "
          f"fill={args.fill_window_size})", flush=True)

    # `inps` holds the activations entering layer `cur_start`. Window starts are
    # non-decreasing, so advance lazily when a round needs a deeper entry point --
    # the growing fill windows all start at 0 and need no advance at all.
    # TWO streams, as SliderQuant keeps them. `inps_q` is the student's input and
    # carries accumulated quantisation error; `inps_fp` is the teacher's and never
    # does. The target is teacher(inps_fp), so it cannot drift along with the student.
    inps_q = inps
    inps_fp = inps.clone()
    del inps
    cur_start = 0
    round_metrics: list[dict[str, float | int | str]] = []
    for r_idx, idxs in enumerate(rounds):
        start, stop = idxs[0], idxs[-1] + 1
        if start > cur_start:
            inps_q = advance_to(inps_q, cur_start, start, full_precision=False)
            inps_fp = advance_to(inps_fp, cur_start, start, full_precision=True)
            cur_start = start
        window = layers[start:stop]
        tag = f"{start}..{stop - 1}"
        huber_delta = huber_delta_at(r_idx, len(rounds), args.huber_loss_max)

        with torch.no_grad(), fp_target(window):
            tgt = run_layers_batched(window, inps_fp)
        drift = float((inps_q - inps_fp).norm() / inps_fp.norm().clamp(min=1e-6))
        wrapped = {}
        for off, layer in enumerate(window):
            wrapped.update({f"model.layers.{start + off}.{k}": v
                            for k, v in wrap_layer(layer, args.group_size, args.lora_r,
                                                   args.s0, args.progressive_ratio,
                                                   args.init_scale_from_raw_weights,
                                                   args.ste).items()})
        print(f"round {r_idx + 1}/{len(rounds)} layers {tag}: "
              f"{len(wrapped)} newly wrapped linears | Huber delta={huber_delta:.4f} "
              f"| input drift={drift:.4f}", flush=True)
        metrics = train_window(window, inps_q, tgt, layer_kwargs=layer_kwargs,
                               epochs=args.epochs, batch_size=args.batch_size,
                               lora_lr=args.lora_lr, factor_lr=args.factor_lr,
                               grad_clip=args.grad_clip, device=device,
                               factor_wd=args.factor_wd, lora_wd=args.lora_wd,
                               huber_delta=huber_delta, lr_schedule=args.lr_schedule,
                               center=args.loss_center == "channel",
                               hard_gradient_mode=args.hard_gradient_mode,
                               hard_stage_abort_ratio=args.hard_stage_abort_ratio,
                               amp_dtype=amp_dtype,
                               suffix=all_layers[stop:],
                               final_norm=model.model.norm, lm_head=model.lm_head,
                               kl_grad_fraction=args.kl_grad_fraction,
                               kl_every=args.kl_every,
                               kl_position_count=args.kl_positions,
                               kl_norm_batches=args.kl_norm_batches,
                               kl_weight_min=args.kl_weight_min,
                               kl_weight_max=args.kl_weight_max,
                               log_prefix=f"[{tag}]")
        round_metrics.append({"round": r_idx, "layers": tag, **metrics})

    # Export ONCE from the final live model. wrap_layer only returns modules it
    # newly wrapped and skips already-wrapped ones, so exporting per-window shipped
    # every overlapping layer's first-window state -- with num_layer 4 /
    # sliding_layer 2 that is half of all layers, frozen before later windows
    # finished training them.
    exported: dict[str, tuple] = {}
    for li, layer in enumerate(layers):
        for name, module in layer.named_modules():
            if isinstance(module, QuantLinear):
                exported[f"model.layers.{li}.{name}"] = module.export_state()
    if len(exported) != expected_linears:
        raise RuntimeError(
            f"expected {expected_linears} quantised linears, found {len(exported)} at export"
        )
    validate_exported_state(exported)
    print(f"exported {len(exported)} QuantLinear modules from the final model", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.out / "ternary.pt")
    (args.out / "metrics.json").write_text(json.dumps(round_metrics, indent=2))
    print(f"wrote {args.out / 'ternary.pt'}: {len(exported)} quantised linears")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
