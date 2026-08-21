# Copyright 2026 Penk Chen <penkia@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from BitTern by Intel Labs China --
# specifically `projects/cat-q/quantize/quantizer.py` -- which is licensed under
# the Apache License, Version 2.0. See https://github.com/IntelChina-AI/BitTern
# and http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE OF MODIFICATION (Apache-2.0 section 4(b)): this file has been changed
# from that source. The differences are:
#   * the three learnable-factor modules are trainable here; upstream they are
#     constructed with `requires_grad=False`, because the released code only
#     replays already-learned values from a checkpoint;
#   * the softened-ternarization relay of the CAT-Q paper (arXiv 2606.26650,
#     Eq. 5/6) is implemented, together with its per-epoch sharpness ramp.
#     Upstream's `forward` discards its `quant_rate` argument and always applies
#     hard ternarization;
#   * a `force_hard` export path and a `deployed_state` probe are added, so a
#     trainer can read the deployed ternary codes mid-run;
#   * only the shipped `learnable_factor_act: sigmoid` and
#     `ter_scale_type: absmean` configuration is supported. Upstream's
#     `double_sigmoid` / `softplus` / `exp` factor activations and its
#     `variance` scale type are not carried over.
#
# The upstream sources carry no copyright headers, SPDX tags, or NOTICE file,
# so there are no such notices to retain under sections 4(c) and 4(d).

"""CAT-Q ternary quantizer — trainable port of the released inference version.

The released `BitTern/projects/cat-q/quantize/quantizer.py` is inference-only:
its three factor modules carry `requires_grad=False` because a checkpoint just
replays learned values. This is the same math with the factors trainable, which
is what a from-scratch trainer needs.

Faithful to the released forward path (per group of `group_size` weights):

    mean  = grouped.mean()                         # shift_mu
    absmean = grouped if init_scale_from_raw_weights else grouped - mean
    scale = |absmean|.mean() + 1e-6                # ter_scale_type=absmean;
                                                   # init_scale_from_raw_weights
                                                   # defaults FALSE (only 4b sets it)
    mean  = mean + (sigmoid(t_mu)*2 - 1) * scale    # learnable_mu
    scale = sigmoid(t_scale)*2 * scale              # learnable_scale
    thd   = init_round_thd * sigmoid(t_round)*2     # learnable_round
    q     = clamp(round(((g - mean)/scale) * 0.5/thd), -1, 1) * scale
    # drop_quant_mu=True in every shipped config, so `mean` is NOT added back —
    # which is what keeps the result pure ternary*scale and Q2_0-exportable.

All factors initialise to zero, so `sigmoid(0) = 0.5` makes the initial state
exactly plain absmean ternarisation with threshold `init_round_thd`.

Softened Ternarisation is the paper's Eq. 5/6 (arXiv 2606.26650), not a soft
rounding. Hard ternarisation has zero gradient almost everywhere, which is why
"ternarization optimization is notably difficult to converge"; ST supplies a
differentiable relay on the normalised weight `W_hat = (W - mu)/alpha`:

    f(W_hat; s, D) = [tanh(s*(W_hat - D)) + tanh(s*(W_hat + D))] / (2*tanh(s))

Two stages over training progress `t` in [0, 1], with gamma = progressive_ratio:

    t <= gamma :  soft, sharpness ramped   s = (t/gamma) * s0     (s0 = 30)
    t >  gamma :  hard ternarisation Q(.), straight-through gradient

As s -> inf the soft form approaches the hard one, so the switch at gamma is not
a discontinuity in behaviour, only in gradient.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TernaryQuantizer(nn.Module):
    def __init__(self, shape: tuple[int, int], group_size: int = 128,
                 init_round_thd: float = 0.5, learnable: bool = True,
                 s0: float = 30.0, gamma: float = 0.8,
                 init_scale_from_raw_weights: bool = False,
                 ste: str = "round"):
        super().__init__()
        self.group_size = group_size or shape[-1]
        self.init_round_thd = init_round_thd
        # Scale comes from the mean-centered group unless this is set.
        self.init_scale_from_raw_weights = init_scale_from_raw_weights
        self.s0 = s0
        self.gamma = gamma          # progressive_ratio
        # Gradient estimator. "round" is SliderQuant's -- CAT-Q is built on it
        # (BitTern README acknowledgement), and its quantize/quantizer.py:53 is
        #     round_ste(x) = (x.round() - x).detach() + x
        # i.e. hard forward, identity backward, with NO tanh anywhere. Applied
        # *inside* the threshold division so `thd` stays in the graph and still
        # receives gradient -- the concern that motivated the tanh relay in the
        # first place, and which turned out to be unfounded.
        # "tanh" is the paper's Eq. 5/6 relay (arXiv 2606.26650).
        if ste not in ("round", "tanh"):
            raise ValueError(f"ste must be 'round' or 'tanh', got {ste!r}")
        self.ste = ste
        if shape[-1] % self.group_size:
            raise ValueError(f"in_features {shape[-1]} not divisible by {self.group_size}")
        dim = int(shape[0] * math.ceil(shape[1] / self.group_size))
        z = torch.zeros((dim, 1))
        self.t_scale = nn.Parameter(z.clone(), requires_grad=learnable)
        self.t_mu = nn.Parameter(z.clone(), requires_grad=learnable)
        self.t_round = nn.Parameter(z.clone(), requires_grad=learnable)

    # --- factors: sigmoid(theta) * alpha + beta, matching _factor_module ------
    def _scale_factor(self):  # ScaleSigmoid(alpha=2.0)
        return torch.sigmoid(self.t_scale) * 2.0

    def _mu_factor(self):     # ScaleSigmoid(alpha=2.0, beta=-1.0)
        return torch.sigmoid(self.t_mu) * 2.0 - 1.0

    def _round_factor(self):  # ScaleSigmoid(alpha=2.0)
        return torch.sigmoid(self.t_round) * 2.0

    def codes_and_scales(self, weight: torch.Tensor, progress: float = 1.0,
                         force_hard: bool = False):
        """Return (ternary codes in {-1,0,1}, per-group scale).

        `progress` is t in [0, 1] over the window's training. Deployment must pass
        `force_hard=True`; progress alone is deliberately not an export contract.
        """
        grouped = weight.reshape(-1, self.group_size)
        mean = grouped.mean(dim=-1, keepdim=True)
        absmean_values = grouped if self.init_scale_from_raw_weights else grouped - mean
        scale = absmean_values.abs().mean(dim=-1, keepdim=True) + 1e-6
        mean = mean + self._mu_factor() * scale             # mu = mu0 + d_mu * a0
        scale = self._scale_factor() * scale                # alpha = d_alpha * a0
        thd = self.init_round_thd * self._round_factor()    # Delta = d_Delta * D0
        w_hat = (grouped - mean) / scale
        codes = self._ternarize(w_hat, thd, progress, force_hard=force_hard)
        return codes, scale

    def _ternarize(self, w_hat: torch.Tensor, thd: torch.Tensor,
                   progress: float, force_hard: bool = False) -> torch.Tensor:
        """Eq. 5/6: sharpness-ramped double tanh, then hard with straight-through."""
        u = w_hat * 0.5 / thd
        hard = torch.clamp(torch.round(u), -1.0, 1.0)
        if not force_hard and self.ste == "round":
            # SliderQuant's estimator: hard forward, identity backward. `u` carries
            # `thd` and `w_hat` carries the scale/mu factors, so all three factors
            # get gradient with no sharpness parameter involved at all.
            return torch.clamp((torch.round(u) - u).detach() + u, -1.0, 1.0)
        if force_hard:
            # Export must not depend on training progress.
            return hard
        # s ramps 0 -> s0 across stage 1; floor it so 2*tanh(s) cannot be 0.
        # (At s -> 0 the transition degenerates to the identity, i.e. no
        # ternarisation at all, which is the intended start of the relay.)
        frac = 1.0 if progress > self.gamma else max(progress / self.gamma, 1e-3)
        s = frac * self.s0
        soft = (torch.tanh(s * (w_hat - thd)) + torch.tanh(s * (w_hat + thd))) / (
            2.0 * math.tanh(s)
        )
        if progress > self.gamma:
            # Stage 2 is hard ternarisation. Route the gradient through the
            # sharpest soft form rather than through w_hat directly: with `hard`
            # detached and w_hat independent of `thd`, the threshold factor would
            # otherwise receive no gradient at all for the last 20% of training.
            return soft + (hard - soft).detach()
        return soft

    def forward(self, weight: torch.Tensor, progress: float = 1.0) -> torch.Tensor:
        # CAT-Q Eq. 6 starts the softened relay at the exact pretrained weight,
        # not at the s->0 limit of alpha*f(W_hat).  The latter reconstructs
        # W-mu because deployment deliberately drops mu, so it is not the
        # identity mapping promised at t=0.
        if self.ste == "tanh" and progress <= 0.0:
            return weight
        codes, scale = self.codes_and_scales(weight, progress)
        return (codes * scale).reshape(weight.shape)

    @torch.no_grad()
    def deployed_state(self, weight: torch.Tensor) -> dict[str, float]:
        """Factor means and the deployed nonzero fraction, at full hardness.

        Run 2 failed with 14 converging windows because `_round_factor` never
        left its init: nonzero came out 0.68 (what threshold 0.5 predicts) where
        the reference checkpoint is ~0.50. The loss could not show that -- only
        the deployed codes can -- so the trainer must report it per window.
        """
        codes, _ = self.codes_and_scales(weight, progress=1.0, force_hard=True)
        return {
            "scale_factor": float(self._scale_factor().mean()),
            "mu_factor": float(self._mu_factor().mean()),
            "round_factor": float(self._round_factor().mean()),
            "nonzero": float((codes != 0).float().mean()),
        }
