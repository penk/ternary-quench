# Copyright 2026 Penk Chen <penkia@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

"""Write `ternary.pt` as an MLX-quantized model.

Ternary codes plus one fp16 scale per group map to MLX affine 2-bit with
``scale=d``, ``bias=-d``, and codes shifted by one. Unquantized tensors are copied
from the base model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .packing import QK2_0, pack_mlx_2bit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ternary", required=True, type=Path, help="ternary.pt from train.py")
    ap.add_argument("--base", required=True, type=Path, help="base fp16 HF model dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-layers", type=int, default=0,
                    help="diagnostic: quantise only decoder layers below this index; "
                         "0 uses every trained layer")
    ap.add_argument(
        "--fp-module",
        action="append",
        default=[],
        help="diagnostic: keep this exact linear stem in fp16; repeat as needed",
    )
    ap.add_argument(
        "--alternate-ternary",
        type=Path,
        help="diagnostic: second ternary.pt used from --alternate-from-layer onward",
    )
    ap.add_argument("--alternate-from-layer", type=int, default=0)
    ap.add_argument(
        "--alternate-to-layer",
        type=int,
        default=0,
        help="exclusive end of alternate range; 0 means through the final layer",
    )
    args = ap.parse_args()

    if bool(args.alternate_ternary) != bool(args.alternate_from_layer):
        ap.error("--alternate-ternary and a positive --alternate-from-layer are required together")

    import mlx.core as mx
    import torch

    trained = torch.load(args.ternary, map_location="cpu")
    alternate = (
        torch.load(args.alternate_ternary, map_location="cpu")
        if args.alternate_ternary
        else None
    )
    print(f"trained linears: {len(trained)}")

    # base weights, so untrained tensors (norms, embeddings, lm_head, and any
    # layer the run did not reach) still make a complete model
    from safetensors.torch import load_file
    base: dict[str, np.ndarray] = {}
    shards = sorted(args.base.glob("*.safetensors"))
    for shard in shards:
        for k, v in load_file(str(shard)).items():
            base[k] = v.to(torch.float16).numpy()
    print(f"base tensors: {len(base)} from {len(shards)} shard(s)")

    out: dict[str, np.ndarray] = {}
    quantised = 0
    for key, arr in base.items():
        stem = key[: -len(".weight")] if key.endswith(".weight") else None
        rec = trained.get(stem) if stem else None
        if stem and alternate is not None:
            parts = stem.split(".")
            if len(parts) >= 3 and parts[:2] == ["model", "layers"]:
                layer_id = int(parts[2])
                if layer_id >= args.alternate_from_layer and (
                    not args.alternate_to_layer or layer_id < args.alternate_to_layer
                ):
                    rec = alternate.get(stem)
        if stem in args.fp_module:
            rec = None
        if rec is not None and args.max_layers:
            parts = stem.split(".")
            if len(parts) >= 3 and parts[:2] == ["model", "layers"]:
                if int(parts[2]) >= args.max_layers:
                    rec = None
        if rec is None:
            out[key] = arr
            continue
        for field, value in rec.items():
            if isinstance(value, torch.Tensor) and not bool(torch.all(torch.isfinite(value))):
                bad = int((~torch.isfinite(value)).sum())
                raise FloatingPointError(
                    f"{stem}.{field} contains {bad}/{value.numel()} non-finite values"
                )
        codes = rec["codes"].numpy()            # (rows*groups, 128) in {-1,0,1}
        scales = rec["scales"].numpy()          # (rows*groups, 1)
        if not np.isin(codes, (-1, 0, 1)).all():
            raise ValueError(f"{stem}.codes is not ternary: {np.unique(codes)[:10]}")
        rows, cols = arr.shape
        groups = cols // QK2_0
        codes = (codes.astype(np.int16) + 1).astype(np.uint8).reshape(rows, cols)
        out[f"{stem}.weight"] = pack_mlx_2bit(codes)
        out[f"{stem}.scales"] = scales.reshape(rows, groups).astype(np.float16)
        out[f"{stem}.biases"] = (-scales.reshape(rows, groups)).astype(np.float16)
        quantised += 1

    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "model.safetensors"),
                        {k: mx.array(v) for k, v in out.items()})
    cfg = json.loads((args.base / "config.json").read_text())
    quant = {"group_size": QK2_0, "bits": 2, "mode": "affine"}
    cfg["quantization"] = quant
    cfg["quantization_config"] = quant
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                 "vocab.json", "merges.txt", "chat_template.jinja"):
        src = args.base / name
        if src.exists():
            shutil.copy2(src, args.out / name)
    print(f"wrote {args.out}: {quantised} quantised + {len(base) - quantised} copied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
