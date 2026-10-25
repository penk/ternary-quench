#!/usr/bin/env python3
"""Convert a base-3 packed MLX model to standard 2-bit MLX format.

The base-3 model is smaller on disk (1.6 bpw vs 2.0 bpw) but requires this
one-time expansion before inference. The output is a standard MLX
QuantizedLinear model that mlx_lm.load() accepts without any modification.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .packing import MLX_BASE3_CODES_PER_WORD, unpack_mlx_base3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path,
                    help="base-3 packed MLX model directory")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for the expanded 2-bit model")
    args = ap.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    base3_layers = config.get("ternary_base3_layers")
    if not base3_layers:
        raise ValueError("config.json has no ternary_base3_layers — not a base-3 model")
    base3_set = set(base3_layers)
    group_size = int(config["quantization"]["group_size"])

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    for filename in shards:
        tensors = load_file(args.model / filename)
        for stem in base3_set:
            weight_key = f"{stem}.weight"
            scales_key = f"{stem}.scales"
            if weight_key not in tensors:
                continue
            packed = tensors[weight_key].numpy().astype(np.uint32)   # 1-D uint32
            scales = tensors[scales_key].numpy()                       # float16 [rows, groups]
            rows, n_groups = scales.shape
            n_weights = rows * n_groups * group_size
            tensors[weight_key] = torch.from_numpy(
                unpack_mlx_base3(packed, n_weights, rows)
            )
        save_file(tensors, args.out / filename)
        print(f"wrote {args.out / filename}", flush=True)

    # Write config without ternary_base3_layers
    out_config = {k: v for k, v in config.items() if k != "ternary_base3_layers"}
    (args.out / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")

    # Copy all other files unchanged (tokenizer, index, etc.)
    for src in args.model.iterdir():
        if src.name in {"config.json"} or src.suffix == ".safetensors":
            continue
        if src.is_file():
            shutil.copy2(src, args.out / src.name)

    print(f"expanded model written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
