#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=1.5.0,<2.0",
#     "numpy>=1.26.0",
#     "safetensors>=0.5.0",
#     "torch==2.7.1",
# ]
# ///
"""Build a mixed 2-bit/4-bit MLX model from a ternary prefix artifact.

The base must already be an MLX affine-quantized model.  Matching prefix
linears are replaced bit-exactly with packed ternary weights; every unmatched
quantized module keeps the base model's precision.  Per-module quantization
entries in config.json make mlx-lm instantiate the mixed layout correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file

from .packing import pack_mlx_2bit


def mlx_stem(name: str) -> str:
    if name.startswith("model.language_model"):
        return name.replace("model.language_model", "language_model.model", 1)
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ternary-prefix", required=True,
                    help="local file or hf:// bucket object")
    ap.add_argument("--base", required=True,
                    help="local MLX affine model directory or Hub model ID")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expected-prefix-modules", type=int, default=0)
    ap.add_argument("--expected-prefix-parameters", type=int, default=0)
    ap.add_argument("--name",
                    help="model name stored in config.json (defaults to --out name)")
    ap.add_argument("--repo", help="Hub model repository for the packed output")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    if args.name is None:
        args.name = args.out.name

    import torch

    from huggingface_hub import HfApi, HfFileSystem, snapshot_download

    prefix_path = Path(args.ternary_prefix)
    if args.ternary_prefix.startswith("hf://"):
        prefix_path = Path("/tmp/ternary-prefix.pt")
        HfFileSystem(token=os.environ.get("HF_TOKEN")).get(
            args.ternary_prefix, str(prefix_path)
        )
    if not prefix_path.is_file():
        raise FileNotFoundError(f"ternary prefix is missing: {prefix_path}")

    base_path = Path(args.base)
    if not base_path.is_dir():
        base_path = Path(snapshot_download(
            args.base,
            token=os.environ.get("HF_TOKEN"),
            local_dir="/tmp/quantized-base",
        ))

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    raw = torch.load(prefix_path, map_location="cpu", weights_only=False)
    trained = {mlx_stem(name): record for name, record in raw.items()}
    if args.expected_prefix_modules and len(trained) != args.expected_prefix_modules:
        raise RuntimeError(
            f"prefix has {len(trained)} modules; expected {args.expected_prefix_modules}"
        )

    index_path = base_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"quantized base index is missing: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    replaced: set[str] = set()
    quantized_parameters = 0
    shard_manifest = []

    for filename in shards:
        source = base_path / filename
        tensors = load_file(source)
        for stem, record in trained.items():
            weight_key = f"{stem}.weight"
            if weight_map.get(weight_key) != filename:
                continue
            scales_key = f"{stem}.scales"
            biases_key = f"{stem}.biases"
            if scales_key not in tensors or biases_key not in tensors:
                raise KeyError(f"base quantization tensors are missing for {stem}")
            codes = record["codes"].numpy()
            scales = record["scales"].numpy()
            rows = int(tensors[scales_key].shape[0])
            cols = int(codes.size // rows)
            if codes.size != rows * cols or cols % 128:
                raise ValueError(f"invalid ternary geometry for {stem}: {codes.shape}")
            if not np.isin(codes, (-1, 0, 1)).all():
                raise ValueError(f"non-ternary codes in {stem}")
            shifted = (codes.astype(np.int16) + 1).astype(np.uint8).reshape(rows, cols)
            tensors[weight_key] = torch.from_numpy(pack_mlx_2bit(shifted))
            tensors[scales_key] = torch.from_numpy(
                scales.reshape(rows, cols // 128).astype(np.float16)
            )
            tensors[biases_key] = torch.from_numpy(
                (-scales).reshape(rows, cols // 128).astype(np.float16)
            )
            replaced.add(stem)
            quantized_parameters += rows * cols
        destination = args.out / filename
        save_file(tensors, destination)
        shard_manifest.append({
            "file": filename,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        })
        print(f"wrote {destination}", flush=True)

    missing = sorted(set(trained) - replaced)
    if missing:
        raise RuntimeError(f"{len(missing)} prefix modules did not match the base: {missing[:5]}")
    if args.expected_prefix_parameters and quantized_parameters != args.expected_prefix_parameters:
        raise RuntimeError(
            f"prefix has {quantized_parameters} parameters; "
            f"expected {args.expected_prefix_parameters}"
        )

    # mlx-lm supports custom per-module quantization records inside the global
    # quantization mapping.  Every base quantized module not replaced above is
    # still 4-bit and therefore needs an explicit override from the 2-bit
    # default used by the ternary prefix.
    base_config = json.loads((base_path / "config.json").read_text())
    base_quant = base_config["quantization"]
    overrides = {}
    for key in weight_map:
        if not key.endswith(".scales"):
            continue
        stem = key.removesuffix(".scales")
        if stem not in replaced:
            overrides[stem] = {
                "group_size": int(base_quant["group_size"]),
                "bits": int(base_quant["bits"]),
                "mode": base_quant.get("mode", "affine"),
            }
    mixed_quant = {"group_size": 128, "bits": 2, "mode": "affine", **overrides}
    base_config["quantization"] = mixed_quant
    base_config["quantization_config"] = mixed_quant
    base_config["_name_or_path"] = args.name
    (args.out / "config.json").write_text(json.dumps(base_config, indent=2) + "\n")

    for source in base_path.iterdir():
        if source.name in {"config.json", "model.safetensors.index.json"}:
            continue
        if source.suffix == ".safetensors":
            continue
        if source.is_file():
            shutil.copy2(source, args.out / source.name)

    total_tensor_bytes = sum(item["bytes"] for item in shard_manifest)
    index["metadata"] = {"total_size": total_tensor_bytes}
    index_path_out = args.out / "model.safetensors.index.json"
    index_path_out.write_text(json.dumps(index, indent=2) + "\n")
    (args.out / "Modelfile").write_text("FROM .\n")

    manifest = {
        "name": args.name,
        "base": args.base,
        "ternary_prefix": args.ternary_prefix,
        "ternary": {
            "bits": 2,
            "group_size": 128,
            "modules": len(replaced),
            "parameters": quantized_parameters,
        },
        "retained_base_quantization": {
            "bits": int(base_quant["bits"]),
            "group_size": int(base_quant["group_size"]),
            "modules": len(overrides),
        },
        "tensor_bytes": total_tensor_bytes,
        "directory_bytes": sum(
            path.stat().st_size for path in args.out.iterdir() if path.is_file()
        ),
        "shards": shard_manifest,
    }
    (args.out / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.repo:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(args.out),
            repo_id=args.repo,
            commit_message=f"Add {args.name} MLX artifact",
        )
        print(f"pushed https://huggingface.co/{args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
