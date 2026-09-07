#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "huggingface-hub>=1.5.0,<2.0",
#     "numpy>=1.26.0,<2.0",
#     "protobuf>=4.21.0,<5.0.0",
#     "requests>=2.31.0,<3.0.0",
#     "safetensors>=0.5.0",
#     "sentencepiece>=0.1.98,<0.3.0",
#     "torch==2.7.1",
#     "transformers>=5.0.0,<6.0.0",
# ]
# ///
"""Export a trained ternary checkpoint as a mixed Q2_0/Q4 GGUF.

The exporter deliberately uses llama.cpp's own Hugging Face converter for the
architecture metadata, tokenizer, tensor names and Qwen3.5 linear-attention
reordering.  It changes only the tensor source and precision policy:

* trained CAT-Q matrices are reconstructed from their deployed ternary
  ``codes * scales`` and written as upstream GGML ``Q2_0``;
* untouched decoder linears at or after ``--suffix-from-layer`` are written at
  ``--suffix-type``;
* token embeddings and the language-model head are written at ``--top-type``;
* all other tensors follow llama.cpp's normal Qwen3.5 conversion rules.

Q2_0 in current upstream llama.cpp uses groups of 64.  The trainer uses groups
of 128, so each trained group becomes two Q2_0 blocks with the same
scale.  This is numerically lossless: the deployed values remain exactly
``{-scale, 0, +scale}`` (modulo the same fp16 scale storage used by GGUF).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


TRAIN_GROUP_SIZE = 128
DEFAULT_Q2_GROUP_SIZE = 64
Q2_SCALE_BYTES = 2

QWEN35_LINEAR_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hf_name(name: str) -> str:
    """Match llama.cpp's Qwen text-model filtering of HF tensor names."""
    return name.replace("language_model.", "") if "language_model." in name else name


def checkpoint_weight_name(name: str) -> str:
    """Convert the trainer's module-stem key to a normalized HF weight key."""
    name = normalized_hf_name(name)
    return name if name.endswith(".weight") else f"{name}.weight"


def decoder_layer(name: str) -> int | None:
    marker = "model.layers."
    if not name.startswith(marker):
        return None
    first = name[len(marker):].split(".", 1)[0]
    return int(first) if first.isdecimal() else None


def is_decoder_linear(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in QWEN35_LINEAR_SUFFIXES)


# One log line per trained tensor, not five.
#
# HF Jobs' log API returns only about the first 1,450 lines.  At five lines per
# trained tensor the window filled after ~57 of 450 tensors -- which is ~9% of
# the output bytes -- so every run appeared to freeze at "Writing: 9%" whether
# it was healthy or dying.  That artifact made a real 94x slowdown and a
# perfectly healthy run produce identical-looking logs, and it cost several
# paid jobs to tell apart.  At one line per tensor the whole run fits.
_LAST_SCAN_SECONDS = 0.0
_LAST_RECONSTRUCT_SECONDS = 0.0


def rss_bytes() -> int:
    """Current RSS on Linux; peak RSS fallback on macOS."""
    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def quantize_q2_0(
    data: np.ndarray,
    *,
    row_chunk: int = 16,
    block_size: int = DEFAULT_Q2_GROUP_SIZE,
    label: str | None = None,
) -> np.ndarray:
    """Reference-compatible upstream Q2_0 packing.

    llama.cpp changed Q2_0 from 128-value blocks to 64-value blocks in 2026.
    The selected llama.cpp checkout is therefore the source of truth for the
    block size; the default exists only for unit-level callers.  Each block
    stores one fp16 scale followed by four 2-bit codes per byte.  Codes 0/1/2
    reconstruct -scale/0/+scale; code 3 is rejected for trained tensors.
    """
    if block_size not in {64, 128}:
        raise ValueError(f"unsupported llama.cpp Q2_0 block size: {block_size}")
    block_bytes = Q2_SCALE_BYTES + block_size // 4
    if data.ndim < 2 or data.shape[-1] % block_size:
        raise ValueError(
            f"Q2_0 requires a matrix with {block_size}-divisible rows, "
            f"got {data.shape}"
        )
    if row_chunk <= 0:
        raise ValueError(f"Q2_0 row chunk must be positive, got {row_chunk}")
    original_shape = data.shape
    columns = original_shape[-1]
    rows = np.asarray(data).reshape(-1, columns)
    output = np.empty(
        (rows.shape[0], columns // block_size * block_bytes),
        dtype=np.uint8,
    )
    started = time.perf_counter()
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    for start in range(0, rows.shape[0], row_chunk):
        stop = min(start + row_chunk, rows.shape[0])
        blocks = np.asarray(rows[start:stop], dtype=np.float32).reshape(
            -1, block_size
        )
        scale = np.max(np.abs(blocks), axis=1, keepdims=True)
        inverse = np.divide(
            np.float32(1.0), scale,
            out=np.zeros_like(scale, dtype=np.float32),
            where=scale != 0,
        )
        codes = np.rint(blocks * inverse).astype(np.int16) + 1
        codes = np.clip(codes, 0, 3).astype(np.uint8)
        packed = np.bitwise_or.reduce(
            codes.reshape(-1, block_size // 4, 4) << shifts,
            axis=2,
        )
        output[start:stop] = np.concatenate(
            [scale.astype(np.float16).view(np.uint8).reshape(-1, 2), packed],
            axis=1,
        ).reshape(stop - start, -1)
    if label:
        print(
            f"[tensor] {label} out={output.nbytes / 2**20:.1f}MiB "
            f"scan={_LAST_SCAN_SECONDS:.2f}s "
            f"reconstruct={_LAST_RECONSTRUCT_SECONDS:.2f}s "
            f"quantize={time.perf_counter() - started:.2f}s "
            f"rss={rss_bytes() / 2**30:.2f}GiB",
            flush=True,
        )
    return output.reshape(
        *original_shape[:-1], columns // block_size * block_bytes
    )


def unpack_q2_0(
    raw: np.ndarray,
    columns: int,
    *,
    block_size: int = DEFAULT_Q2_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode upstream Q2_0 blocks for export verification."""
    if block_size not in {64, 128}:
        raise ValueError(f"unsupported llama.cpp Q2_0 block size: {block_size}")
    if columns % block_size:
        raise ValueError(f"invalid Q2_0 column count: {columns}")
    block_bytes = Q2_SCALE_BYTES + block_size // 4
    blocks = np.asarray(raw, dtype=np.uint8).reshape(-1, block_bytes)
    scales = blocks[:, :2].copy().view(np.float16).reshape(-1)
    packed = blocks[:, 2:]
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    codes = ((packed[..., None] >> shifts) & 3).reshape(-1, block_size)
    return codes, scales


def reconstruct_ternary(
    record: dict[str, Any],
    shape: tuple[int, ...],
    *,
    label: str | None = None,
):
    """Materialize one deployed CAT-Q matrix without consulting base weights."""
    import torch

    if len(shape) != 2:
        raise ValueError(f"trained tensor is not a matrix: {shape}")
    rows, columns = shape
    if columns % TRAIN_GROUP_SIZE:
        raise ValueError(f"trained tensor columns are not group-128 aligned: {shape}")
    codes = record["codes"]
    scales = record["scales"]
    if codes.numel() != rows * columns:
        raise ValueError(
            f"trained code count {codes.numel()} does not match tensor shape {shape}"
        )
    groups = rows * columns // TRAIN_GROUP_SIZE
    if scales.numel() != groups:
        raise ValueError(
            f"trained scale count {scales.numel()} does not match {groups} groups"
        )
    # The timer must start *before* the validation scans.  They are the first
    # thing that touches the checkpoint's pages, so on an mmap-backed file they
    # pay for every fault the reconstruction below then gets for free.  Timing
    # them separately is what turned "stuck at 9%" into a measurement: reading
    # 85 MiB of codes cost ~114 s per tensor over a FUSE mount (4 KiB demand
    # paging, no readahead) and nothing in the log attributed it.
    global _LAST_SCAN_SECONDS, _LAST_RECONSTRUCT_SECONDS
    started = time.perf_counter()
    if not bool(torch.all((codes >= -1) & (codes <= 1))):
        values = torch.unique(codes).tolist()
        raise ValueError(f"non-ternary codes in trained tensor: {values[:10]}")
    if not bool(torch.all(torch.isfinite(scales))):
        raise FloatingPointError("trained scales contain non-finite values")
    scanned = time.perf_counter()
    weight = (
        codes.reshape(groups, TRAIN_GROUP_SIZE).to(torch.float32)
        * scales.reshape(groups, 1).to(torch.float32)
    )
    weight = weight.reshape(rows, columns)
    # Reported by quantize_q2_0, which runs immediately after this for the same
    # tensor, so the whole per-tensor cost lands on a single line.
    _LAST_SCAN_SECONDS = scanned - started
    _LAST_RECONSTRUCT_SECONDS = time.perf_counter() - scanned
    return weight


def register_q2_quantizer(gguf_module, *, row_chunk: int) -> tuple[int, int]:
    """Install the Q2_0 Python writer missing from current gguf-py."""
    qtype = gguf_module.GGMLQuantizationType.Q2_0
    block_size, block_bytes = gguf_module.constants.GGML_QUANT_SIZES[qtype]
    if block_size not in {64, 128} or block_bytes != Q2_SCALE_BYTES + block_size // 4:
        raise RuntimeError(
            "unsupported llama.cpp Q2_0 traits: "
            f"block_size={block_size}, type_size={block_bytes}"
        )

    class Q2Writer:
        @classmethod
        def quantize(cls, tensor):
            if isinstance(tensor, gguf_module.LazyNumpyTensor):
                shape = tuple(int(value) for value in tensor.shape)
                byte_shape = (
                    *shape[:-1],
                    shape[-1] // block_size * block_bytes,
                )
                meta = gguf_module.LazyNumpyTensor.meta_with_dtype_and_shape(
                    np.uint8, byte_shape
                )
                return gguf_module.LazyNumpyTensor(
                    meta=meta,
                    args=(tensor,),
                    func=lambda eager: quantize_q2_0(
                        eager,
                        row_chunk=row_chunk,
                        block_size=block_size,
                        label=f"Q2_0{shape}",
                    ),
                )
            if isinstance(tensor, np.ndarray):
                return quantize_q2_0(
                    tensor,
                    row_chunk=row_chunk,
                    block_size=block_size,
                    label=f"Q2_0{tensor.shape}",
                )
            raise TypeError(f"unsupported Q2_0 input: {type(tensor)}")

    gguf_module.quants._type_traits[qtype] = Q2Writer
    return block_size, block_bytes


STAGE_BUFFER_BYTES = 32 * 2**20


def report_filesystems() -> None:
    """Log mount points and free space.

    Treat these numbers as advisory only.  On HF Jobs ``df`` reports the *node*
    filesystem (measured 1.2 TiB free) while the pod is capped by a separate
    ephemeral-storage quota -- 50 GB, enforced by eviction with the message
    "Pod ephemeral local storage usage exceeds the total limit of containers".
    Staging both the 21.2 GiB checkpoint and 14.9 GiB of base shards alongside
    a 9.2 GiB output exceeded it even though df showed a terabyte spare.
    """
    import shutil

    for label in ("/tmp", "/", "/dev/shm"):
        target = Path(label)
        if not target.is_dir():
            continue
        usage = shutil.disk_usage(target)
        print(
            f"[disk] {label} total={usage.total / 2**30:.1f}GiB "
            f"free={usage.free / 2**30:.1f}GiB",
            flush=True,
        )


def stage_file(source: Path, destination: Path, *, label: str) -> Path:
    """Copy one file to local disk with large sequential reads.

    Reading a 21 GiB checkpoint straight off a FUSE mount is the reason exports
    crawled: ``torch.load(mmap=True)`` turns every tensor access into 4 KiB
    demand paging over the network, measured at well under 1 MiB/s.  One
    sequential copy with a large buffer pays the transfer once and lets every
    subsequent read come from local page cache.
    """
    import shutil

    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    free = shutil.disk_usage(destination.parent).free
    if free < size + 2**30:
        raise RuntimeError(
            f"staging {label} needs {size / 2**30:.1f}GiB plus headroom; "
            f"{destination.parent} has {free / 2**30:.1f}GiB free"
        )
    started = time.perf_counter()
    print(f"[stage] start {label} {size / 2**30:.2f}GiB -> {destination}", flush=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, STAGE_BUFFER_BYTES)
    elapsed = time.perf_counter() - started
    print(
        f"[stage] done {label} {size / 2**30:.2f}GiB in {elapsed:.1f}s "
        f"({size / max(elapsed, 1e-6) / 2**20:.1f}MiB/s)",
        flush=True,
    )
    return destination


def load_ternary(
    path_or_uri: str,
    *,
    stage_dir: Path | None = None,
    trained_before_layer: int | None = None,
):
    import torch

    path = Path(path_or_uri)
    if path_or_uri.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        path = (stage_dir or Path("/tmp")) / "ternary-prefix.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        HfFileSystem(token=os.environ.get("HF_TOKEN")).get(path_or_uri, str(path))
    if not path.is_file():
        raise FileNotFoundError(f"ternary checkpoint is missing: {path}")
    if stage_dir is not None and not path.is_relative_to(stage_dir):
        path = stage_file(path, stage_dir / path.name, label="ternary checkpoint")
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    normalized = {
        checkpoint_weight_name(name): {
            "codes": record["codes"],
            "scales": record["scales"],
        }
        for name, record in raw.items()
    }
    if len(normalized) != len(raw):
        raise RuntimeError("normalizing trained names produced a collision")
    if trained_before_layer is None:
        trained = normalized
    else:
        trained = {
            name: record
            for name, record in normalized.items()
            if (layer := decoder_layer(name)) is not None
            and layer < trained_before_layer
        }
        dropped = len(normalized) - len(trained)
        print(
            f"[checkpoint] selected {len(trained)} modules before layer "
            f"{trained_before_layer}; excluded {dropped}",
            flush=True,
        )
    return trained, path, len(normalized)


def safetensors_shard_bytes(base_path: Path) -> dict[str, dict[str, int]]:
    """Map each shard file to the byte size of every tensor it holds."""
    from safetensors import safe_open

    index_path = base_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shards = ["model.safetensors"]
    sizes: dict[str, dict[str, int]] = {}
    for shard in shards:
        with safe_open(str(base_path / shard), framework="pt") as reader:
            per_tensor = {}
            for name in reader.keys():
                slice_ = reader.get_slice(name)
                count = 1
                for dimension in slice_.get_shape():
                    count *= int(dimension)
                per_tensor[name] = count * _DTYPE_BYTES.get(
                    slice_.get_dtype(), 2
                )
            sizes[shard] = per_tensor
    return sizes


_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def stage_base(
    base_path: Path,
    stage_dir: Path,
    trained: dict[str, Any],
    *,
    min_shard_bytes: int = 64 * 2**20,
) -> Path:
    """Mirror the base model locally, copying only the shards that get read.

    Trained matrices are reconstructed from the checkpoint, so their base
    weights are never materialized -- llama.cpp reads those shards' headers and
    nothing else.  Every shard does hold a few kilobytes of untrained norms,
    which is why the copy decision is made on *untrained bytes per shard*
    rather than on whether a shard contains any untrained tensor at all.
    Shards below the threshold are symlinked: their handful of norm reads cost
    a couple of page faults, while copying them would mean transferring the
    whole 54 GiB base to get at a few megabytes.
    """
    destination = stage_dir / f"base-{base_path.name}"
    destination.mkdir(parents=True, exist_ok=True)
    shard_sizes = safetensors_shard_bytes(base_path)
    staged_bytes = 0
    linked = 0
    for shard, per_tensor in sorted(shard_sizes.items()):
        untrained = sum(
            size
            for name, size in per_tensor.items()
            if normalized_hf_name(name) not in trained
        )
        target = destination / shard
        if target.exists() or target.is_symlink():
            target.unlink()
        if untrained >= min_shard_bytes:
            stage_file(
                base_path / shard,
                target,
                label=f"base {shard} ({untrained / 2**20:.0f}MiB needed)",
            )
            staged_bytes += (base_path / shard).stat().st_size
        else:
            target.symlink_to((base_path / shard).resolve())
            linked += 1
    for extra in sorted(base_path.iterdir()):
        if extra.name in shard_sizes or extra.is_dir():
            continue
        target = destination / extra.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.write_bytes(extra.read_bytes())
    print(
        f"[stage] base ready: copied {staged_bytes / 2**30:.2f}GiB, "
        f"symlinked {linked} shard(s) read for headers only -> {destination}",
        flush=True,
    )
    return destination


def resolve_base(base: str) -> tuple[Path, str | None]:
    path = Path(base)
    if path.is_dir():
        return path, None
    from huggingface_hub import snapshot_download

    # Download the weights too.  Passing a repo id back as remote_hf_model_id
    # makes llama.cpp fetch every weight byte with HTTP range requests, which
    # is the same per-access-latency trap as reading them over a FUSE mount.
    local = snapshot_download(repo_id=base, token=os.environ.get("HF_TOKEN"))
    return Path(local), None


def install_llama_imports(llama_cpp: Path):
    if not (llama_cpp / "convert_hf_to_gguf.py").is_file():
        raise FileNotFoundError(
            f"--llama-cpp does not contain convert_hf_to_gguf.py: {llama_cpp}"
        )
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    sys.path.insert(0, str(llama_cpp))
    import gguf
    from conversion.base import LazyTorchTensor
    from conversion import ModelType, get_model_architecture, get_model_class

    return gguf, LazyTorchTensor, ModelType, get_model_architecture, get_model_class


def verify_gguf(
    path: Path,
    *,
    expected_types: dict[str, int],
    q2_block_size: int,
) -> dict[str, Any]:
    import gguf

    reader = gguf.GGUFReader(str(path))
    q2_type = int(gguf.GGMLQuantizationType.Q2_0)
    q4_types = {
        int(gguf.GGMLQuantizationType.Q4_0),
        int(gguf.GGMLQuantizationType.Q4_1),
        int(gguf.GGMLQuantizationType.Q4_K),
    }
    q2 = [tensor for tensor in reader.tensors if int(tensor.tensor_type) == q2_type]
    q4 = [tensor for tensor in reader.tensors if int(tensor.tensor_type) in q4_types]
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(set(expected_types) - set(by_name))
    if missing:
        raise RuntimeError(f"GGUF is missing forced tensors: {missing[:5]}")
    wrong = [
        (name, int(by_name[name].tensor_type), expected)
        for name, expected in expected_types.items()
        if int(by_name[name].tensor_type) != expected
    ]
    if wrong:
        raise RuntimeError(f"GGUF forced tensor types do not match: {wrong[:5]}")
    sample_indices = sorted({0, len(q2) // 2, len(q2) - 1}) if q2 else []
    samples = []
    for index in sample_indices:
        tensor = q2[index]
        columns = int(tensor.shape[0])
        codes, scales = unpack_q2_0(
            tensor.data, columns, block_size=q2_block_size
        )
        if int(codes.max(initial=0)) > 2:
            raise RuntimeError(f"Q2_0 tensor {tensor.name} contains forbidden code 3")
        if not np.isfinite(scales).all():
            raise FloatingPointError(f"Q2_0 tensor {tensor.name} has non-finite scales")
        samples.append({
            "name": tensor.name,
            "codes": sorted(int(value) for value in np.unique(codes)),
            "scale_min": float(scales.min(initial=np.float16(0))),
            "scale_max": float(scales.max(initial=np.float16(0))),
        })
    return {
        "tensors": len(reader.tensors),
        "q2_0_block_size": q2_block_size,
        "q2_0_tensors": len(q2),
        "q4_tensors": len(q4),
        "forced_tensor_types": {
            str(kind): sum(1 for value in expected_types.values() if value == kind)
            for kind in sorted(set(expected_types.values()))
        },
        "samples": samples,
    }


def native_verify_gguf(llama_cli: Path, path: Path) -> dict[str, Any]:
    """Require the C++ runtime—not only gguf-py—to load the artifact."""
    if not llama_cli.is_file():
        raise FileNotFoundError(f"llama.cpp native loader is missing: {llama_cli}")
    version = subprocess.run(
        [str(llama_cli), "--version"],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    loaded = subprocess.run(
        [
            str(llama_cli),
            "-m", str(path),
            "-n", "0",
            "-c", "512",
            "--simple-io",
        ],
        input="/exit\n",
        text=True,
        capture_output=True,
        timeout=300,
    )
    if loaded.returncode:
        detail = (loaded.stdout + "\n" + loaded.stderr)[-4000:]
        raise RuntimeError(
            f"native llama.cpp load failed with exit {loaded.returncode}:\n{detail}"
        )
    return {
        "status": "PASS",
        "loader": str(llama_cli),
        "version": (version.stdout + version.stderr).strip().splitlines()[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llama-cpp", required=True, type=Path,
                    help="current upstream llama.cpp checkout")
    ap.add_argument("--ternary-prefix", required=True,
                    help="local ternary.pt or hf:// bucket object")
    ap.add_argument("--base", required=True,
                    help="local BF16 HF directory or Hugging Face model ID")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--suffix-from-layer", type=int, required=True,
                    help="first decoder layer retained at --suffix-type")
    ap.add_argument(
        "--suffix-type",
        choices=("q4_1", "q4_0", "f16", "bf16"),
        default="q4_1",
        help="precision for decoder linears at and after --suffix-from-layer",
    )
    ap.add_argument(
        "--top-type",
        choices=("q4_1", "q4_0", "f16", "bf16"),
        default="q4_1",
        help="precision for token embeddings and lm_head (default: q4_1)",
    )
    ap.add_argument(
        "--filter-full-checkpoint",
        action="store_true",
        help="accept a deeper checkpoint and select only trained modules before "
             "--suffix-from-layer; required for post-hoc Hybrid exports",
    )
    ap.add_argument("--q2-row-chunk", type=int, default=16,
                    help="matrix rows per eager Q2_0 packing slice")
    ap.add_argument("--expected-prefix-modules", type=int, default=0)
    ap.add_argument("--expected-prefix-parameters", type=int, default=0)
    ap.add_argument("--expected-suffix-modules", type=int, default=0)
    ap.add_argument("--expected-top-modules", type=int, default=0)
    ap.add_argument("--model-name",
                    help="GGUF metadata name (defaults to the output filename stem)")
    ap.add_argument("--repo", help="optional Hub model repository")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--llama-cli",
        type=Path,
        help="native llama.cpp loader; defaults to LLAMA_CPP/build/bin/llama-cli",
    )
    ap.add_argument(
        "--skip-native-verify",
        action="store_true",
        help="skip the required C++ loader check (development only)",
    )
    ap.add_argument(
        "--stage-dir", type=Path, default=Path("/tmp/export-stage"),
        help="local directory to copy FUSE-mounted inputs into before reading "
             "them; pass --no-stage to read them in place",
    )
    ap.add_argument("--no-stage", action="store_true",
                    help="read inputs in place (only safe on local disk)")
    ap.add_argument(
        "--stage-base", action="store_true",
        help="also copy the base model's heavyweight shards locally.  Off by "
             "default: the pod's 50 GB ephemeral-storage quota does not fit "
             "the 21.2 GiB checkpoint, 14.9 GiB of base shards and a 9.2 GiB "
             "output at once, and the checkpoint is the read that actually "
             "matters -- it is scanned per trained tensor, while base shards "
             "are read once in large sequential slices.",
    )
    args = ap.parse_args()
    if args.model_name is None:
        args.model_name = args.out.stem
    stage_dir = None if args.no_stage else args.stage_dir

    logging.basicConfig(level=logging.INFO)
    import torch

    report_filesystems()
    trained, trained_path, checkpoint_modules = load_ternary(
        args.ternary_prefix,
        stage_dir=stage_dir,
        trained_before_layer=(
            args.suffix_from_layer if args.filter_full_checkpoint else None
        ),
    )
    layers = sorted({layer for name in trained if (layer := decoder_layer(name)) is not None})
    expected_layers = list(range(args.suffix_from_layer))
    if layers != expected_layers:
        raise RuntimeError(
            f"trained layers are {layers[:3]}..{layers[-3:] if layers else []}; "
            f"expected exactly 0..{args.suffix_from_layer - 1}"
        )
    if args.expected_prefix_modules and len(trained) != args.expected_prefix_modules:
        raise RuntimeError(
            f"checkpoint has {len(trained)} modules; expected {args.expected_prefix_modules}"
        )
    trained_parameters = sum(int(record["codes"].numel()) for record in trained.values())
    if args.expected_prefix_parameters and trained_parameters != args.expected_prefix_parameters:
        raise RuntimeError(
            f"checkpoint has {trained_parameters} parameters; "
            f"expected {args.expected_prefix_parameters}"
        )

    (
        gguf,
        LazyTorchTensor,
        ModelType,
        get_model_architecture,
        get_model_class,
    ) = install_llama_imports(args.llama_cpp)
    q2_block_size, q2_block_bytes = register_q2_quantizer(
        gguf, row_chunk=args.q2_row_chunk
    )
    base_path, remote_repo = resolve_base(args.base)
    if (
        args.stage_base
        and stage_dir is not None
        and not base_path.is_relative_to(stage_dir)
    ):
        base_path = stage_base(base_path, stage_dir, trained)
    hparams = json.loads((base_path / "config.json").read_text())
    architecture = get_model_architecture(hparams, ModelType.TEXT)
    base_class = get_model_class(architecture)
    if gguf.MODEL_ARCH_NAMES[base_class.model_arch] != "qwen35":
        raise RuntimeError(
            f"base resolves to {gguf.MODEL_ARCH_NAMES[base_class.model_arch]!r}, "
            "expected llama.cpp qwen35"
        )

    qtypes = {
        "q4_1": gguf.GGMLQuantizationType.Q4_1,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "f16": gguf.GGMLQuantizationType.F16,
        "bf16": gguf.GGMLQuantizationType.BF16,
    }
    suffix_qtype = qtypes[args.suffix_type]
    top_qtype = qtypes[args.top_type]

    class MixedPrecisionModel(base_class):
        # llama.cpp deliberately requires every concrete converter subclass to
        # declare this attribute in its own class body; inheriting it is not
        # accepted by Model.__init_subclass__.
        model_arch = base_class.model_arch
        no_mtp = True

        def __init__(self, *model_args, **model_kwargs):
            self._tq_trained = trained
            self._tq_q2_source_names: set[str] = set()
            self._tq_q2_names: set[str] = set()
            self._tq_q4_names: set[str] = set()
            self._tq_suffix_names: set[str] = set()
            self._tq_top_names: set[str] = set()
            self._tq_forced_types: dict[str, int] = {}
            super().__init__(*model_args, **model_kwargs)

        def get_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
            for name, generator in self.model_tensors.items():
                if name not in self._tq_trained:
                    yield name, generator()
                    continue
                # generator() returns llama.cpp's lazy tensor. Reading its
                # metadata shape does not materialize the BF16 base weight.
                source = generator()
                shape = tuple(int(value) for value in source.shape)
                record = self._tq_trained[name]
                meta = LazyTorchTensor.meta_with_dtype_and_shape(torch.float32, shape)
                lazy = LazyTorchTensor(
                    meta=meta,
                    args=(record, shape, name),
                    func=lambda rec, shp, tensor_name: reconstruct_ternary(
                        rec, shp, label=tensor_name
                    ),
                )
                yield name, lazy

        def tensor_force_quant(self, name, new_name, bid, n_dims):
            if name in self._tq_trained:
                if n_dims != 2:
                    raise RuntimeError(f"trained tensor {name} has {n_dims} dimensions")
                self._tq_q2_source_names.add(name)
                self._tq_q2_names.add(new_name)
                self._tq_forced_types[new_name] = int(
                    gguf.GGMLQuantizationType.Q2_0
                )
                return gguf.GGMLQuantizationType.Q2_0
            layer = decoder_layer(name)
            suffix_linear = (
                layer is not None
                and layer >= args.suffix_from_layer
                and is_decoder_linear(name)
            )
            if n_dims == 2 and suffix_linear:
                self._tq_suffix_names.add(new_name)
                self._tq_forced_types[new_name] = int(suffix_qtype)
                if suffix_qtype in {
                    gguf.GGMLQuantizationType.Q4_0,
                    gguf.GGMLQuantizationType.Q4_1,
                }:
                    self._tq_q4_names.add(new_name)
                return suffix_qtype
            top_weight = name in {"model.embed_tokens.weight", "lm_head.weight"}
            if n_dims == 2 and top_weight:
                self._tq_top_names.add(new_name)
                self._tq_forced_types[new_name] = int(top_qtype)
                if top_qtype in {
                    gguf.GGMLQuantizationType.Q4_0,
                    gguf.GGMLQuantizationType.Q4_1,
                }:
                    self._tq_q4_names.add(new_name)
                return top_qtype
            return super().tensor_force_quant(name, new_name, bid, n_dims)

        def prepare_metadata(self, vocab_only):
            # llama.cpp's HF converter only accepts source conversion modes
            # that it knows how to apply as a default (F32/F16/BF16/Q8/TQ).
            # Individual tensors above are still forced to Q2_0 or Q4.  Once
            # tensor conversion is complete, record the artifact's dominant
            # on-disk type as Q2_0 in the GGUF metadata.
            self.ftype = gguf.LlamaFileType.MOSTLY_Q2_0
            return super().prepare_metadata(vocab_only)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    model = MixedPrecisionModel(
        base_path,
        gguf.LlamaFileType.MOSTLY_BF16,
        args.out,
        model_name=args.model_name,
        remote_hf_model_id=remote_repo,
    )
    source_names = {normalized_hf_name(name) for name in model.model_tensors}
    unmatched = sorted(set(trained) - source_names)
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} trained tensors do not match base-model weights: "
            f"{unmatched[:5]}"
        )
    model.write()

    if model._tq_q2_source_names != set(trained):
        missing = sorted(set(trained) - model._tq_q2_source_names)
        raise RuntimeError(
            f"only {len(model._tq_q2_source_names)} trained tensors were written: "
            f"{missing[:5]}"
        )
    if (
        args.expected_suffix_modules
        and len(model._tq_suffix_names) != args.expected_suffix_modules
    ):
        raise RuntimeError(
            f"wrote {len(model._tq_suffix_names)} suffix modules; "
            f"expected {args.expected_suffix_modules}"
        )
    if args.expected_top_modules and len(model._tq_top_names) != args.expected_top_modules:
        raise RuntimeError(
            f"wrote {len(model._tq_top_names)} top-level modules; "
            f"expected {args.expected_top_modules}"
        )

    verification = verify_gguf(
        args.out,
        expected_types=model._tq_forced_types,
        q2_block_size=q2_block_size,
    )
    if not args.skip_native_verify:
        llama_cli = args.llama_cli or args.llama_cpp / "build/bin/llama-cli"
        verification["native_loader"] = native_verify_gguf(llama_cli, args.out)
    manifest = {
        "name": args.model_name,
        "base": args.base,
        "ternary_prefix": args.ternary_prefix,
        "ternary_prefix_file": str(trained_path),
        "checkpoint_modules": checkpoint_modules,
        "checkpoint_filtered": bool(args.filter_full_checkpoint),
        "suffix_from_layer": args.suffix_from_layer,
        "suffix_type": args.suffix_type,
        "top_type": args.top_type,
        "llama_cpp_commit": subprocess.run(
            ["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "q2_0_block_size": q2_block_size,
        "q2_0_block_bytes": q2_block_bytes,
        "q2_row_chunk": args.q2_row_chunk,
        "ternary_modules": len(model._tq_q2_names),
        "ternary_parameters": trained_parameters,
        "suffix_modules": len(model._tq_suffix_names),
        "top_modules": len(model._tq_top_names),
        "bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "verification": verification,
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.out),
            path_in_repo=args.out.name,
            repo_id=args.repo,
            commit_message=f"Add {args.model_name} GGUF",
        )
        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo=manifest_path.name,
            repo_id=args.repo,
            commit_message="Add GGUF verification manifest",
        )
        print(f"pushed https://huggingface.co/{args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
