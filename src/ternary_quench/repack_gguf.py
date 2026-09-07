#!/usr/bin/env python3
"""Repack Q2_0 ternary GGUF tensors as portable Q2_K, preserving code signs.

Q2_K shares an FP16 scale across 256 weights, with 4-bit multipliers for
each 16 weights. Both its scale and minimum use the same multiplier: this
keeps zero exact and preserves the {-s, 0, +s} alphabet in every subgroup.
Group scales change; this is not a lossless conversion. All other tensors
and tokenizer metadata are copied unchanged. No upstream model is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def pack_q2_k(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert current 64-weight/18-byte Q2_0 blocks to 256-weight Q2_K."""
    blocks = np.asarray(raw, dtype=np.uint8).reshape(-1, 4, 18)
    scales = blocks[:, :, :2].copy().view('<f2').reshape(-1, 4).astype(np.float32)
    if not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError('Q2_0 scales must be finite and nonnegative')
    codes = ((blocks[:, :, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3)
    codes = codes.reshape(-1, 256)
    if (codes > 2).any():
        raise ValueError('Q2_0 contains non-ternary code 3')
    d = (scales.max(axis=1) / 15).astype('<f2')
    ratio = np.divide(scales, d.astype(np.float32)[:, None],
                      out=np.zeros_like(scales), where=d[:, None] != 0)
    multipliers = np.clip(np.rint(ratio), 1, 15).astype(np.uint8)
    multipliers[scales == 0] = 0
    if np.any((d == 0) & (scales.max(axis=1) > 0)):
        raise ValueError('Q2_K superblock scale underflows FP16')
    out = np.empty((len(blocks), 84), np.uint8)
    out[:, :16] = np.repeat(multipliers * 17, 4, axis=1)
    out[:, 16:80] = np.bitwise_or.reduce(
        codes.reshape(-1, 2, 4, 32) << np.array([0, 2, 4, 6], np.uint8)[None, None, :, None],
        axis=2).reshape(-1, 64)
    out[:, 80:82] = d.view(np.uint8).reshape(-1, 2)
    out[:, 82:84] = out[:, 80:82]
    restored = d.astype(np.float32)[:, None] * multipliers
    nonzero = (codes.reshape(-1, 4, 64) != 1).sum(axis=2)
    error = float(np.sum((restored - scales).astype(np.float64)**2 * nonzero))
    energy = float(np.sum(scales.astype(np.float64)**2 * nonzero))
    return out, {'squared_error': error, 'squared_norm': energy,
                 'relative_rmse': (error / energy)**0.5 if energy else 0.0}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--llama-cpp', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--chunk-blocks', type=int, default=4096)
    ap.add_argument('--expected-ternary-tensors', type=int, required=True)
    args = ap.parse_args()
    if args.chunk_blocks < 1 or args.out.exists():
        ap.error('chunk-blocks must be positive and output must not exist')
    sys.path.insert(0, str(args.llama_cpp / 'gguf-py'))
    import gguf
    if gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q2_0] != (64, 18):
        raise RuntimeError('This converter requires Q2_0 geometry 64/18')
    reader = gguf.GGUFReader(str(args.source))
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError('Only little-endian GGUF is supported')
    q2 = gguf.GGMLQuantizationType.Q2_0
    qk = gguf.GGMLQuantizationType.Q2_K
    selected = [t for t in reader.tensors if t.tensor_type == q2]
    if not selected or len(selected) != args.expected_ternary_tensors:
        raise ValueError(f'Expected {args.expected_ternary_tensors} Q2_0 tensors, found {len(selected)}')
    for t in selected:
        if int(t.shape[0]) % 256:
            raise ValueError(f'{t.name}: row width must be divisible by 256')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + '.partial')
    if partial.exists():
        raise FileExistsError(partial)
    writer = gguf.GGUFWriter(str(partial), reader.get_field('general.architecture').contents())
    for name, field in reader.fields.items():
        if name.startswith('GGUF.') or name in {'general.architecture', 'general.file_type'}:
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             field.types[-1] if len(field.types) > 1 else None)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q2_K)
    writer.add_string('ternary_quench.encoding', 'Q2_K symmetric ternary subgroups')
    for t in reader.tensors:
        kind = qk if t.tensor_type == q2 else t.tensor_type
        shape = tuple(int(x) for x in t.shape[::-1])
        nbytes = int(np.prod(shape)) // 256 * 84 if kind == qk and t.tensor_type == q2 else t.n_bytes
        writer.add_tensor_info(t.name, shape, np.dtype(np.float32), nbytes, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    records = []
    start = time.monotonic()
    for t in reader.tensors:
        if t.tensor_type != q2:
            writer.write_tensor_data(t.data)
            continue
        source = t.data.reshape(-1, 72)
        packed = np.empty((len(source), 84), np.uint8)
        error = energy = 0.0
        for lo in range(0, len(source), args.chunk_blocks):
            data, stats = pack_q2_k(source[lo:lo + args.chunk_blocks])
            # Independent upstream decoder verifies the actual on-disk layout.
            decoded = gguf.dequantize(data, qk).reshape(-1, 256)
            source_blocks = source[lo:lo + args.chunk_blocks].reshape(-1, 18)
            source_scales = source_blocks[:, :2].copy().view('<f2').astype(np.float32)
            source_codes = ((source_blocks[:, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3).reshape(-1, 64)
            original = ((source_codes.astype(np.float32)-1) * source_scales).reshape(-1, 256)
            if not np.isfinite(decoded).all() or not np.array_equal(np.sign(decoded), np.sign(original)):
                raise ValueError(f'{t.name}: sign/zero preservation failed')
            observed = float(np.sum((decoded.astype(np.float64) - original)**2))
            if not np.isclose(observed, stats['squared_error'], rtol=1e-5, atol=1e-9):
                raise ValueError(f'{t.name}: Q2_K layout verification failed')
            packed[lo:lo + len(data)] = data
            error += stats['squared_error']
            energy += stats['squared_norm']
        writer.write_tensor_data(packed)
        records.append({'name': t.name, 'squared_error': error, 'squared_norm': energy,
                        'relative_rmse': (error / energy)**0.5 if energy else 0.0})
        print(json.dumps({'tensor': t.name, 'done': len(records), 'total': len(selected),
                          'relative_rmse': records[-1]['relative_rmse'],
                          'elapsed_s': round(time.monotonic()-start, 1)}), flush=True)
    writer.close()
    check = gguf.GGUFReader(str(partial))
    if len(check.tensors) != len(reader.tensors):
        raise ValueError('Output tensor count mismatch')
    for old, new in zip(reader.tensors, check.tensors, strict=True):
        if old.name != new.name or not np.array_equal(old.shape, new.shape):
            raise ValueError('Output tensor name/shape mismatch')
        if old.tensor_type != q2 and (old.tensor_type != new.tensor_type or not np.array_equal(old.data, new.data)):
            raise ValueError(f'{old.name}: an untouched tensor changed')
    partial.rename(args.out)
    report = {'source': str(args.source), 'source_sha256': file_hash(args.source),
              'bytes': args.out.stat().st_size, 'sha256': file_hash(args.out),
              'encoding': 'Q2_K', 'ternary_tensors': len(records),
              'signs_and_zeros_preserved': True, 'other_tensors_byte_identical': True,
              'relative_rmse': (sum(r['squared_error'] for r in records) /
                                max(sum(r['squared_norm'] for r in records), 1e-300))**0.5,
              'tensors': records, 'runtime_verification': 'PENDING'}
    args.out.with_suffix(args.out.suffix + '.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'tensors'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
