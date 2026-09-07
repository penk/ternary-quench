import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'src/ternary_quench' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_q2_k(raw):
    """Independent reference matching the GGML C decoder's indexing."""
    output = []
    for block in raw:
        d, dm = block[80:].copy().view('<f2').astype(np.float32)
        values = []
        for half in range(2):
            for shift in range(4):
                for group in range(2):
                    scale = int(block[half*8 + shift*2 + group])
                    for i in range(16):
                        q = (int(block[16 + half*32 + group*16 + i]) >> (shift*2)) & 3
                        values.append(d*(scale & 15)*q - dm*(scale >> 4))
        output.append(values)
    return np.array(output)


@pytest.mark.parametrize('scales', [[0, 0, 0, 0], [1, 1, 1, 1], [.125, .25, .5, 1], [.0001, .1, .7, .9]])
def test_q2_k_preserves_codes_with_independent_decoder(scales):
    q2, repack = load('export_gguf'), load('repack_gguf')
    rng = np.random.default_rng(11)
    signs = rng.integers(-1, 2, size=(3, 4, 64)).astype(np.float32)
    original = (signs * np.array(scales, np.float32)[None, :, None]).reshape(3, 256)
    raw = q2.quantize_q2_0(original)
    packed, stats = repack.pack_q2_k(raw)
    decoded = decode_q2_k(packed)
    codes, sf = q2.unpack_q2_0(raw, 256)
    reference = ((codes.astype(np.float32)-1)*sf[:, None]).reshape(3, 256)
    assert packed.shape == (3, 84)
    assert np.array_equal(np.sign(reference), np.sign(decoded))
    assert np.sum((decoded.astype(np.float64)-reference)**2) == pytest.approx(stats['squared_error'], rel=1e-6, abs=1e-9)


def test_reject_non_ternary_codes():
    raw = np.zeros((4, 18), np.uint8)
    raw[0, 2] = 3
    with pytest.raises(ValueError, match='non-ternary'):
        load('repack_gguf').pack_q2_k(raw)


def test_streaming_file_preserves_metadata_shapes_and_untouched_bytes(tmp_path, monkeypatch):
    import json
    import sys
    gguf = pytest.importorskip('gguf')
    q2, repack = load('export_gguf'), load('repack_gguf')
    source, output = tmp_path/'source.gguf', tmp_path/'output.gguf'
    writer = gguf.GGUFWriter(str(source), 'qwen35')
    writer.add_string('tokenizer.chat_template', 'template\n{{ messages }}')
    writer.add_array('tokenizer.ggml.tokens', ['a', 'b', 'c'])
    writer.add_uint32('qwen35.block_count', 1)
    x = np.resize(np.array([-.25, 0, .25], np.float32), (2, 256))
    writer.add_tensor('blk.0.ffn_up.weight', q2.quantize_q2_0(x), raw_dtype=gguf.GGMLQuantizationType.Q2_0)
    writer.add_tensor('output_norm.weight', np.array([.5, 1, 2], np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    monkeypatch.setattr(sys, 'argv', ['repack', '--llama-cpp', str(tmp_path), '--source', str(source),
                                     '--out', str(output), '--expected-ternary-tensors', '1', '--chunk-blocks', '1'])
    assert repack.main() == 0
    reader = gguf.GGUFReader(str(output))
    assert reader.get_field('tokenizer.chat_template').contents() == 'template\n{{ messages }}'
    assert reader.get_field('tokenizer.ggml.tokens').contents() == ['a', 'b', 'c']
    assert list(reader.tensors[0].shape) == [256, 2]
    assert reader.tensors[0].tensor_type == gguf.GGMLQuantizationType.Q2_K
    assert list(reader.tensors[1].data) == [.5, 1, 2]
    assert json.loads(output.with_suffix('.gguf.json').read_text())['other_tensors_byte_identical']
