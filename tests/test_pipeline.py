import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("jsonschema")

from ternary_quench import generate_traces
from ternary_quench.export_gguf import (
    checkpoint_weight_name,
    load_ternary,
    quantize_q2_0,
    unpack_q2_0,
)
from ternary_quench.trace_data import accepted_trace_texts, pack_sequences, read_jsonl
from ternary_quench.trace_generation import parse_native_call, validate_completion
from ternary_quench.trace_tasks import TRAIN_TOOLS, build_tasks
from ternary_quench.train import build_window_scheduler, resume_advance_segments


def test_agentic_tasks_are_deterministic_and_held_out():
    first = build_tasks(100, direct_fraction=0.2, seed=2)
    assert first == build_tasks(100, direct_fraction=0.2, seed=2)
    names = {tool["function"]["name"] for tool in TRAIN_TOOLS}
    assert names.isdisjoint({"get_weather", "calculate", "write_file", "search_web"})


def test_trace_generator_can_report_interpreter_version():
    assert generate_traces.sys.version


def test_trace_filter_keeps_only_unique_accepted_rows(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps({"accepted": True, "text": "kept"}) + "\n"
        + json.dumps({"accepted": True, "text": "kept"}) + "\n"
        + json.dumps({"accepted": False, "text": "drop"}) + "\n"
    )
    assert accepted_trace_texts(read_jsonl(path)) == ["kept"]


def test_trace_packer_never_splits_a_trace():
    sequences = [[11, 12, 13], [21, 22], [31, 32, 33], [41, 42]]
    rows, metrics = pack_sequences(
        sequences, n_samples=2, seqlen=5, pad_token_id=0, seed=3, lookahead=4
    )
    assert metrics["padding_tokens"] == 0
    for sequence in sequences:
        assert any(
            row[start:start + len(sequence)] == sequence
            for row in rows
            for start in range(len(row) - len(sequence) + 1)
        )


def test_qwen3_and_qwen35_tool_formats_validate():
    call, error = parse_native_call(
        '<tool_call>{"name":"query_inventory","arguments":{"sku":"A"}}</tool_call>'
    )
    assert error is None
    assert call == {"name": "query_inventory", "arguments": {"sku": "A"}}

    task = next(
        row for row in build_tasks(60, direct_fraction=0, seed=2)
        if row["expected"]["name"] == "reserve_room"
    )
    expected = task["expected"]
    output = (
        "<tool_call><function=reserve_room>"
        f"<parameter=room>{expected['arguments']['room']}</parameter>"
        f"<parameter=people>{expected['arguments']['people']}</parameter>"
        "</function></tool_call>"
    )
    assert validate_completion(task, output)[0]


def test_resume_schedule_replays_each_original_boundary():
    rounds = build_window_scheduler(6, 4, 2, 4)
    assert resume_advance_segments(rounds, 6) == [(0, 2), (2, 3), (3, 4)]


def test_llama_cpp_q2_0_round_trip():
    scales = np.array([0.125, 0.5, 0.03125, 1.0], dtype=np.float32)
    codes = np.resize(np.array([-1, 0, 1, 0], dtype=np.float32), (4, 64))
    weight = (codes * scales[:, None]).reshape(2, 128)
    raw = quantize_q2_0(weight, row_chunk=1)
    np.testing.assert_array_equal(raw, quantize_q2_0(weight, row_chunk=16))
    packed_codes, packed_scales = unpack_q2_0(raw, 128)
    restored = (
        (packed_codes.astype(np.int8) - 1) * packed_scales[:, None]
    ).reshape(2, 128)
    np.testing.assert_array_equal(restored, weight)


def test_legacy_llama_cpp_q2_0_group128_round_trip():
    scale = np.float32(0.125)
    codes = np.resize(np.array([-1, 0, 1, 0], dtype=np.float32), (2, 128))
    weight = codes * scale
    raw = quantize_q2_0(weight, block_size=128)
    packed_codes, packed_scales = unpack_q2_0(raw, 128, block_size=128)
    restored = (
        (packed_codes.astype(np.int8) - 1) * packed_scales[:, None]
    ).reshape(2, 128)
    assert raw.shape == (2, 34)
    np.testing.assert_array_equal(restored, weight)


def test_full_checkpoint_filter_builds_hybrid_prefix(tmp_path):
    record = {
        "codes": torch.zeros((1, 128), dtype=torch.int8),
        "scales": torch.ones((1, 1), dtype=torch.float32),
    }
    path = tmp_path / "ternary.pt"
    torch.save(
        {
            "model.language_model.layers.61.mlp.up_proj": record,
            "model.language_model.layers.62.mlp.up_proj": record,
        },
        path,
    )
    trained, loaded_path, checkpoint_modules = load_ternary(
        str(path), trained_before_layer=62
    )
    assert loaded_path == path
    assert checkpoint_modules == 2
    assert list(trained) == ["model.layers.61.mlp.up_proj.weight"]


def test_checkpoint_module_stem_becomes_normalized_weight_name():
    assert checkpoint_weight_name(
        "model.language_model.layers.7.mlp.up_proj"
    ) == "model.layers.7.mlp.up_proj.weight"
    assert checkpoint_weight_name(
        "model.layers.7.mlp.up_proj.weight"
    ) == "model.layers.7.mlp.up_proj.weight"
