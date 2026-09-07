#!/usr/bin/env python3
"""Generate and filter agentic AYOT traces with the FP16 target model itself."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .trace_data import read_jsonl

TOOL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION_BLOCK = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
PARAMETER_BLOCK = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _coerce_xml_parameters(
    raw: dict[str, str], input_schema: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None]:
    properties = (input_schema or {}).get("properties", {})
    result: dict[str, Any] = {}
    for name, value in raw.items():
        value = value.strip()
        expected_type = properties.get(name, {}).get("type")
        if expected_type in {"integer", "number", "boolean", "object", "array"}:
            try:
                result[name] = json.loads(value)
            except json.JSONDecodeError:
                return None, f"parameter {name!r} is not valid {expected_type} JSON"
        else:
            result[name] = value
    return result, None


def parse_native_call(
    output: str, input_schema: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse either Qwen3 JSON calls or Qwen3.5's XML-parameter calls."""
    blocks = TOOL_BLOCK.findall(output)
    if len(blocks) != 1:
        return None, f"expected one tool-call block, found {len(blocks)}"
    body = blocks[0].strip()
    if body.startswith("<function="):
        function = FUNCTION_BLOCK.fullmatch(body)
        if function is None or function.group(1).strip() == "":
            return None, "expected exactly one complete function block"
        name, parameter_text = function.groups()
        matches = PARAMETER_BLOCK.findall(parameter_text)
        if PARAMETER_BLOCK.sub("", parameter_text).strip():
            return None, "unexpected text outside parameter blocks"
        params: dict[str, str] = {}
        for param_name, value in matches:
            param_name = param_name.strip()
            if not param_name:
                return None, "tool call has an empty parameter name"
            if param_name in params:
                return None, f"parameter {param_name!r} appears more than once"
            params[param_name] = value
        arguments, error = _coerce_xml_parameters(params, input_schema)
        if error:
            return None, error
        return {"name": name.strip(), "arguments": arguments}, None
    try:
        call = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"invalid tool-call JSON: {exc.msg}"
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None, "tool call has no string name"
    if not isinstance(call.get("arguments"), dict):
        return None, "tool call arguments are not an object"
    return call, None


def validate_completion(task: dict[str, Any], output: str) -> tuple[bool, str]:
    expected = task.get("expected")
    if expected is None:
        if "<tool_call>" in output:
            return False, "direct task called a tool"
        visible = output.partition("</think>")[2] if "</think>" in output else output
        if not visible.strip():
            return False, "direct task has no visible answer"
        return True, "direct answer"

    schema = next(
        tool["function"]["parameters"]
        for tool in task["tools"]
        if tool["function"]["name"] == expected["name"]
    )
    call, error = parse_native_call(output, schema)
    if error:
        return False, error
    assert call is not None
    if call != expected:
        return False, f"call mismatch: expected {expected!r}, got {call!r}"
    errors = list(Draft202012Validator(schema).iter_errors(call["arguments"]))
    if errors:
        return False, f"schema error: {errors[0].message}"
    return True, "exact schema-valid call"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append to an existing output and skip task IDs already recorded",
    )
    args = parser.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    tasks = read_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    existing = read_jsonl(args.output) if args.resume and args.output.exists() else []
    by_id = {row["task_id"]: row for row in existing}
    for task in tasks:
        old = by_id.get(task["id"])
        if old and old.get("task_fingerprint") != task_fingerprint(task):
            raise ValueError(
                f"cannot resume: task definition changed or fingerprint is missing for {task['id']}"
            )
    pending_tasks = [task for task in tasks if task["id"] not in by_id]
    model, tokenizer = load(str(args.model))
    sampler = make_sampler(temp=0.0)
    eos = tokenizer.eos_token or ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = sum(row.get("accepted") is True for row in existing)
    rejected = sum(row.get("accepted") is False for row in existing)
    tool_ok = sum(row.get("accepted") is True and row.get("kind") == "tool" for row in existing)
    direct_ok = sum(
        row.get("accepted") is True and row.get("kind") == "direct" for row in existing
    )
    tokens = sum(row.get("tokens", 0) for row in existing if row.get("accepted") is True)
    started = time.monotonic()
    mode = "a" if existing else "w"
    with args.output.open(mode) as handle:
        for index, task in enumerate(pending_tasks, 1):
            prompt = tokenizer.apply_chat_template(
                task["messages"],
                tools=task["tools"],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=args.thinking,
                preserve_thinking=False,
            )
            completion = generate(
                model,
                tokenizer,
                tokenizer.encode(prompt, add_special_tokens=False),
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            ok, reason = validate_completion(task, completion)
            trace_text = prompt + completion + eos
            token_count = len(tokenizer.encode(trace_text, add_special_tokens=False))
            row = {
                "version": 1,
                "task_id": task["id"],
                "task_fingerprint": task_fingerprint(task),
                "kind": task["kind"],
                "accepted": ok,
                "reason": reason,
                "expected": task.get("expected"),
                "completion": completion,
                "text": trace_text if ok else None,
                "tokens": token_count,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            accepted += int(ok)
            rejected += int(not ok)
            tool_ok += int(ok and task["kind"] == "tool")
            direct_ok += int(ok and task["kind"] == "direct")
            tokens += token_count if ok else 0
            print(
                f"[{index}/{len(pending_tasks)}] {'KEEP' if ok else 'DROP'} {task['id']} "
                f"tokens={token_count} {reason}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    summary = {
        "tasks": len(existing) + len(pending_tasks),
        "resumed_records": len(existing),
        "generated_records": len(pending_tasks),
        "accepted": accepted,
        "rejected": rejected,
        "tool_accepted": tool_ok,
        "direct_accepted": direct_ok,
        "accepted_tokens": tokens,
        "elapsed_seconds": round(elapsed, 3),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output} and {summary_path}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
