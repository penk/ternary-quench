#!/usr/bin/env python3
"""Build deterministic, held-out-safe tool-use tasks for agentic AYOT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

SYSTEM = (
    "You are a tool-using assistant. Use exactly one tool when the request is "
    "covered by a provided function. Otherwise answer directly and do not call a tool."
)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# Deliberately disjoint from the evaluation probe's weather, arithmetic, file-write,
# and web-search families. The first experiment tests transfer to unseen tools, not
# memorisation of the twelve probe answers.
TRAIN_TOOLS = [
    _tool(
        "query_inventory",
        "Return stock information for an item at a warehouse.",
        {"sku": {"type": "string"}, "warehouse": {"type": "string"}},
        ["sku", "warehouse"],
    ),
    _tool(
        "translate_text",
        "Translate text into a requested language.",
        {"text": {"type": "string"}, "target_language": {"type": "string"}},
        ["text", "target_language"],
    ),
    _tool(
        "schedule_meeting",
        "Schedule a meeting at an ISO-8601 time.",
        {"title": {"type": "string"}, "when": {"type": "string"}},
        ["title", "when"],
    ),
    _tool(
        "get_stock_quote",
        "Get a market quote in the requested currency.",
        {"symbol": {"type": "string"}, "currency": {"type": "string"}},
        ["symbol", "currency"],
    ),
    _tool(
        "plan_route",
        "Plan a route between two named places using a travel mode.",
        {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "mode": {"type": "string", "enum": ["walk", "bike", "transit"]},
        },
        ["origin", "destination", "mode"],
    ),
    _tool(
        "reserve_room",
        "Reserve a named room for a stated number of people.",
        {"room": {"type": "string"}, "people": {"type": "integer"}},
        ["room", "people"],
    ),
]


def _positive(index: int, rng: random.Random) -> dict[str, Any]:
    family = index % len(TRAIN_TOOLS)
    if family == 0:
        sku = f"SKU-{rng.randint(1000, 9999)}"
        warehouse = rng.choice(["Berlin", "Oslo", "Madrid", "Prague"])
        prompt = f"Check inventory for {sku} at the {warehouse} warehouse using a tool."
        name, arguments = "query_inventory", {"sku": sku, "warehouse": warehouse}
    elif family == 1:
        text = rng.choice(["good morning", "where is the station", "thank you", "see you soon"])
        language = rng.choice(["German", "Japanese", "Spanish", "Norwegian"])
        prompt = f"Use a tool to translate exactly {text!r} into {language}."
        name, arguments = "translate_text", {"text": text, "target_language": language}
    elif family == 2:
        title = rng.choice(["design review", "release check", "weekly sync", "incident review"])
        when = (
            f"2026-{rng.randint(9, 12):02d}-{rng.randint(1, 28):02d}"
            f"T{rng.randint(8, 17):02d}:00:00Z"
        )
        prompt = f"Schedule {title!r} for {when} with the appropriate tool."
        name, arguments = "schedule_meeting", {"title": title, "when": when}
    elif family == 3:
        symbol = rng.choice(["AAPL", "AMD", "INTC", "NVDA", "SAP"])
        currency = rng.choice(["USD", "EUR", "GBP"])
        prompt = f"Get a tool-based market quote for {symbol} in {currency}."
        name, arguments = "get_stock_quote", {"symbol": symbol, "currency": currency}
    elif family == 4:
        origin, destination = rng.sample(
            ["Central Station", "Museum Island", "City Hall", "University", "Airport"], 2
        )
        mode = rng.choice(["walk", "bike", "transit"])
        prompt = f"Plan a {mode} route from {origin} to {destination} using a tool."
        name, arguments = "plan_route", {
            "origin": origin,
            "destination": destination,
            "mode": mode,
        }
    else:
        room = rng.choice(["Ada", "Babbage", "Curie", "Hopper", "Turing"])
        people = rng.randint(2, 18)
        prompt = f"Reserve room {room} for exactly {people} people using a tool."
        name, arguments = "reserve_room", {"room": room, "people": people}
    decoys = [tool for tool in TRAIN_TOOLS if tool["function"]["name"] != name]
    selected_tools = [
        next(tool for tool in TRAIN_TOOLS if tool["function"]["name"] == name),
        *rng.sample(decoys, 3),
    ]
    rng.shuffle(selected_tools)
    return {
        "id": f"tool-{index:05d}",
        "kind": "tool",
        "tools": selected_tools,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "expected": {"name": name, "arguments": arguments},
    }


def _direct(index: int, rng: random.Random) -> dict[str, Any]:
    family = index % 4
    if family == 0:
        left = 20 + index
        right = 3 + (index * 7) % 19
        prompt = f"Answer {left} plus {right} directly. Do not use any tool."
    elif family == 1:
        word = f"TOKEN-{index:04d}"
        prompt = f"Reply with exactly {word}. Do not call a function."
    elif family == 2:
        noun = ["compiler", "database", "network", "algorithm", "protocol"][index % 5]
        prompt = (
            f"In one sentence, define a {noun}; this is direct-answer request {index}. "
            "Do not use tools."
        )
    else:
        number = 1000 + index
        prompt = f"State whether {number} is odd or even. Answer directly without tools."
    return {
        "id": f"direct-{index:05d}",
        "kind": "direct",
        "tools": rng.sample(TRAIN_TOOLS, 4),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "expected": None,
    }


def build_tasks(count: int, *, direct_fraction: float = 0.2, seed: int = 2) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= direct_fraction < 1:
        raise ValueError("direct_fraction must be in [0, 1)")
    rng = random.Random(seed)
    direct_count = round(count * direct_fraction)
    positive_count = count - direct_count
    tasks = [_positive(i, rng) for i in range(positive_count)]
    tasks += [_direct(i, rng) for i in range(direct_count)]
    rng.shuffle(tasks)
    return tasks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--direct-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    rows = build_tasks(args.count, direct_fraction=args.direct_fraction, seed=args.seed)
    write_jsonl(args.output, rows)
    positives = sum(row["kind"] == "tool" for row in rows)
    print(f"wrote {len(rows)} tasks to {args.output}: {positives} tool, "
          f"{len(rows) - positives} direct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
