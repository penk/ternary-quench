#!/usr/bin/env python3
"""Run the five zero-shot accuracy tasks on an MLX model."""

from __future__ import annotations

import argparse
import json
import os
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np

TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande")


def _task_specs():
    """Match BitTern/projects/cat-q/quantize/utils.py."""
    import lm_eval
    from lm_eval import utils as lm_eval_utils

    specs: list[str | dict] = []
    for task in TASKS:
        if task != "piqa":
            specs.append(task)
            continue
        path = Path(lm_eval.__file__).parent / "tasks" / "piqa" / "piqa.yaml"
        config = lm_eval_utils.load_yaml_config(path, mode="full")
        config["dataset_name"] = "plain_text"
        specs.append(config)
    return specs


def _metrics(raw: dict) -> dict[str, float]:
    out = {}
    for task in TASKS:
        result = raw[task]
        value = result.get("acc_norm,none", result.get("acc,none"))
        if value is None:
            raise KeyError(f"{task}: neither acc_norm,none nor acc,none is present")
        out[task] = round(float(value), 6)
    out["avg_5"] = round(sum(out[task] for task in TASKS) / len(TASKS), 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)

    import lm_eval
    import mlx.core as mx
    from lm_eval.tasks import TaskManager
    from mlx_lm.evaluate import MLXLM

    mx.random.seed(args.seed)
    model = MLXLM(
        str(args.model),
        batch_size=args.batch_size,
        use_chat_template=False,
    )
    evaluated = lm_eval.simple_evaluate(
        model=model,
        tasks=_task_specs(),
        batch_size=args.batch_size,
        num_fewshot=0,
        limit=args.limit,
        task_manager=TaskManager(include_defaults=True),
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    raw = evaluated["results"]
    metrics = _metrics(raw)
    payload = {
        "label": args.label,
        "model": str(args.model.resolve()),
        "protocol": {
            "tasks": list(TASKS),
            "num_fewshot": 0,
            "apply_chat_template": False,
            "piqa_dataset_name": "plain_text",
            "metric": "acc_norm,none with acc,none fallback",
            "batch_size": args.batch_size,
            "limit": args.limit,
            "seed": args.seed,
            "lm_eval": version("lm-eval"),
            "mlx_lm": version("mlx-lm"),
        },
        "metrics": metrics,
        "raw": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
