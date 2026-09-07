# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "accelerate>=0.30.0,<2.0.0",
#     "lm-eval==0.4.7",
#     "numpy>=1.26.0",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "transformers @ git+https://github.com/huggingface/transformers.git@a353632607c59463e6ced86a44c2de3c2cd62d5e",
# ]
# ///
"""Run PIQA, ARC-e/c, HellaSwag and WinoGrande zero-shot on a HF model.

lm-eval is pinned to 0.4.7 to preserve the published result protocol. Its unused
multimodal loader is stubbed because that version predates Transformers 5.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from importlib.metadata import version
from pathlib import Path

import numpy as np

# Must precede any lm_eval import.
sys.modules.setdefault(
    "lm_eval.models.hf_vlms", types.ModuleType("lm_eval.models.hf_vlms")
)

TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande")

# Namespaced parquet datasets replace legacy script-based dataset names that
# datasets 5.x no longer loads.
DATASET_OVERRIDES = {
    "piqa": {
        "dataset_path": "ybisk/piqa",
        "dataset_name": None,
        "dataset_kwargs": {"revision": "refs/convert/parquet"},
    },
    "hellaswag": {"dataset_path": "Rowan/hellaswag"},
    "winogrande": {"dataset_path": "allenai/winogrande"},
    # arc_easy / arc_challenge already point at allenai/ai2_arc.
}


def task_specs():
    """The five-task suite, with dataset sources repaired for datasets 5.x."""
    from lm_eval import utils as lm_eval_utils
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_defaults=True)
    specs: list[str | dict] = []
    for task in TASKS:
        override = DATASET_OVERRIDES.get(task)
        if override is None:
            specs.append(task)
            continue
        entry = manager.task_index[task]
        config = lm_eval_utils.load_yaml_config(Path(entry["yaml_path"]), mode="full")
        config.update(override)
        specs.append(config)
    return specs


def metrics(raw: dict) -> dict[str, float]:
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    # `hf jobs uv run` reserves --label, hence --run-label here.
    ap.add_argument("--run-label", required=True, dest="label")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--repo", help="dataset/model repo to upload the result JSON to")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use ternary-quench-evaluate-mlx for MLX")
    torch.manual_seed(args.seed)
    output = args.output or Path(f"/tmp/five-task-{args.label}.json")

    token = os.environ.get("HF_TOKEN")
    print(f"loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(
        f"loaded on {torch.cuda.get_device_name(0)}; "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params",
        flush=True,
    )

    evaluated = simple_evaluate(
        model=HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        tasks=task_specs(),
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
    scored = metrics(raw)
    payload = {
        "label": args.label,
        "model": args.model,
        "protocol": {
            "tasks": list(TASKS),
            "num_fewshot": 0,
            "apply_chat_template": False,
            "piqa_dataset": "ybisk/piqa@refs/convert/parquet (default)",
            "metric": "acc_norm,none with acc,none fallback",
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "limit": args.limit,
            "seed": args.seed,
            "runtime": "native pytorch (HFLM)",
            "dtype": "bfloat16",
            "lm_eval": version("lm-eval"),
            "transformers": version("transformers"),
            "torch": torch.__version__,
            "hf_vlms_stubbed": True,
        },
        "metrics": scored,
        "raw": raw,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(scored, indent=2), flush=True)

    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(output),
            path_in_repo=output.name,
            repo_id=args.repo,
            commit_message=f"Add {args.label} five-task accuracy",
        )
        print(f"pushed https://huggingface.co/{args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
