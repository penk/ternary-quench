# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "accelerate>=0.30.0,<2.0.0",
#     "huggingface-hub>=1.5.0,<2.0",
#     "jsonschema>=4.20",
#     "numpy>=1.26",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "transformers @ git+https://github.com/huggingface/transformers.git@a353632607c59463e6ced86a44c2de3c2cd62d5e",
# ]
# ///
"""Generate, validate, and pack self-generated agentic calibration traces."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("out/traces")

from .trace_data import accepted_trace_texts, pack_sequences, read_jsonl
from .trace_generation import task_fingerprint, validate_completion
from .trace_tasks import build_tasks, write_jsonl


def _first_stop(ids: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    for index, token_id in enumerate(ids):
        if token_id in stop_ids:
            return ids[:index], True
    return ids, False


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _model_loader_kind(model_type: str) -> str:
    return "image_text" if model_type == "qwen3_5" else "causal_lm"


def _summarize(
    rows: list[dict[str, Any]],
    *,
    status: str,
    started: float,
    requested_tasks: int,
    initial_batch_size: int,
    active_batch_size: int,
) -> dict[str, Any]:
    accepted_rows = [row for row in rows if row.get("accepted") is True]
    return {
        "status": status,
        "requested_tasks": requested_tasks,
        "generated_records": len(rows),
        "accepted": len(accepted_rows),
        "rejected": len(rows) - len(accepted_rows),
        "tool_accepted": sum(row.get("kind") == "tool" for row in accepted_rows),
        "direct_accepted": sum(row.get("kind") == "direct" for row in accepted_rows),
        "accepted_tokens": sum(int(row.get("tokens", 0)) for row in accepted_rows),
        "acceptance_rate": len(accepted_rows) / len(rows) if rows else 0.0,
        "initial_batch_size": initial_batch_size,
        "active_batch_size": active_batch_size,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _download_resume_file(repo: str, filename: str, destination: Path) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        cached = hf_hub_download(repo, filename, repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)
    return True


def _upload_progress(api, output: Path, repo: str) -> None:
    """Publish traces and summary together as one repository commit."""
    api.upload_folder(
        folder_path=str(output),
        path_in_repo=".",
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=["traces.jsonl", "summary.json"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True, help="Hub dataset repo for traces and .npy")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--count", type=int, default=2500)
    ap.add_argument("--direct-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--nsamples", type=int, default=512)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--resume-repo",
        help="optional source repo for traces.jsonl; writes still go to --repo",
    )
    ap.add_argument("--upload-every-batches", type=int, default=25)
    ap.add_argument("--final-upload-attempts", type=int, default=12)
    ap.add_argument("--min-tool-accepted", type=int, default=0)
    ap.add_argument("--min-direct-accepted", type=int, default=0)
    ap.add_argument("--public", action="store_true", help="publish publicly; default is private")
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.upload_every_batches <= 0:
        raise ValueError("upload-every-batches must be positive")
    if args.final_upload_attempts <= 0:
        raise ValueError("final-upload-attempts must be positive")

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this script is the HF Jobs GPU generator")

    out = args.output
    tasks_path = out / "tasks.jsonl"
    traces_path = out / "traces.jsonl"
    summary_path = out / "summary.json"
    artifact_path = out / f"agentic-{args.nsamples}x{args.seqlen}.npy"
    manifest_path = artifact_path.with_suffix(".json")
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    if args.resume:
        resume_repo = args.resume_repo or args.repo
        restored = _download_resume_file(resume_repo, "traces.jsonl", traces_path)
        print(
            f"Hub resume from {resume_repo}: "
            f"{'restored traces.jsonl' if restored else 'no checkpoint'}",
            flush=True,
        )

    tasks = build_tasks(args.count, direct_fraction=args.direct_fraction, seed=args.seed)
    write_jsonl(tasks_path, tasks)
    existing = read_jsonl(traces_path) if args.resume and traces_path.exists() else []
    by_id = {row["task_id"]: row for row in existing}
    if len(by_id) != len(existing):
        raise ValueError("resume checkpoint contains duplicate task IDs")
    task_ids = {task["id"] for task in tasks}
    extra_ids = set(by_id) - task_ids
    if extra_ids:
        raise ValueError(
            f"resume checkpoint does not match --count/--seed: {len(extra_ids)} extra task IDs"
        )
    for task in tasks:
        old = by_id.get(task["id"])
        if old and old.get("task_fingerprint") != task_fingerprint(task):
            raise ValueError(f"resume fingerprint mismatch for {task['id']}")
    pending = [task for task in tasks if task["id"] not in by_id]

    device_name = torch.cuda.get_device_name(0)
    print(
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"transformers={transformers.__version__} cuda={device_name}",
        flush=True,
    )
    print(
        f"tasks={len(tasks)} resumed={len(existing)} pending={len(pending)} "
        f"batch={args.batch_size}",
        flush=True,
    )

    config = AutoConfig.from_pretrained(args.model)
    loader_kind = _model_loader_kind(config.model_type)
    model_class = (
        AutoModelForImageTextToText if loader_kind == "image_text" else AutoModelForCausalLM
    )
    print(f"model_type={config.model_type} loader={loader_kind}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()

    generation_eos = model.generation_config.eos_token_id
    if generation_eos is None:
        stop_ids = {int(tokenizer.eos_token_id)}
    elif isinstance(generation_eos, int):
        stop_ids = {generation_eos}
    else:
        stop_ids = {int(token_id) for token_id in generation_eos}
    stop_ids.add(int(tokenizer.eos_token_id))

    started = time.monotonic()
    rows = list(existing)
    mode = "a" if existing else "w"
    active_batch_size = args.batch_size
    completed_batches = 0
    cursor = 0
    with traces_path.open(mode) as handle:
        while cursor < len(pending):
            batch = pending[cursor:cursor + active_batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    task["messages"],
                    tools=task["tools"],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=args.thinking,
                    preserve_thinking=False,
                )
                for task in batch
            ]
            try:
                encoded = tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {name: value.to("cuda") for name, value in encoded.items()}
                prompt_width = encoded["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=sorted(stop_ids),
                    )
                generated_ids = generated[:, prompt_width:].cpu().tolist()
            except RuntimeError as exc:
                if not _is_oom(exc) or active_batch_size == 1:
                    raise
                active_batch_size = max(1, active_batch_size // 2)
                print(f"CUDA OOM; retrying at batch_size={active_batch_size}", flush=True)
                if "encoded" in locals():
                    del encoded
                if "generated" in locals():
                    del generated
                gc.collect()
                torch.cuda.empty_cache()
                continue

            kept = 0
            for task, prompt, token_ids in zip(batch, prompts, generated_ids, strict=True):
                completion_ids, terminated = _first_stop(token_ids, stop_ids)
                completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
                ok, reason = validate_completion(task, completion)
                trace_text = prompt + completion + tokenizer.eos_token
                token_count = len(tokenizer.encode(trace_text, add_special_tokens=False))
                row = {
                    "version": 1,
                    "backend": "transformers-cuda",
                    "task_id": task["id"],
                    "task_fingerprint": task_fingerprint(task),
                    "kind": task["kind"],
                    "accepted": ok,
                    "reason": reason,
                    "expected": task.get("expected"),
                    "completion": completion,
                    "terminated": terminated,
                    "text": trace_text if ok else None,
                    "tokens": token_count,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows.append(row)
                kept += int(ok)
                if not ok:
                    print(f"DROP {task['id']}: {reason}", flush=True)
            handle.flush()
            cursor += len(batch)
            completed_batches += 1
            print(
                f"batch {completed_batches}: {cursor}/{len(pending)} generated, "
                f"{kept}/{len(batch)} kept, batch_size={active_batch_size}",
                flush=True,
            )

            if completed_batches % args.upload_every_batches == 0:
                summary = _summarize(
                    rows,
                    status="generating",
                    started=started,
                    requested_tasks=len(tasks),
                    initial_batch_size=args.batch_size,
                    active_batch_size=active_batch_size,
                )
                _write_summary(summary_path, summary)
                try:
                    _upload_progress(api, out, args.repo)
                except HfHubHTTPError as exc:
                    # Local /data is durable for the lifetime of the job. A transient
                    # Hub failure must not throw away completed GPU generation work.
                    print(
                        f"WARNING progress upload failed; continuing locally: "
                        f"{str(exc).splitlines()[-1]}",
                        flush=True,
                    )
                else:
                    print(f"checkpointed {len(rows)} records to {args.repo}", flush=True)

            del encoded, generated, generated_ids

    summary = _summarize(
        rows,
        status="packing",
        started=started,
        requested_tasks=len(tasks),
        initial_batch_size=args.batch_size,
        active_batch_size=active_batch_size,
    )
    _write_summary(summary_path, summary)
    try:
        _upload_progress(api, out, args.repo)
    except HfHubHTTPError as exc:
        print(
            f"WARNING packing-state upload failed; final upload will retry: "
            f"{str(exc).splitlines()[-1]}",
            flush=True,
        )

    texts = accepted_trace_texts(rows)
    accepted_tool_count = sum(
        row.get("accepted") is True and row.get("kind") == "tool" for row in rows
    )
    accepted_direct_count = sum(
        row.get("accepted") is True and row.get("kind") == "direct" for row in rows
    )
    if accepted_tool_count < args.min_tool_accepted:
        raise RuntimeError(
            f"accepted {accepted_tool_count} tool traces; "
            f"required at least {args.min_tool_accepted}"
        )
    if accepted_direct_count < args.min_direct_accepted:
        raise RuntimeError(
            f"accepted {accepted_direct_count} direct traces; "
            f"required at least {args.min_direct_accepted}"
        )
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    packed, metrics = pack_sequences(
        sequences,
        n_samples=args.nsamples,
        seqlen=args.seqlen,
        pad_token_id=int(tokenizer.eos_token_id),
        seed=args.seed,
    )
    array = np.asarray(packed, dtype=np.int64)
    np.save(artifact_path, array)
    manifest = {
        **metrics,
        "model": args.model,
        "source": "self-generated strict-filtered BF16 traces",
        "model_type": config.model_type,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "generator": "transformers-cuda",
        "task_count": len(tasks),
        "direct_fraction": args.direct_fraction,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary.update({"status": "complete", "artifact": artifact_path.name, "packing": metrics})
    _write_summary(summary_path, summary)

    for attempt in range(1, args.final_upload_attempts + 1):
        try:
            api.upload_folder(
                folder_path=str(out),
                path_in_repo=".",
                repo_id=args.repo,
                repo_type="dataset",
                ignore_patterns=["*.tmp"],
            )
            break
        except HfHubHTTPError as exc:
            if attempt == args.final_upload_attempts:
                raise
            delay = min(300, 30 * 2 ** (attempt - 1))
            print(
                f"WARNING final upload attempt {attempt}/"
                f"{args.final_upload_attempts} failed; retrying in {delay}s: "
                f"{str(exc).splitlines()[-1]}",
                flush=True,
            )
            time.sleep(delay)
    visibility = "" if args.public else " (private)"
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"pushed https://huggingface.co/datasets/{args.repo}{visibility}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
