#!/usr/bin/env python3
"""Run myAgent table-QA experiments across multiple local vLLM endpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


TASK_DEFAULTS = {
    "wtq": "datasets_ready/full/wtq_unseen.jsonl",
    "tabfact": "datasets_ready/full/tabfact_test.jsonl",
    "crt": "datasets_ready/full/crt.jsonl",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_contiguous(rows: List[Dict[str, Any]], shards: int) -> List[List[Dict[str, Any]]]:
    result: List[List[Dict[str, Any]]] = []
    total = len(rows)
    for index in range(shards):
        start = total * index // shards
        end = total * (index + 1) // shards
        result.append(rows[start:end])
    return result


def endpoint_list(raw: str) -> List[str]:
    endpoints = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not endpoints:
        raise ValueError("--endpoints must contain at least one endpoint")
    return endpoints


def task_list(raw: str) -> List[str]:
    tasks = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [task for task in tasks if task not in TASK_DEFAULTS]
    if invalid:
        raise ValueError(f"Unsupported task(s): {invalid}. Supported: {sorted(TASK_DEFAULTS)}")
    return tasks


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "model"


def merge_outputs(reference_rows: List[Dict[str, Any]], shard_outputs: List[Path], merged_path: Path) -> None:
    by_id: Dict[str, Dict[str, Any]] = {}
    order = [str(row.get("id") or "") for row in reference_rows]
    for output in shard_outputs:
        for row in read_jsonl(output):
            row_id = str(row.get("id") or "")
            if row_id:
                by_id[row_id] = row
    missing = [row_id for row_id in order if row_id and row_id not in by_id]
    if missing:
        raise RuntimeError(f"{merged_path.name}: missing {len(missing)} rows, first missing id={missing[0]}")
    write_jsonl(merged_path, (by_id[row_id] for row_id in order if row_id))


def run_eval(repo_root: Path, merged_path: Path, eval_path: Path) -> None:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "code" / "evaluate_results.py"),
        str(merged_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    eval_path.write_text(completed.stdout, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="Repository root on the server.")
    parser.add_argument("--tasks", default="wtq,tabfact,crt", help="Comma-separated tasks.")
    parser.add_argument("--endpoints", required=True, help="Comma-separated OpenAI-compatible base URLs ending with /v1.")
    parser.add_argument("--model", required=True, help="Served model name configured in vLLM.")
    parser.add_argument("--output-root", required=True, help="Directory for shards, logs, merged outputs, and eval files.")
    parser.add_argument("--api-key-env", default="LOCAL_VLLM_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--mact-avg-tokens", type=float, default=47439.2633)
    parser.add_argument("--max-replan", type=int, default=3)
    parser.add_argument("--collaboration-mode", choices=("legacy", "selective", "calibration"), default="selective")
    parser.add_argument("--disable-strong-verification", action="store_true")
    parser.add_argument("--disable-deterministic-shortcuts", action="store_true")
    parser.add_argument("--enable-multiview-validation", action="store_true")
    parser.add_argument("--limit-per-task", type=int, default=0, help="Optional smoke-test limit before sharding.")
    parser.add_argument("--resume", action="store_true", help="Skip shard output files whose line count matches the shard input.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    endpoints = endpoint_list(args.endpoints)
    tasks = task_list(args.tasks)
    model_name_for_path = safe_name(args.model)

    env_key = args.api_key_env
    if not os.getenv(env_key):
        raise RuntimeError(f"Environment variable {env_key} is not set.")

    for task in tasks:
        input_path = repo_root / TASK_DEFAULTS[task]
        rows = read_jsonl(input_path)
        if args.limit_per_task:
            rows = rows[: args.limit_per_task]
        shards = split_contiguous(rows, len(endpoints))

        shard_input_paths: List[Path] = []
        shard_output_paths: List[Path] = []
        processes: List[subprocess.Popen[Any]] = []

        for index, shard_rows in enumerate(shards):
            shard_input = output_root / "shards" / task / f"{task}_shard{index:02d}.jsonl"
            shard_output = output_root / "raw" / task / f"{task}_shard{index:02d}_out.jsonl"
            log_path = output_root / "logs" / task / f"{task}_shard{index:02d}.log"
            write_jsonl(shard_input, shard_rows)
            shard_output.parent.mkdir(parents=True, exist_ok=True)
            shard_input_paths.append(shard_input)
            shard_output_paths.append(shard_output)

            if args.resume and count_jsonl(shard_output) == len(shard_rows):
                print(f"[run] skip completed {task} shard {index}: {len(shard_rows)} rows")
                continue

            cmd = [
                sys.executable,
                str(repo_root / "code" / "tqa.py"),
                "--task",
                task,
                "--dataset_path",
                str(shard_input),
                "--output_path",
                str(shard_output),
                "--plan_model_name",
                args.model,
                "--code_model_name",
                args.model,
                "--model_provider",
                "openai_compatible",
                "--api_base",
                endpoints[index],
                "--api_key_env",
                env_key,
                "--api_timeout",
                str(args.api_timeout),
                "--api_max_retries",
                str(args.api_max_retries),
                "--temperature",
                str(args.temperature),
                "--max_tokens",
                str(args.max_tokens),
                "--mact_avg_tokens",
                str(args.mact_avg_tokens),
                "--max_replan",
                str(args.max_replan),
                "--collaboration_mode",
                args.collaboration_mode,
            ]
            if args.disable_strong_verification:
                cmd.append("--disable_strong_verification")
            if args.disable_deterministic_shortcuts:
                cmd.append("--disable_deterministic_shortcuts")
            if args.enable_multiview_validation:
                cmd.append("--enable_multiview_validation")
            print("[run]", " ".join(cmd))
            if args.dry_run:
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            processes.append(subprocess.Popen(cmd, cwd=str(repo_root), stdout=log_file, stderr=subprocess.STDOUT))

        if args.dry_run:
            continue

        failures = []
        for proc in processes:
            code = proc.wait()
            if code != 0:
                failures.append(code)
        if failures:
            raise RuntimeError(f"{task}: {len(failures)} shard process(es) failed: {failures}")

        merged_path = output_root / "merged" / f"{task}_{model_name_for_path}.jsonl"
        merge_outputs(rows, shard_output_paths, merged_path)
        eval_path = output_root / "eval" / f"{task}_{model_name_for_path}_eval.json"
        run_eval(repo_root, merged_path, eval_path)
        print(f"[run] merged: {merged_path}")
        print(f"[run] eval:   {eval_path}")


if __name__ == "__main__":
    main()
