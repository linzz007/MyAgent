#!/usr/bin/env python3
"""Run MACT one-sample wrapper across multiple endpoints and merge in order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, IO, Iterable, List, Tuple

from run_mact_one_by_one import count_jsonl, load_jsonl


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def endpoint_list(raw: str) -> List[str]:
    endpoints = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not endpoints:
        raise ValueError("--endpoints must contain at least one endpoint")
    return endpoints


def split_contiguous(rows: List[Dict[str, Any]], shards: int) -> List[Tuple[int, List[Dict[str, Any]]]]:
    if shards < 1:
        raise ValueError("shards must be positive")
    result: List[Tuple[int, List[Dict[str, Any]]]] = []
    total = len(rows)
    for index in range(shards):
        start = total * index // shards
        end = total * (index + 1) // shards
        result.append((start, rows[start:end]))
    return result


def build_one_by_one_command(
    *,
    runner_path: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    output_path: Path,
    log_path: Path,
    endpoint: str,
) -> List[str]:
    return [
        sys.executable,
        str(runner_path),
        "--mact-root",
        args.mact_root,
        "--dataset-path",
        str(dataset_path),
        "--output-path",
        str(output_path),
        "--log-path",
        str(log_path),
        "--task",
        args.task,
        "--python-executable",
        args.python_executable,
        "--plan-model-name",
        args.plan_model_name,
        "--code-model-name",
        args.code_model_name,
        "--model-provider",
        args.model_provider,
        "--api-base",
        endpoint,
        "--api-key-env",
        args.api_key_env,
        "--thinking",
        args.thinking,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--api-timeout",
        str(args.api_timeout),
        "--api-max-retries",
        str(args.api_max_retries),
        "--plan-sample",
        str(args.plan_sample),
        "--code-sample",
        str(args.code_sample),
        "--max-step",
        str(args.max_step),
        "--max-actual-step",
        str(args.max_actual_step),
        "--temp-dir",
        args.temp_dir,
        "--resume",
    ]


def merge_outputs(
    *,
    output_path: Path,
    prefix_rows: List[Dict[str, Any]],
    shard_output_paths: List[Path],
) -> None:
    merged_rows = list(prefix_rows)
    for shard_output in shard_output_paths:
        merged_rows.extend(load_jsonl(shard_output))
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    write_jsonl(tmp_path, merged_rows)
    tmp_path.replace(output_path)


def run_dataset(args: argparse.Namespace) -> None:
    endpoints = endpoint_list(args.endpoints)
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    log_dir = Path(args.log_dir)
    shard_root = Path(args.shard_dir)
    runner_path = Path(__file__).resolve().with_name("run_mact_one_by_one.py")

    samples = load_jsonl(dataset_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    if output_path.exists() and not args.resume:
        raise SystemExit(f"Output already exists; pass --resume: {output_path}")

    prefix_rows = load_jsonl(output_path) if args.resume and output_path.exists() else []
    if len(prefix_rows) > len(samples):
        raise SystemExit(
            f"Output has {len(prefix_rows)} rows, but dataset has {len(samples)} rows."
        )

    remaining = samples[len(prefix_rows) :]
    if not remaining:
        print(f"[mact-sharded] complete: {output_path} already has {len(prefix_rows)} rows")
        return

    run_root = shard_root / f"{args.task}_{output_path.stem}_{len(prefix_rows):05d}_{len(samples):05d}"
    input_dir = run_root / "input"
    output_dir = run_root / "output"
    stdout_dir = run_root / "stdout"
    shard_specs = split_contiguous(remaining, len(endpoints))
    processes: List[Tuple[subprocess.Popen[Any], Path, Path, IO[str]]] = []
    shard_output_paths: List[Path] = []
    active_shards: List[Tuple[int, List[Dict[str, Any]]]] = []

    for shard_index, (relative_start, shard_rows) in enumerate(shard_specs):
        if not shard_rows:
            continue
        active_shards.append((relative_start, shard_rows))
        absolute_start = len(prefix_rows) + relative_start
        absolute_end = absolute_start + len(shard_rows)
        shard_input = input_dir / f"shard{shard_index:02d}_{absolute_start:05d}_{absolute_end:05d}.jsonl"
        shard_output = output_dir / f"shard{shard_index:02d}_{absolute_start:05d}_{absolute_end:05d}.jsonl"
        shard_log = log_dir / f"{output_path.stem}_shard{shard_index:02d}_{absolute_start:05d}_{absolute_end:05d}.log"
        shard_stdout = stdout_dir / f"shard{shard_index:02d}_{absolute_start:05d}_{absolute_end:05d}.stdout.log"
        write_jsonl(shard_input, shard_rows)
        shard_output_paths.append(shard_output)

        if args.resume and count_jsonl(shard_output) == len(shard_rows):
            print(f"[mact-sharded] skip shard {shard_index}: {absolute_start}-{absolute_end}")
            continue

        command = build_one_by_one_command(
            runner_path=runner_path,
            args=args,
            dataset_path=shard_input,
            output_path=shard_output,
            log_path=shard_log,
            endpoint=endpoints[shard_index],
        )
        print(
            f"[mact-sharded] start shard {shard_index}: rows {absolute_start}-{absolute_end} "
            f"on {endpoints[shard_index]}",
            flush=True,
        )
        shard_stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = shard_stdout.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=args.myagent_root,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((proc, shard_output, shard_stdout, stdout_handle))

    failures = []
    for proc, shard_output, shard_stdout, stdout_handle in processes:
        try:
            code = proc.wait()
        finally:
            stdout_handle.close()
        if code != 0:
            failures.append((code, shard_output, shard_stdout))
    if failures:
        details = ", ".join(f"{path} rc={code} stdout={stdout}" for code, path, stdout in failures)
        raise RuntimeError(f"{len(failures)} MACT shard process(es) failed: {details}")

    for (_relative_start, shard_rows), shard_output in zip(active_shards, shard_output_paths):
        if len(shard_rows) and count_jsonl(shard_output) != len(shard_rows):
            raise RuntimeError(
                f"{shard_output} has {count_jsonl(shard_output)} rows; expected {len(shard_rows)}"
            )

    merge_outputs(
        output_path=output_path,
        prefix_rows=prefix_rows,
        shard_output_paths=shard_output_paths,
    )
    print(f"[mact-sharded] merged: {output_path} ({count_jsonl(output_path)} rows)")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--myagent-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--mact-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--task", required=True, choices=["wtq", "crt", "scitab"])
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--plan-model-name", required=True)
    parser.add_argument("--code-model-name", required=True)
    parser.add_argument("--model-provider", default="openai_compatible")
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--api-key-env", default="LOCAL_VLLM_API_KEY")
    parser.add_argument("--thinking", default="disabled", choices=["disabled", "enabled"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-max-retries", type=int, default=5)
    parser.add_argument("--plan-sample", type=int, default=1)
    parser.add_argument("--code-sample", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--max-actual-step", type=int, default=3)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    run_dataset(create_parser().parse_args())


if __name__ == "__main__":
    main()
