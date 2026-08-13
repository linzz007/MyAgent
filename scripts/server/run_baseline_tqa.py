#!/usr/bin/env python3
"""Run simple table-QA baselines with the common evaluator schema."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_backends import build_llm_fn  # noqa: E402
from my_agents import Calculator, build_df_from_table  # noqa: E402
from run_sharded_tqa import (  # noqa: E402
    TASK_DEFAULTS,
    count_jsonl,
    dataset_path_for_task,
    endpoint_list,
    read_jsonl,
    safe_name,
    split_contiguous,
    task_list,
    write_jsonl,
)


BASELINES = ("direct_cot", "single_agent_pandas")


class BaselineExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_output: str = "",
        code: str = "",
        attempts: List[Dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.code = code
        self.attempts = attempts or []


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def table_context_for_row(row: Dict[str, Any]) -> str:
    values: List[str] = []
    for key in ("entity", "table_title", "title", "caption", "page_title"):
        value = str(row.get(key) or "").strip()
        if value and value not in values:
            values.append(value)
    return " | ".join(values)


def dataframe_for_row(row: Dict[str, Any]) -> pd.DataFrame:
    if "table_text" not in row:
        raise KeyError("row is missing table_text")
    return build_df_from_table(row["table_text"])


def compact_table_text(df: pd.DataFrame, *, max_rows: int = 80) -> str:
    shown = df.head(max_rows)
    table = shown.to_csv(index=False)
    if len(df) > max_rows:
        table += f"\n[truncated: showing first {max_rows} of {len(df)} rows]\n"
    return table


def direct_prompt(row: Dict[str, Any], df: pd.DataFrame) -> str:
    task = str(row.get("source_dataset") or "")
    label_hint = ""
    if task == "tabfact":
        label_hint = "For TabFact, the answer must be exactly true or false."
    question = row.get("question") or row.get("statement") or row.get("utterance") or ""
    return f"""You are the Direct-CoT baseline for table question answering.
Reason from the table text only. Do not write or execute code. Return only a JSON object:
{{"answer": <final answer>, "reasoning_summary": <short summary>}}
{label_hint}

Context: {table_context_for_row(row)}
Question: {question}
Table CSV:
{compact_table_text(df)}
"""


def pandas_prompt(row: Dict[str, Any], df: pd.DataFrame) -> str:
    task = str(row.get("source_dataset") or "")
    label_hint = ""
    if task == "tabfact":
        label_hint = "For TabFact, final_answer_value must be exactly 'true' or 'false'."
    question = row.get("question") or row.get("statement") or row.get("utterance") or ""
    return f"""You are the Single-Agent Pandas baseline for table question answering.
Use one pandas code path only. A pandas DataFrame named df already exists.
Store the final answer in a variable named final_answer_value.
Return only one Python code block. Do not print. Do not import anything except pandas, numpy, math, or re.
Write defensive pandas code:
- Inspect df.columns instead of assuming exact column names.
- Treat table values as strings first; convert numeric text with pandas.to_numeric only after removing units or punctuation.
- Guard empty filters before using iloc, values[0], max, min, idxmax, or idxmin.
- If an exact computation is not possible, assign the best directly supported table answer rather than raising an exception.
{label_hint}

Context: {table_context_for_row(row)}
Question: {question}
DataFrame preview:
{compact_table_text(df)}
"""


def pandas_repair_prompt(
    row: Dict[str, Any],
    df: pd.DataFrame,
    *,
    previous_code: str,
    error_message: str,
) -> str:
    question = row.get("question") or row.get("statement") or row.get("utterance") or ""
    return f"""You are still the same Single-Agent Pandas baseline.
Your previous pandas code failed. Rewrite one safer pandas code block.
A pandas DataFrame named df already exists. Store the final answer in final_answer_value.
Return only one Python code block. Do not print.

Question: {question}
Context: {table_context_for_row(row)}
Error: {error_message}
Previous code:
{previous_code}

DataFrame columns: {list(df.columns)}
DataFrame preview:
{compact_table_text(df)}
"""


def extract_json_payload(text: str) -> Dict[str, Any] | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def extract_direct_answer(text: str) -> Any:
    payload = extract_json_payload(text)
    if payload is not None and "answer" in payload:
        return payload.get("answer")
    match = re.search(r"therefore,\s*the answer is\s*:?\s*(.+)", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip().strip("`")
    return text.strip()


def extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def snapshot(llm_fn: Any) -> Dict[str, int]:
    if hasattr(llm_fn, "snapshot"):
        return dict(llm_fn.snapshot())
    return {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def metric_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in ("request_count", "prompt_tokens", "completion_tokens", "total_tokens")
    }


def gold_answer(row: Dict[str, Any]) -> Any:
    for key in ("gold_answer", "answer", "targetValue", "target_value"):
        if row.get(key) not in (None, ""):
            return row.get(key)
    return ""


def run_direct_cot(row: Dict[str, Any], df: pd.DataFrame, llm_fn: Any) -> Dict[str, Any]:
    prompt = direct_prompt(row, df)
    raw_output = llm_fn(prompt)
    answer = extract_direct_answer(raw_output)
    return {
        "final_answer": answer,
        "final_value": answer,
        "pred_answer": answer,
        "llm_raw_output": raw_output,
        "prompt_chars": len(prompt),
        "exec_success": True,
        "exec_error": None,
    }


def run_single_agent_pandas(
    row: Dict[str, Any],
    df: pd.DataFrame,
    llm_fn: Any,
    *,
    max_code_retries: int = 1,
) -> Dict[str, Any]:
    prompt = pandas_prompt(row, df)
    raw_output = llm_fn(prompt)
    code = extract_python_code(raw_output)
    attempts: List[Dict[str, str]] = []
    for attempt in range(max(0, max_code_retries) + 1):
        try:
            local_env = Calculator._safe_execute(code, df)
            break
        except Exception as exc:  # noqa: BLE001
            message = f"{exc.__class__.__name__}: {exc}"
            attempts.append({"code": code, "error": message, "raw_output": raw_output})
            if attempt >= max(0, max_code_retries):
                raise BaselineExecutionError(
                    message,
                    raw_output=raw_output,
                    code=code,
                    attempts=attempts,
                ) from exc
            repair = pandas_repair_prompt(
                row,
                df,
                previous_code=code,
                error_message=message,
            )
            raw_output = llm_fn(repair)
            code = extract_python_code(raw_output)
    answer = local_env.get("final_answer_value", local_env.get("result"))
    return {
        "final_answer": answer,
        "final_value": answer,
        "pred_answer": answer,
        "llm_raw_output": raw_output,
        "planner_code": code,
        "pandas_attempts": attempts,
        "prompt_chars": len(prompt),
        "exec_success": True,
        "exec_error": None,
    }


def failure_payload(
    row: Dict[str, Any],
    *,
    baseline: str,
    error: BaseException,
    elapsed_seconds: float,
    api_metrics: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    payload = dict(row)
    payload.update(
        {
            "baseline_method": baseline,
            "final_answer": "",
            "final_value": "",
            "pred_answer": "",
            "gold_answer": gold_answer(row),
            "elapsed_seconds_total": elapsed_seconds,
            "exec_success": False,
            "exec_error": f"{error.__class__.__name__}: {error}"[:2000],
            "api_metrics": api_metrics
            or {
                "request_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )
    if isinstance(error, BaselineExecutionError):
        payload["llm_raw_output"] = error.raw_output
        payload["planner_code"] = error.code
        payload["pandas_attempts"] = error.attempts
    return payload


def run_row(
    row: Dict[str, Any],
    *,
    baseline: str,
    llm_fn: Any,
    max_code_retries: int = 1,
) -> Dict[str, Any]:
    started = time.perf_counter()
    before = snapshot(llm_fn)
    try:
        df = dataframe_for_row(row)
        if baseline == "direct_cot":
            result = run_direct_cot(row, df, llm_fn)
        elif baseline == "single_agent_pandas":
            result = run_single_agent_pandas(
                row,
                df,
                llm_fn,
                max_code_retries=max_code_retries,
            )
        else:
            raise ValueError(f"unsupported baseline: {baseline}")
        after = snapshot(llm_fn)
        payload = dict(row)
        payload.update(result)
        payload["baseline_method"] = baseline
        payload["gold_answer"] = gold_answer(row)
        payload["api_metrics"] = metric_delta(before, after)
        payload["elapsed_seconds_total"] = time.perf_counter() - started
        return payload
    except Exception as exc:  # noqa: BLE001
        after = snapshot(llm_fn)
        return failure_payload(
            row,
            baseline=baseline,
            error=exc,
            elapsed_seconds=time.perf_counter() - started,
            api_metrics=metric_delta(before, after),
        )


def run_worker(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    rows = read_jsonl(dataset_path)
    if args.limit:
        rows = rows[: args.limit]

    if output_path.exists() and not args.resume:
        raise SystemExit(f"Output already exists; pass --resume: {output_path}")

    start_index = count_jsonl(output_path) if args.resume else 0
    if start_index > len(rows):
        raise SystemExit(f"Output has {start_index} rows, but input has {len(rows)} rows")

    llm_fn = build_llm_fn(args)
    for index, row in enumerate(rows[start_index:], start=start_index):
        payload = run_row(
            row,
            baseline=args.baseline,
            llm_fn=llm_fn,
            max_code_retries=getattr(args, "max_code_retries", 1),
        )
        append_jsonl(output_path, payload)
        status = "failed" if payload.get("exec_error") else "ok"
        print(f"[baseline] {args.baseline} {args.task} {index + 1}/{len(rows)} {status}", flush=True)


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
    completed = subprocess.run(
        [sys.executable, str(repo_root / "code" / "evaluate_results.py"), str(merged_path)],
        cwd=str(repo_root),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    eval_path.write_text(completed.stdout, encoding="utf-8")


def worker_command(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    task: str,
    shard_input: Path,
    shard_output: Path,
    endpoint: str,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repo-root",
        str(repo_root),
        "--baseline",
        args.baseline,
        "--task",
        task,
        "--dataset-path",
        str(shard_input),
        "--output-path",
        str(shard_output),
        "--plan_model_name",
        args.model,
        "--code_model_name",
        args.model,
        "--model_provider",
        "openai_compatible",
        "--api_base",
        endpoint,
        "--api_key_env",
        args.api_key_env,
        "--thinking",
        args.thinking,
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--api_timeout",
        str(args.api_timeout),
        "--api_max_retries",
        str(args.api_max_retries),
        "--max-code-retries",
        str(args.max_code_retries),
    ]
    if args.resume:
        command.append("--resume")
    return command


def run_orchestrator(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    endpoints = endpoint_list(args.endpoints)
    tasks = task_list(args.tasks)
    model_name_for_path = safe_name(args.model)

    if not args.dry_run and not os.getenv(args.api_key_env):
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set.")

    for task in tasks:
        input_path = dataset_path_for_task(args, repo_root, task)
        rows = read_jsonl(input_path)
        if args.limit_per_task:
            rows = rows[: args.limit_per_task]
        shards = split_contiguous(rows, len(endpoints))
        shard_outputs: List[Path] = []
        processes: List[subprocess.Popen[Any]] = []

        for index, shard_rows in enumerate(shards):
            shard_input = output_root / "shards" / args.baseline / task / f"{task}_shard{index:02d}.jsonl"
            shard_output = output_root / "raw" / args.baseline / task / f"{task}_shard{index:02d}_out.jsonl"
            log_path = output_root / "logs" / args.baseline / task / f"{task}_shard{index:02d}.log"
            write_jsonl(shard_input, shard_rows)
            shard_output.parent.mkdir(parents=True, exist_ok=True)
            shard_outputs.append(shard_output)

            if args.resume and count_jsonl(shard_output) == len(shard_rows):
                print(f"[baseline] skip completed {args.baseline} {task} shard {index}: {len(shard_rows)} rows")
                continue

            command = worker_command(
                args=args,
                repo_root=repo_root,
                task=task,
                shard_input=shard_input,
                shard_output=shard_output,
                endpoint=endpoints[index],
            )
            print("[baseline-run]", " ".join(command))
            if args.dry_run:
                continue
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            processes.append(subprocess.Popen(command, cwd=str(repo_root), stdout=log_file, stderr=subprocess.STDOUT))

        if args.dry_run:
            continue

        failures = []
        for proc in processes:
            code = proc.wait()
            if code != 0:
                failures.append(code)
        if failures:
            raise RuntimeError(f"{args.baseline} {task}: {len(failures)} shard process(es) failed: {failures}")

        merged_path = output_root / "merged" / f"{task}_{model_name_for_path}.jsonl"
        merge_outputs(rows, shard_outputs, merged_path)
        eval_path = output_root / "eval" / f"{task}_{model_name_for_path}_eval.json"
        run_eval(repo_root, merged_path, eval_path)
        print(f"[baseline] merged: {merged_path}")
        print(f"[baseline] eval:   {eval_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default=".", help="Repository root on the server.")
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--tasks", default="wtq,tabfact,crt")
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="wtq")
    parser.add_argument("--wtq-dataset", default="")
    parser.add_argument("--tabfact-dataset", default="")
    parser.add_argument("--crt-dataset", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--endpoints", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--api-key-env", "--api_key_env", dest="api_key_env", default="LOCAL_VLLM_API_KEY")
    parser.add_argument("--api-timeout", "--api_timeout", dest="api_timeout", type=float, default=180.0)
    parser.add_argument("--api-max-retries", "--api_max_retries", dest="api_max_retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=1024)
    parser.add_argument("--thinking", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--plan_model_name", default="")
    parser.add_argument("--code_model_name", default="")
    parser.add_argument("--model_provider", default="openai_compatible")
    parser.add_argument("--api_base", default="")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-code-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if not args.dataset_path or not args.output_path:
            raise SystemExit("--worker requires --dataset-path and --output-path")
        if not args.plan_model_name:
            args.plan_model_name = args.model
        run_worker(args)
        return
    if not args.endpoints or not args.model or not args.output_root:
        raise SystemExit("orchestrator mode requires --endpoints, --model, and --output-root")
    run_orchestrator(args)


if __name__ == "__main__":
    main()
