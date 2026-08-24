"""Run MACT one sample at a time and preserve one output row per input row."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from robust_outputs import apply_fallback_answer, attach_robust_fields  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    return len(load_jsonl(path))


def failure_row(
    sample: Dict[str, Any],
    *,
    error_message: str,
    elapsed_seconds: float,
    returncode: int,
    log_path: Path,
) -> Dict[str, Any]:
    row = dict(sample)
    row["pred_answer"] = ""
    row["history"] = ""
    row["pred_answer_all"] = []
    row["elapsed_seconds_total"] = elapsed_seconds
    row["api_metrics"] = {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    row["exec_error"] = error_message[:2000]
    row["mact_error"] = {
        "returncode": returncode,
        "log_path": str(log_path),
    }
    return apply_fallback_answer(
        row,
        task=str(sample.get("source_dataset") or ""),
        error_message=row["exec_error"],
        retry_count=0,
        fallback_reason="mact_wrapper_failure",
    )


def build_mact_command(
    *,
    python_executable: Path,
    mact_root: Path,
    task: str,
    dataset_path: Path,
    output_path: Path,
    plan_model_name: str,
    code_model_name: str,
    model_provider: str,
    api_base: str,
    api_key_env: str,
    thinking: str,
    temperature: float,
    max_tokens: int,
    api_timeout: float,
    api_max_retries: int,
    plan_sample: int,
    code_sample: int,
    max_step: int,
    max_actual_step: int,
) -> List[str]:
    return [
        str(python_executable),
        str(mact_root / "code" / "tqa.py"),
        "--task",
        task,
        "--dataset_path",
        str(dataset_path),
        "--output_path",
        str(output_path),
        "--plan_model_name",
        plan_model_name,
        "--code_model_name",
        code_model_name,
        "--model_provider",
        model_provider,
        "--api_base",
        api_base,
        "--api_key_env",
        api_key_env,
        "--thinking",
        thinking,
        "--temperature",
        str(temperature),
        "--max_tokens",
        str(max_tokens),
        "--api_timeout",
        str(api_timeout),
        "--api_max_retries",
        str(api_max_retries),
        "--plan_sample",
        str(plan_sample),
        "--code_sample",
        str(code_sample),
        "--max_step",
        str(max_step),
        "--max_actual_step",
        str(max_actual_step),
    ]


def _write_single_sample(path: Path, sample: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one_sample(
    *,
    sample_index: int,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    temp_dir: Path,
    log_path: Path,
) -> Dict[str, Any]:
    sample_input = temp_dir / f"sample_{sample_index:05d}.jsonl"
    sample_output = temp_dir / f"sample_{sample_index:05d}_out.jsonl"
    _write_single_sample(sample_input, sample)

    command = build_mact_command(
        python_executable=Path(args.python_executable),
        mact_root=Path(args.mact_root),
        task=args.task,
        dataset_path=sample_input,
        output_path=sample_output,
        plan_model_name=args.plan_model_name,
        code_model_name=args.code_model_name,
        model_provider=args.model_provider,
        api_base=args.api_base,
        api_key_env=args.api_key_env,
        thinking=args.thinking,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_timeout=args.api_timeout,
        api_max_retries=args.api_max_retries,
        plan_sample=args.plan_sample,
        code_sample=args.code_sample,
        max_step=args.max_step,
        max_actual_step=args.max_actual_step,
    )

    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n=== sample {sample_index} id={sample.get('id', '')} start ===\n"
        )
        result = subprocess.run(
            command,
            cwd=str(args.mact_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(
            f"=== sample {sample_index} returncode={result.returncode} end ===\n"
        )
    elapsed = time.perf_counter() - started

    output_rows = load_jsonl(sample_output)
    if result.returncode == 0 and len(output_rows) == 1:
        return attach_robust_fields(output_rows[0], fallback_used=False, retry_count=0)

    message = (
        f"MACT sample {sample_index} failed or produced {len(output_rows)} rows "
        f"with returncode {result.returncode}; see {log_path}"
    )
    return failure_row(
        sample,
        error_message=message,
        elapsed_seconds=elapsed,
        returncode=result.returncode,
        log_path=log_path,
    )


def run_dataset(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    samples = load_jsonl(dataset_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    if output_path.exists() and not args.resume:
        raise SystemExit(f"Output already exists; pass --resume: {output_path}")

    start_index = count_jsonl(output_path) if args.resume else 0
    if start_index > len(samples):
        raise SystemExit(
            f"Output has {start_index} rows, but dataset has {len(samples)} rows."
        )

    temp_parent = Path(args.temp_dir) if args.temp_dir else output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"mact_one_by_one_{args.task}_",
        dir=str(temp_parent),
    ) as temp_name:
        temp_dir = Path(temp_name)
        for index in range(start_index, len(samples)):
            row = run_one_sample(
                sample_index=index,
                sample=samples[index],
                args=args,
                temp_dir=temp_dir,
                log_path=log_path,
            )
            append_jsonl(output_path, row)
            status = "failed" if row.get("exec_error") else "ok"
            print(
                f"[mact-one] {args.task} {index + 1}/{len(samples)} {status}",
                flush=True,
            )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mact-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--task", required=True, choices=["wtq", "crt", "scitab"])
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--plan-model-name", required=True)
    parser.add_argument("--code-model-name", required=True)
    parser.add_argument("--model-provider", default="openai_compatible")
    parser.add_argument("--api-base", required=True)
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
    parser.add_argument("--temp-dir", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    run_dataset(create_parser().parse_args())


if __name__ == "__main__":
    main()
