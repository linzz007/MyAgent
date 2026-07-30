#!/usr/bin/env python3
"""Summarize myAgent Gate-10/Gate-50 eval outputs and choose the next gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


TASKS = ("wtq", "tabfact", "crt")
DEFAULT_REFERENCE_CORRECT = 124
DEFAULT_MACT_AVG_TOKENS = 11262.41
DEFAULT_MAX_FAILURE_RATE = 0.02
DEFAULT_MAX_TOKEN_RATIO = 0.75


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_eval_file(eval_dir: Path, task: str) -> Path | None:
    matches = sorted(eval_dir.glob(f"{task}_*_eval.json"))
    return matches[0] if matches else None


def correct_count(evaluation: Mapping[str, Any]) -> int:
    if "correct" in evaluation:
        return int(evaluation["correct"])
    rows = int(evaluation.get("num_with_gold") or evaluation.get("num_samples") or 0)
    accuracy = float(evaluation.get("primary_accuracy") or 0.0)
    return int(round(rows * accuracy))


def summarize_eval(path: Path) -> dict[str, Any]:
    evaluation = read_json(path)
    rows = int(evaluation.get("num_with_gold") or evaluation.get("num_samples") or 0)
    failed = int(evaluation.get("num_failed_exec") or 0)
    missing = int(evaluation.get("num_missing_answer") or 0)
    return {
        "eval_path": str(path),
        "rows": rows,
        "correct": correct_count(evaluation),
        "accuracy": float(evaluation.get("primary_accuracy") or 0.0),
        "avg_total_tokens": float(evaluation.get("avg_total_tokens") or 0.0),
        "num_failed_exec": failed,
        "num_missing_answer": missing,
        "bad_rows": max(failed, missing),
    }


def weighted_average(items: list[dict[str, Any]], key: str) -> float:
    rows = sum(item["rows"] for item in items)
    if rows <= 0:
        return 0.0
    return sum(item[key] * item["rows"] for item in items) / rows


def choose_decision(
    *,
    missing_tasks: list[str],
    correct: int,
    rows: int,
    bad_rows: int,
    token_ratio: float,
    reference_correct: int,
    max_failure_rate: float,
    max_token_ratio: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if missing_tasks:
        return "incomplete", [f"missing_eval:{task}" for task in missing_tasks]
    failure_rate = bad_rows / rows if rows else 1.0
    if correct < reference_correct:
        reasons.append("overall_correct_below_reference")
    if failure_rate > max_failure_rate:
        reasons.append("failure_rate_above_threshold")
    if token_ratio > max_token_ratio:
        reasons.append("token_ratio_above_threshold")
    return ("no-go", reasons) if reasons else ("gate150", ["gate50_criteria_passed"])


def summarize_gate_results(
    *,
    gate_root: Path,
    model_tag: str,
    reference_correct: int = DEFAULT_REFERENCE_CORRECT,
    mact_avg_tokens: float = DEFAULT_MACT_AVG_TOKENS,
    max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
    max_token_ratio: float = DEFAULT_MAX_TOKEN_RATIO,
) -> dict[str, Any]:
    eval_dir = gate_root / "eval"
    per_dataset: dict[str, dict[str, Any]] = {}
    missing_tasks: list[str] = []
    for task in TASKS:
        eval_path = find_eval_file(eval_dir, task)
        if eval_path is None:
            missing_tasks.append(task)
            continue
        per_dataset[task] = summarize_eval(eval_path)

    items = list(per_dataset.values())
    rows = sum(item["rows"] for item in items)
    correct = sum(item["correct"] for item in items)
    bad_rows = sum(item["bad_rows"] for item in items)
    avg_tokens = weighted_average(items, "avg_total_tokens")
    token_ratio = avg_tokens / mact_avg_tokens if mact_avg_tokens else 0.0
    decision, reasons = choose_decision(
        missing_tasks=missing_tasks,
        correct=correct,
        rows=rows,
        bad_rows=bad_rows,
        token_ratio=token_ratio,
        reference_correct=reference_correct,
        max_failure_rate=max_failure_rate,
        max_token_ratio=max_token_ratio,
    )

    return {
        "model_tag": model_tag,
        "gate_root": str(gate_root),
        "per_dataset": per_dataset,
        "missing_tasks": missing_tasks,
        "overall": {
            "correct": correct,
            "rows": rows,
            "accuracy": correct / rows if rows else 0.0,
            "avg_total_tokens": avg_tokens,
            "mact_avg_tokens_reference": mact_avg_tokens,
            "token_ratio_to_mact": token_ratio,
            "bad_rows": bad_rows,
            "failure_rate": bad_rows / rows if rows else 1.0,
        },
        "criteria": {
            "reference_correct": reference_correct,
            "max_failure_rate": max_failure_rate,
            "max_token_ratio": max_token_ratio,
        },
        "decision": decision,
        "decision_reasons": reasons,
    }


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def render_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        f"# Gate-50 Summary: {summary['model_tag']}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| correct | {overall['correct']}/{overall['rows']} |",
        f"| accuracy | {format_percent(overall['accuracy'])} |",
        f"| avg total tokens | {overall['avg_total_tokens']:.2f} |",
        f"| token ratio to MACT | {overall['token_ratio_to_mact']:.4f} |",
        f"| bad rows | {overall['bad_rows']} |",
        f"| decision | {summary['decision']} |",
        "",
        "| dataset | correct | accuracy | avg tokens | bad rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        result = summary["per_dataset"].get(task)
        if not result:
            lines.append(f"| {task} | missing | missing | missing | missing |")
            continue
        lines.append(
            f"| {task} | {result['correct']}/{result['rows']} | "
            f"{format_percent(result['accuracy'])} | {result['avg_total_tokens']:.2f} | {result['bad_rows']} |"
        )
    lines.extend(
        [
            "",
            "Decision reasons: " + ", ".join(summary["decision_reasons"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--reference-correct", type=int, default=DEFAULT_REFERENCE_CORRECT)
    parser.add_argument("--mact-avg-tokens", type=float, default=DEFAULT_MACT_AVG_TOKENS)
    parser.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE)
    parser.add_argument("--max-token-ratio", type=float, default=DEFAULT_MAX_TOKEN_RATIO)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_gate_results(
        gate_root=args.gate_root.resolve(),
        model_tag=args.model_tag,
        reference_correct=args.reference_correct,
        mact_avg_tokens=args.mact_avg_tokens,
        max_failure_rate=args.max_failure_rate,
        max_token_ratio=args.max_token_ratio,
    )
    output_text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
