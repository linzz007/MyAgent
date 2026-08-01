#!/usr/bin/env python3
"""Summarize WTQ targeted-fix fresh validation after run_sharded_tqa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from evaluate_results import dataset_accuracy  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/home/ubuntu/lzz/MACT/outputs/server_runs/"
    "qwen3_32b_policy_v6b_newseed_gate50_20260801_0305"
)
DEFAULT_OUTPUT_ROOT_NAME = "myagent_wtq_targeted_fix"
DEFAULT_INPUT_RELATIVE = "input/wtq_p4b_targeted_fix_affected_slice.jsonl"
DEFAULT_PROJECTION_NAME = "p4b_wtq_targeted_fix_projection.json"
DEFAULT_SUMMARY_JSON = "p4b_wtq_targeted_fresh_summary.json"
DEFAULT_SUMMARY_MD = "p4b_wtq_targeted_fresh_summary.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_single(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {path / pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one file matching {path / pattern}, found {len(matches)}")
    return matches[0]


def _default_merged_path(output_root: Path) -> Path:
    exact = output_root / "merged" / "wtq_qwen3-32b-local.jsonl"
    return exact if exact.exists() else _find_single(output_root / "merged", "wtq_*.jsonl")


def _default_eval_path(output_root: Path) -> Path:
    exact = output_root / "eval" / "wtq_qwen3-32b-local_eval.json"
    return exact if exact.exists() else _find_single(output_root / "eval", "wtq_*_eval.json")


def _correct_count(evaluation: Mapping[str, Any]) -> int:
    if "correct" in evaluation:
        return int(evaluation["correct"])
    rows = int(evaluation.get("num_with_gold") or evaluation.get("num_samples") or 0)
    return int(round(rows * float(evaluation.get("primary_accuracy") or 0.0)))


def _ids(rows: list[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("id") or "") for row in rows if row.get("id")]


def _fresh_wrong_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("id")) for row in rows if not dataset_accuracy(row)]


def summarize_fresh_validation(
    *,
    run_dir: Path,
    output_root: Path | None = None,
    input_path: Path | None = None,
    projection_path: Path | None = None,
    merged_path: Path | None = None,
    eval_path: Path | None = None,
    min_correct: int = 7,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_root = (output_root or run_dir / DEFAULT_OUTPUT_ROOT_NAME).resolve()
    input_path = (input_path or run_dir / DEFAULT_INPUT_RELATIVE).resolve()
    projection_path = (projection_path or run_dir / DEFAULT_PROJECTION_NAME).resolve()
    merged_path = (merged_path or _default_merged_path(output_root)).resolve()
    eval_path = (eval_path or _default_eval_path(output_root)).resolve()

    input_rows = read_jsonl(input_path)
    merged_rows = read_jsonl(merged_path)
    evaluation = read_json(eval_path)
    projection = read_json(projection_path)

    expected_ids = _ids(input_rows)
    output_ids = _ids(merged_rows)
    projected_ids = list(projection.get("wrong_to_correct_ids") or [])
    fresh_wrong_ids = _fresh_wrong_ids(merged_rows)
    rows = int(evaluation.get("num_with_gold") or evaluation.get("num_samples") or 0)
    correct = _correct_count(evaluation)
    failed = int(evaluation.get("num_failed_exec") or 0)
    missing = int(evaluation.get("num_missing_answer") or 0)

    missing_output_ids = [row_id for row_id in expected_ids if row_id not in set(output_ids)]
    extra_output_ids = [row_id for row_id in output_ids if row_id not in set(expected_ids)]
    decision_reasons: list[str] = []
    incomplete_reasons: list[str] = []
    if missing_output_ids:
        incomplete_reasons.append("missing_output_ids")
    if extra_output_ids:
        incomplete_reasons.append("extra_output_ids")
    if rows != len(expected_ids):
        incomplete_reasons.append("eval_row_count_mismatch")
    if len(output_ids) != len(expected_ids):
        incomplete_reasons.append("merged_row_count_mismatch")
    if failed:
        decision_reasons.append("failed_exec_present")
    if missing:
        decision_reasons.append("missing_answer_present")
    if correct < min_correct:
        decision_reasons.append("correct_below_threshold")
    if fresh_wrong_ids and correct < len(projected_ids):
        decision_reasons.append("fresh_wrong_targeted_ids")

    if incomplete_reasons:
        decision = "incomplete"
        reasons = incomplete_reasons + decision_reasons
    elif decision_reasons:
        decision = "inspect"
        reasons = decision_reasons
    else:
        decision = "pass"
        reasons = ["fresh_targeted_validation_passed"]

    return {
        "run_dir": str(run_dir),
        "scope": "fresh Qwen WTQ targeted-fix affected-slice validation",
        "paths": {
            "input_jsonl": str(input_path),
            "projection_json": str(projection_path),
            "merged_jsonl": str(merged_path),
            "eval_json": str(eval_path),
        },
        "coverage": {
            "expected_rows": len(expected_ids),
            "merged_rows": len(output_ids),
            "eval_rows": rows,
            "expected_ids": expected_ids,
            "missing_output_ids": missing_output_ids,
            "extra_output_ids": extra_output_ids,
        },
        "projection": {
            "targeted_ids": projected_ids,
            "targeted_count": len(projected_ids),
            "full50_current_correct": projection.get("current_correct"),
            "full50_projected_correct": projection.get("projected_correct"),
            "wrong_to_correct": projection.get("wrong_to_correct"),
            "correct_to_wrong": projection.get("correct_to_wrong"),
        },
        "fresh": {
            "correct": correct,
            "rows": rows,
            "accuracy": correct / rows if rows else 0.0,
            "min_correct": min_correct,
            "num_failed_exec": failed,
            "num_missing_answer": missing,
            "avg_total_tokens": float(evaluation.get("avg_total_tokens") or 0.0),
            "avg_elapsed_seconds": float(evaluation.get("avg_elapsed_seconds") or 0.0),
            "fresh_wrong_ids": fresh_wrong_ids,
        },
        "decision": decision,
        "decision_reasons": reasons,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    fresh = summary["fresh"]
    coverage = summary["coverage"]
    projection = summary["projection"]
    lines = [
        "# P4b WTQ Targeted Fresh Validation",
        "",
        f"Run dir: `{summary['run_dir']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| decision | `{summary['decision']}` |",
        f"| fresh correct | {fresh['correct']}/{fresh['rows']} |",
        f"| min correct | {fresh['min_correct']} |",
        f"| failed exec | {fresh['num_failed_exec']} |",
        f"| missing answer | {fresh['num_missing_answer']} |",
        f"| avg total tokens | {fresh['avg_total_tokens']:.2f} |",
        f"| avg elapsed seconds | {fresh['avg_elapsed_seconds']:.2f} |",
        f"| expected rows | {coverage['expected_rows']} |",
        f"| merged rows | {coverage['merged_rows']} |",
        f"| eval rows | {coverage['eval_rows']} |",
        f"| projected targeted rows | {projection['targeted_count']} |",
        "",
        f"Decision reasons: `{', '.join(summary['decision_reasons'])}`",
        "",
        "Fresh wrong IDs:",
        "",
    ]
    wrong_ids = fresh["fresh_wrong_ids"]
    if wrong_ids:
        lines.extend(f"- `{row_id}`" for row_id in wrong_ids)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Input/output paths:",
            "",
        ]
    )
    for name, path in summary["paths"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Mapping[str, Any], summary_json: Path, summary_md: Path) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--projection-json", type=Path, default=None)
    parser.add_argument("--merged-jsonl", type=Path, default=None)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--min-correct", type=int, default=7)
    parser.add_argument("--fail-on-inspect", action="store_true")
    args = parser.parse_args()

    summary = summarize_fresh_validation(
        run_dir=args.run_dir,
        output_root=args.output_root,
        input_path=args.input_jsonl,
        projection_path=args.projection_json,
        merged_path=args.merged_jsonl,
        eval_path=args.eval_json,
        min_correct=args.min_correct,
    )
    summary_json = args.summary_json or args.run_dir / DEFAULT_SUMMARY_JSON
    summary_md = args.summary_md or args.run_dir / DEFAULT_SUMMARY_MD
    write_outputs(summary, summary_json, summary_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_inspect and summary["decision"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
