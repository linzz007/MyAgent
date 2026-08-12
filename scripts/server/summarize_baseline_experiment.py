#!/usr/bin/env python3
"""Summarize the P0 baseline experiment package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluate_results import load_jsonl, summarize_rows  # noqa: E402


DATASETS = ("wtq", "tabfact", "crt")
METHODS = (
    ("myagent", "MyAgent"),
    ("mact", "MACT"),
    ("direct_cot", "Direct-CoT"),
    ("single_agent_pandas", "Single-Agent Pandas"),
)


def find_jsonl(run_dir: Path, method_key: str, dataset: str) -> Path | None:
    patterns = {
        "myagent": [f"myagent_formal200/merged/{dataset}_*.jsonl"],
        "mact": [f"mact/{dataset}_mact_formal200.jsonl"],
        "direct_cot": [f"direct_cot_formal200/merged/{dataset}_*.jsonl"],
        "single_agent_pandas": [f"single_agent_pandas_formal200/merged/{dataset}_*.jsonl"],
    }[method_key]
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def dataset_summary(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "pending",
            "path": "" if path is None else str(path),
            "rows": 0,
            "accuracy": None,
            "correct": None,
            "avg_total_tokens": None,
            "avg_elapsed_seconds": None,
            "failed": None,
            "missing": None,
        }
    rows = load_jsonl(str(path))
    summary, _ = summarize_rows(rows)
    correct = round(float(summary.get("primary_accuracy", 0.0)) * int(summary.get("num_with_gold", 0)))
    return {
        "status": "done",
        "path": str(path),
        "rows": summary.get("num_samples", 0),
        "num_with_gold": summary.get("num_with_gold", 0),
        "accuracy": summary.get("primary_accuracy", 0.0),
        "correct": correct,
        "avg_total_tokens": summary.get("avg_total_tokens", 0.0),
        "avg_elapsed_seconds": summary.get("avg_elapsed_seconds", 0.0),
        "failed": summary.get("num_failed_exec", 0),
        "missing": summary.get("num_missing_answer", 0),
        "token_measurement": summary.get("token_measurement", ""),
    }


def method_summary(run_dir: Path, method_key: str, method_name: str) -> Dict[str, Any]:
    datasets = {
        dataset: dataset_summary(find_jsonl(run_dir, method_key, dataset))
        for dataset in DATASETS
    }
    done = [item for item in datasets.values() if item["status"] == "done"]
    total_rows = sum(int(item.get("rows") or 0) for item in done)
    total_with_gold = sum(int(item.get("num_with_gold") or 0) for item in done)
    total_correct = sum(int(item.get("correct") or 0) for item in done)
    total_failed = sum(int(item.get("failed") or 0) for item in done)
    total_missing = sum(int(item.get("missing") or 0) for item in done)
    weighted_tokens = sum(
        float(item.get("avg_total_tokens") or 0.0) * int(item.get("rows") or 0)
        for item in done
    )
    weighted_time = sum(
        float(item.get("avg_elapsed_seconds") or 0.0) * int(item.get("rows") or 0)
        for item in done
    )
    return {
        "method": method_name,
        "method_key": method_key,
        "datasets": datasets,
        "overall": {
            "status": "done" if len(done) == len(DATASETS) else "pending",
            "rows": total_rows,
            "num_with_gold": total_with_gold,
            "correct": total_correct,
            "accuracy": total_correct / total_with_gold if total_with_gold else None,
            "avg_total_tokens": weighted_tokens / total_rows if total_rows else None,
            "avg_elapsed_seconds": weighted_time / total_rows if total_rows else None,
            "failed": total_failed,
            "missing": total_missing,
        },
    }


def token_ratio_to_mact(method: Dict[str, Any], mact: Dict[str, Any]) -> float | None:
    method_tokens = method["overall"].get("avg_total_tokens")
    mact_tokens = mact["overall"].get("avg_total_tokens")
    if method_tokens is None or not mact_tokens:
        return None
    return float(method_tokens) / float(mact_tokens)


def fmt_accuracy(item: Dict[str, Any]) -> str:
    if item.get("status") != "done" or item.get("accuracy") is None:
        return "pending"
    return f"{item['correct']}/{item.get('num_with_gold', item.get('rows'))} = {float(item['accuracy']):.4f}"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# P0 Baseline Experiment Summary",
        "",
        f"Run dir: `{summary['run_dir']}`",
        "",
        "## Main Table",
        "",
        "| Method | WTQ Acc | TabFact Acc | CRT Acc | Overall Acc | Avg Token | Avg Time | Fail/Missing | Token Ratio to MACT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mact = next(item for item in summary["methods"] if item["method_key"] == "mact")
    for item in summary["methods"]:
        overall = item["overall"]
        token_ratio = token_ratio_to_mact(item, mact)
        lines.append(
            "| {method} | {wtq} | {tabfact} | {crt} | {overall_acc} | {tokens} | {time} | {fail_missing} | {ratio} |".format(
                method=item["method"],
                wtq=fmt_accuracy(item["datasets"]["wtq"]),
                tabfact=fmt_accuracy(item["datasets"]["tabfact"]),
                crt=fmt_accuracy(item["datasets"]["crt"]),
                overall_acc=fmt_accuracy(overall),
                tokens=fmt_num(overall.get("avg_total_tokens")),
                time=fmt_num(overall.get("avg_elapsed_seconds")),
                fail_missing=f"{overall.get('failed', 0)}/{overall.get('missing', 0)}",
                ratio=fmt_num(token_ratio, 4),
            )
        )
    lines.extend(["", "## Source Files", ""])
    for item in summary["methods"]:
        lines.append(f"### {item['method']}")
        for dataset in DATASETS:
            path = item["datasets"][dataset].get("path") or "pending"
            lines.append(f"- {dataset}: `{path}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_summary(run_dir: Path) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "methods": [
            method_summary(run_dir, method_key, method_name)
            for method_key, method_name in METHODS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary = build_summary(run_dir)
    output_json = Path(args.output_json) if args.output_json else run_dir / "summary" / "main_baseline_summary.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "summary" / "main_baseline_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
