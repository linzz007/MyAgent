#!/usr/bin/env python3
"""Write a compact progress snapshot for the final thesis experiment run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("wtq", "tabfact", "crt")


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    input_slice: str
    output_patterns: tuple[str, ...]
    raw_patterns: tuple[str, ...]
    eval_patterns: tuple[str, ...]
    log_patterns: tuple[str, ...]


TARGETS: tuple[Target, ...] = (
    Target(
        "myagent_formal200",
        "MyAgent Formal-200",
        "formal200",
        ("myagent_formal200/merged/{dataset}_*.jsonl",),
        ("myagent_formal200/raw/{dataset}/*.jsonl",),
        ("myagent_formal200/eval/{dataset}_*_eval.json",),
        ("myagent_formal200/logs/{dataset}/*.log",),
    ),
    Target(
        "mact_formal200",
        "MACT Formal-200",
        "formal200",
        ("mact/{dataset}_mact_formal200.jsonl",),
        (),
        ("eval/{dataset}_mact_formal200_eval.json",),
        ("logs/mact_{dataset}_formal200.log", "logs/mact_{dataset}_formal200*.log"),
    ),
    Target(
        "direct_cot_formal200",
        "Direct-CoT Formal-200",
        "formal200",
        ("direct_cot_formal200/merged/{dataset}_*.jsonl",),
        ("direct_cot_formal200/raw/{dataset}/*.jsonl",),
        ("direct_cot_formal200/eval/{dataset}_*_eval.json",),
        ("direct_cot_formal200/logs/direct_cot/{dataset}/*.log", "direct_cot_formal200/logs/{dataset}/*.log"),
    ),
    Target(
        "single_agent_pandas_formal200",
        "Single-Agent Pandas Formal-200",
        "formal200",
        ("single_agent_pandas_formal200/merged/{dataset}_*.jsonl",),
        ("single_agent_pandas_formal200/raw/{dataset}/*.jsonl",),
        ("single_agent_pandas_formal200/eval/{dataset}_*_eval.json",),
        (
            "single_agent_pandas_formal200/logs/single_agent_pandas/{dataset}/*.log",
            "single_agent_pandas_formal200/logs/{dataset}/*.log",
        ),
    ),
    Target(
        "ablation_legacy_gate50",
        "Ablation Legacy Gate-50",
        "ablation50",
        ("ablation/legacy_gate50/merged/{dataset}_*.jsonl",),
        ("ablation/legacy_gate50/raw/{dataset}/*.jsonl",),
        ("ablation/legacy_gate50/eval/{dataset}_*_eval.json",),
        ("ablation/legacy_gate50/logs/{dataset}/*.log",),
    ),
    Target(
        "ablation_no_strong_gate50",
        "Ablation No Strong Verification Gate-50",
        "ablation50",
        ("ablation/no_strong_gate50/merged/{dataset}_*.jsonl",),
        ("ablation/no_strong_gate50/raw/{dataset}/*.jsonl",),
        ("ablation/no_strong_gate50/eval/{dataset}_*_eval.json",),
        ("ablation/no_strong_gate50/logs/{dataset}/*.log",),
    ),
    Target(
        "ablation_no_deterministic_shortcuts_gate50",
        "Ablation No Deterministic Validation Gate-50",
        "ablation50",
        ("ablation/no_deterministic_shortcuts_gate50/merged/{dataset}_*.jsonl",),
        ("ablation/no_deterministic_shortcuts_gate50/raw/{dataset}/*.jsonl",),
        ("ablation/no_deterministic_shortcuts_gate50/eval/{dataset}_*_eval.json",),
        ("ablation/no_deterministic_shortcuts_gate50/logs/{dataset}/*.log",),
    ),
    Target(
        "ablation_no_question_routing_gate50",
        "Ablation No Question Routing Gate-50",
        "ablation50",
        ("ablation/no_question_routing_gate50/merged/{dataset}_*.jsonl",),
        ("ablation/no_question_routing_gate50/raw/{dataset}/*.jsonl",),
        ("ablation/no_question_routing_gate50/eval/{dataset}_*_eval.json",),
        ("ablation/no_question_routing_gate50/logs/{dataset}/*.log",),
    ),
    Target(
        "ablation_no_risk_scoring_gate50",
        "Ablation No Risk Scoring Gate-50",
        "ablation50",
        ("ablation/no_risk_scoring_gate50/merged/{dataset}_*.jsonl",),
        ("ablation/no_risk_scoring_gate50/raw/{dataset}/*.jsonl",),
        ("ablation/no_risk_scoring_gate50/eval/{dataset}_*_eval.json",),
        ("ablation/no_risk_scoring_gate50/logs/{dataset}/*.log",),
    ),
    Target(
        "ablation_no_table_compression_gate50",
        "Ablation No Table Compression Gate-50",
        "ablation50",
        ("ablation/no_table_compression_gate50/merged/{dataset}_*.jsonl",),
        ("ablation/no_table_compression_gate50/raw/{dataset}/*.jsonl",),
        ("ablation/no_table_compression_gate50/eval/{dataset}_*_eval.json",),
        ("ablation/no_table_compression_gate50/logs/{dataset}/*.log",),
    ),
)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def git_head(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def newest(paths: Iterable[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def glob_many(run_dir: Path, patterns: Iterable[str], dataset: str) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(run_dir.glob(pattern.format(dataset=dataset)))
    return sorted(set(matches))


def tail_lines(path: Path | None, max_lines: int) -> list[str]:
    if path is None or max_lines <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return []
    return lines[-max_lines:]


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def controller_jobs(run_dir: Path, log_tail_lines: int) -> list[dict[str, Any]]:
    controller_dir = run_dir / "logs" / "controller"
    if not controller_dir.exists():
        return []

    jobs: list[dict[str, Any]] = []
    for pid_file in sorted(controller_dir.glob("*.pid")):
        raw_pid = pid_file.read_text(encoding="utf-8", errors="replace").strip()
        try:
            pid = int(raw_pid)
        except ValueError:
            pid = 0
        step_name = pid_file.stem
        log_path = controller_dir / f"{step_name}.nohup.log"
        log_exists = log_path.exists()
        jobs.append(
            {
                "step": step_name,
                "pid": pid,
                "running": is_process_running(pid),
                "pid_file": str(pid_file),
                "log_path": str(log_path) if log_exists else "",
                "log_bytes": log_path.stat().st_size if log_exists else 0,
                "log_mtime": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds")
                if log_exists
                else "",
                "log_tail": tail_lines(log_path if log_exists else None, log_tail_lines),
            }
        )
    return jobs


def input_rows(run_dir: Path, input_slice: str, dataset: str) -> tuple[int, str]:
    path = run_dir / "input" / input_slice / f"{dataset}.jsonl"
    return count_jsonl(path), str(path) if path.exists() else ""


def eval_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "parseable": False}
    data = load_json(path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "parseable": data is not None,
    }
    if data is None:
        return result
    rows = int(data.get("num_with_gold") or data.get("num_samples") or 0)
    acc = data.get("primary_accuracy")
    correct = data.get("correct")
    if correct is None and acc is not None and rows:
        correct = round(float(acc) * rows)
    result.update(
        {
            "rows": rows,
            "correct": correct,
            "accuracy": acc,
            "avg_total_tokens": data.get("avg_total_tokens"),
            "avg_elapsed_seconds": data.get("avg_elapsed_seconds"),
            "failed": data.get("num_failed_exec"),
            "missing": data.get("num_missing_answer"),
        }
    )
    return result


def status_for(expected: int, merged_rows: int, raw_rows: int, eval_info: dict[str, Any]) -> str:
    has_eval = bool(eval_info.get("exists") and eval_info.get("parseable"))
    invalid_eval = bool(eval_info.get("exists") and not eval_info.get("parseable"))
    if expected <= 0:
        return "missing_input"
    if invalid_eval:
        return "invalid_eval"
    if merged_rows == expected and has_eval:
        return "done"
    if merged_rows == expected and not has_eval:
        return "needs_eval"
    if merged_rows > expected:
        return "row_mismatch"
    if merged_rows or raw_rows:
        return "running_or_partial"
    return "pending"


def build_snapshot(run_dir: Path, myagent_root: Path, mact_root: Path, log_tail_lines: int) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    snapshot: dict[str, Any] = {
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "myagent_root": str(myagent_root),
        "mact_root": str(mact_root),
        "git": {
            "myagent_head": git_head(myagent_root),
            "mact_head": git_head(mact_root),
        },
        "targets": [],
        "controller_jobs": controller_jobs(run_dir, log_tail_lines),
        "totals": {
            "expected_rows": 0,
            "merged_rows": 0,
            "raw_rows": 0,
            "done_items": 0,
            "needs_attention_items": 0,
        },
    }

    for target in TARGETS:
        target_record = {"key": target.key, "name": target.name, "datasets": {}}
        for dataset in DATASETS:
            expected, input_path = input_rows(run_dir, target.input_slice, dataset)
            output_paths = glob_many(run_dir, target.output_patterns, dataset)
            raw_paths = glob_many(run_dir, target.raw_patterns, dataset)
            eval_paths = glob_many(run_dir, target.eval_patterns, dataset)
            log_paths = glob_many(run_dir, target.log_patterns, dataset)

            output_path = output_paths[0] if output_paths else None
            eval_path = eval_paths[0] if eval_paths else None
            log_path = newest(log_paths)
            merged_rows = count_jsonl(output_path) if output_path else 0
            raw_rows = sum(count_jsonl(path) for path in raw_paths)
            eval_info = eval_summary(eval_path)
            status = status_for(expected, merged_rows, raw_rows, eval_info)

            needs_attention = status in {"missing_input", "needs_eval", "row_mismatch", "invalid_eval"}
            record = {
                "dataset": dataset,
                "status": status,
                "expected_rows": expected,
                "merged_rows": merged_rows,
                "raw_rows": raw_rows,
                "input_path": input_path,
                "output_path": str(output_path) if output_path else "",
                "eval": eval_info,
                "latest_log_path": str(log_path) if log_path else "",
                "latest_log_tail": tail_lines(log_path, log_tail_lines) if needs_attention else [],
            }
            target_record["datasets"][dataset] = record

            snapshot["totals"]["expected_rows"] += expected
            snapshot["totals"]["merged_rows"] += merged_rows
            snapshot["totals"]["raw_rows"] += raw_rows
            if status == "done":
                snapshot["totals"]["done_items"] += 1
            if needs_attention:
                snapshot["totals"]["needs_attention_items"] += 1

        snapshot["targets"].append(target_record)
    return snapshot


def fmt_eval(eval_info: dict[str, Any]) -> str:
    if not eval_info.get("exists"):
        return "missing"
    if not eval_info.get("parseable"):
        return "invalid"
    acc = eval_info.get("accuracy")
    correct = eval_info.get("correct")
    rows = eval_info.get("rows")
    if acc is None:
        return "present"
    return f"{correct}/{rows} = {float(acc):.4f}"


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# Experiment Progress Snapshot",
        "",
        f"Generated at: `{snapshot['generated_at']}`",
        f"Run dir: `{snapshot['run_dir']}`",
        f"MyAgent HEAD: `{snapshot['git'].get('myagent_head') or 'unknown'}`",
        f"MACT HEAD: `{snapshot['git'].get('mact_head') or 'unknown'}`",
        "",
        "## Totals",
        "",
        "| Expected rows | Merged rows | Raw rows | Done items | Needs attention |",
        "|---:|---:|---:|---:|---:|",
        "| {expected_rows} | {merged_rows} | {raw_rows} | {done_items} | {needs_attention_items} |".format(
            **snapshot["totals"]
        ),
        "",
        "## Controller Jobs",
        "",
        "| Step | PID | Running | Log bytes | Log mtime | Log |",
        "|---|---:|---|---:|---|---|",
    ]
    if snapshot.get("controller_jobs"):
        for job in snapshot["controller_jobs"]:
            lines.append(
                "| {step} | {pid} | {running} | {log_bytes} | {log_mtime} | `{log_path}` |".format(
                    step=job["step"],
                    pid=job["pid"] or "",
                    running="yes" if job["running"] else "no",
                    log_bytes=job["log_bytes"],
                    log_mtime=job["log_mtime"] or "",
                    log_path=job["log_path"] or "",
                )
            )
    else:
        lines.append("| none |  |  | 0 |  |  |")

    lines.extend(
        [
        "",
        "## Matrix",
        "",
        "| Target | Dataset | Status | Expected | Merged | Raw | Eval | Latest log |",
        "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for target in snapshot["targets"]:
        for dataset in DATASETS:
            item = target["datasets"][dataset]
            log_path = item.get("latest_log_path") or ""
            lines.append(
                "| {target} | {dataset} | {status} | {expected} | {merged} | {raw} | {eval_status} | `{log}` |".format(
                    target=target["name"],
                    dataset=dataset,
                    status=item["status"],
                    expected=item["expected_rows"],
                    merged=item["merged_rows"],
                    raw=item["raw_rows"],
                    eval_status=fmt_eval(item["eval"]),
                    log=log_path,
                )
            )

    attention = [
        (target["name"], dataset, item)
        for target in snapshot["targets"]
        for dataset, item in target["datasets"].items()
        if item["status"] in {"missing_input", "needs_eval", "row_mismatch", "invalid_eval"}
    ]
    if attention:
        lines.extend(["", "## Needs Attention", ""])
        for target_name, dataset, item in attention:
            lines.append(
                f"- {target_name} / {dataset}: `{item['status']}`, "
                f"expected `{item['expected_rows']}`, merged `{item['merged_rows']}`, raw `{item['raw_rows']}`."
            )
            if item.get("latest_log_tail"):
                lines.append("")
                lines.append("```text")
                lines.extend(item["latest_log_tail"])
                lines.append("```")
                lines.append("")

    stopped_incomplete = [
        (target["name"], dataset, item)
        for target in snapshot["targets"]
        for dataset, item in target["datasets"].items()
        if item["status"] in {"pending", "running_or_partial"}
    ]
    running_any = any(bool(job.get("running")) for job in snapshot.get("controller_jobs", []))
    if stopped_incomplete and snapshot.get("controller_jobs") and not running_any:
        lines.extend(["", "## Stopped While Incomplete", ""])
        lines.append(
            "No controller PID is currently running, but at least one target is still pending or partial. "
            "Inspect the controller log tail before starting the next step."
        )
        for target_name, dataset, item in stopped_incomplete[:12]:
            lines.append(
                f"- {target_name} / {dataset}: `{item['status']}`, "
                f"expected `{item['expected_rows']}`, merged `{item['merged_rows']}`, raw `{item['raw_rows']}`."
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--myagent-root", default="/home/ubuntu/lzz/MyAgent")
    parser.add_argument("--mact-root", default="/home/ubuntu/lzz/MACT")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--log-tail-lines", type=int, default=40)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    snapshot = build_snapshot(
        run_dir=run_dir,
        myagent_root=Path(args.myagent_root).resolve(),
        mact_root=Path(args.mact_root).resolve(),
        log_tail_lines=max(args.log_tail_lines, 0),
    )

    output_json = Path(args.output_json) if args.output_json else run_dir / "summary" / "progress_snapshot.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "summary" / "progress_snapshot.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(snapshot), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
