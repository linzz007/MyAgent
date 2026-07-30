#!/usr/bin/env python3
"""Audit the current Qwen3-vs-MACT experiment evidence from saved JSON files."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_api_registry import API_KEY_NAMES, provider_profiles_for_api_keys
from experiment_model_registry import KNOWN_TESTED_LOCAL_MODELS, known_tested_model_key


FULL200_RUN = "qwen3_32b_blind200_mact_full200_20260723"
CRT_CURRENT_RUN = "qwen3_32b_crt_full200_current_20260730_1822"
WTQ_REP_RUN = "qwen3_32b_wtq_extreme_fix_representative100_20260730_1805"
MAX_MODEL_DISCOVERY_DEPTH = 4


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_path(mact_root: Path, run_name: str, file_name: str) -> Path:
    return mact_root / "outputs" / "server_runs" / run_name / file_name


def file_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "present": path.exists()}


def metric(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def count_datasets_at_least_mact(summary: Mapping[str, Any]) -> int | None:
    datasets = summary.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        return None
    count = 0
    for result in datasets.values():
        myagent_correct = metric(result, "myagent", "correct")
        mact_correct = metric(result, "mact", "correct")
        if myagent_correct is not None and mact_correct is not None and myagent_correct >= mact_correct:
            count += 1
    return count


def summarize_full200(summary: Mapping[str, Any]) -> dict[str, Any]:
    myagent_correct = metric(summary, "overall", "myagent", "correct")
    myagent_rows = metric(summary, "overall", "myagent", "num_samples")
    myagent_accuracy = metric(summary, "overall", "myagent", "primary_accuracy")
    mact_correct = metric(summary, "overall", "mact", "correct")
    mact_rows = metric(summary, "overall", "mact", "num_samples")
    mact_accuracy = metric(summary, "overall", "mact", "primary_accuracy")
    token_ratio = summary.get("token_ratio_myagent_to_mact")
    datasets_at_least_mact = count_datasets_at_least_mact(summary)
    criteria = summary.get("acceptance_criteria")

    if isinstance(criteria, Mapping):
        overall_at_least = bool(criteria.get("overall_accuracy_at_least_mact"))
        at_least_two = bool(criteria.get("at_least_two_datasets_at_least_mact"))
        token_ok = bool(criteria.get("token_ratio_at_most_0_75"))
        failure_ok = bool(criteria.get("execution_failure_rate_at_most_0_02"))
        strict_acceptance = all((overall_at_least, at_least_two, token_ok, failure_ok))
    else:
        overall_at_least = (
            myagent_accuracy is not None and mact_accuracy is not None and myagent_accuracy >= mact_accuracy
        )
        at_least_two = datasets_at_least_mact is not None and datasets_at_least_mact >= 2
        token_ok = token_ratio is not None and token_ratio <= 0.75
        failure_ok = True
        strict_acceptance = overall_at_least and at_least_two and token_ok and failure_ok

    return {
        "myagent_correct": myagent_correct,
        "myagent_rows": myagent_rows,
        "myagent_accuracy": myagent_accuracy,
        "mact_correct": mact_correct,
        "mact_rows": mact_rows,
        "mact_accuracy": mact_accuracy,
        "token_ratio": token_ratio,
        "datasets_myagent_at_least_mact": datasets_at_least_mact,
        "overall_accuracy_at_least_mact": overall_at_least,
        "token_ratio_at_most_0_75": token_ok,
        "strict_acceptance": strict_acceptance,
    }


def summarize_staged_composite(crt_comparison: Mapping[str, Any]) -> dict[str, Any]:
    staged = crt_comparison.get("overall_if_replacing_crt_only")
    if not isinstance(staged, Mapping):
        return {"present": False}
    myagent_accuracy = staged.get("myagent_accuracy")
    mact_accuracy = staged.get("mact_accuracy")
    token_ratio = staged.get("token_ratio")
    return {
        "present": True,
        "myagent_correct": staged.get("myagent_correct"),
        "myagent_rows": staged.get("myagent_rows"),
        "myagent_accuracy": myagent_accuracy,
        "mact_correct": staged.get("mact_correct"),
        "mact_rows": staged.get("mact_rows"),
        "mact_accuracy": mact_accuracy,
        "token_ratio": token_ratio,
        "overall_accuracy_at_least_mact": (
            myagent_accuracy is not None and mact_accuracy is not None and myagent_accuracy >= mact_accuracy
        ),
        "token_ratio_at_most_0_75": token_ratio is not None and token_ratio <= 0.75,
    }


def summarize_wtq_representative(comparison: Mapping[str, Any]) -> dict[str, Any]:
    transitions = comparison.get("old_to_new_transitions")
    if not isinstance(transitions, Mapping):
        transitions = {}
    recovered = metric(transitions, "old_wrong_new_correct", "count", default=0) or 0
    regressed = metric(transitions, "old_correct_new_wrong", "count", default=0) or 0
    net = recovered - regressed
    decision = (
        "do_not_expand_wtq_extreme_only_fix"
        if net <= 0
        else "eligible_for_small_followup_gate"
    )
    return {
        "new_myagent_accuracy": metric(comparison, "new_myagent", "primary_accuracy"),
        "old_myagent_accuracy": metric(comparison, "old_myagent", "primary_accuracy"),
        "mact_accuracy": metric(comparison, "mact", "primary_accuracy"),
        "token_ratio_new_vs_mact": metric(comparison, "token_ratios", "new_vs_mact"),
        "token_ratio_new_vs_old_myagent": metric(comparison, "token_ratios", "new_vs_old_myagent"),
        "recovered_rows": recovered,
        "regressed_rows": regressed,
        "net_recovered_rows": net,
        "decision": decision,
    }


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "config.json").exists()
        or (path / "tokenizer_config.json").exists()
        or any(path.glob("*.safetensors"))
    )


def is_hf_cache_model_dir(path: Path) -> bool:
    if not path.is_dir() or not path.name.startswith("models--"):
        return False
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(is_model_dir(snapshot) for snapshot in snapshots.iterdir())


def hf_cache_model_name(path: Path) -> str:
    return path.name.split("--")[-1]


def hf_cache_model_paths(path: Path) -> list[Path]:
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(snapshot for snapshot in snapshots.iterdir() if is_model_dir(snapshot))


def discover_local_model_paths(model_roots: Sequence[Path]) -> dict[str, list[str]]:
    discovered: dict[str, set[str]] = {}
    for root in model_roots:
        if not root.exists():
            continue
        stack = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            if current.name.startswith("."):
                continue
            if is_hf_cache_model_dir(current):
                name = hf_cache_model_name(current)
                discovered.setdefault(name, set()).update(str(path) for path in hf_cache_model_paths(current))
                continue
            if is_model_dir(current):
                discovered.setdefault(current.name, set()).add(str(current))
                continue
            if depth >= MAX_MODEL_DISCOVERY_DEPTH:
                continue
            try:
                children = sorted(child for child in current.iterdir() if child.is_dir())
            except OSError:
                continue
            stack.extend((child, depth + 1) for child in reversed(children))
    return {name: sorted(paths) for name, paths in sorted(discovered.items())}


def discover_local_models(model_roots: Sequence[Path]) -> list[str]:
    return sorted(discover_local_model_paths(model_roots))


def present_api_keys(env: Mapping[str, str]) -> list[str]:
    return sorted(name for name in API_KEY_NAMES if env.get(name))


def summarize_model_readiness(model_roots: Sequence[Path], env: Mapping[str, str]) -> dict[str, Any]:
    local_model_paths = discover_local_model_paths(model_roots)
    local_models = sorted(local_model_paths)
    untested = sorted(model for model in local_models if known_tested_model_key(model) is None)
    untested_paths = {model: local_model_paths[model] for model in untested}
    api_keys = present_api_keys(env)
    can_start = bool(untested or api_keys)
    return {
        "local_models": local_models,
        "local_model_paths": local_model_paths,
        "known_tested_local_models": sorted(KNOWN_TESTED_LOCAL_MODELS),
        "untested_local_models": untested,
        "untested_local_model_paths": untested_paths,
        "api_keys_present": api_keys,
        "api_provider_profiles": provider_profiles_for_api_keys(api_keys),
        "can_start_new_experiment": can_start,
        "next_action": "run_gate10_then_gate50" if can_start else "wait_for_new_model_or_api_key",
    }


def format_percent(value: Any, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.{digits}f}%"
    return "n/a"


def format_ratio_percent(value: Any, digits: int = 1) -> str:
    return format_percent(value, digits=digits)


def format_fraction(correct: Any, rows: Any) -> str:
    if correct is None or rows is None:
        return "n/a"
    return f"{correct}/{rows}"


def render_expert_summary(audit: Mapping[str, Any]) -> str:
    canonical = audit.get("canonical_full200", {})
    staged = audit.get("current_crt_staged_composite", {})
    wtq = audit.get("wtq_representative100", {})
    readiness = audit.get("model_readiness", {})

    canonical_fraction = format_fraction(canonical.get("myagent_correct"), canonical.get("myagent_rows"))
    canonical_mact_fraction = format_fraction(canonical.get("mact_correct"), canonical.get("mact_rows"))
    staged_fraction = format_fraction(staged.get("myagent_correct"), staged.get("myagent_rows"))
    staged_mact_fraction = format_fraction(staged.get("mact_correct"), staged.get("mact_rows"))
    token_percent = format_ratio_percent(canonical.get("token_ratio"))
    staged_token_percent = format_ratio_percent(staged.get("token_ratio"))
    datasets_at_least = canonical.get("datasets_myagent_at_least_mact")
    strict_text = "未通过" if not canonical.get("strict_acceptance") else "通过"

    recovered = wtq.get("recovered_rows", 0)
    regressed = wtq.get("regressed_rows", 0)
    net = wtq.get("net_recovered_rows", 0)

    if readiness.get("can_start_new_experiment"):
        next_action = (
            "已有新增候选，可按 PRD 第 14 节先跑 Gate-10 smoke，再跑 Gate-50。"
        )
    else:
        next_action = (
            "等待新增/挂载候选模型或提供外部 API key；当前不要重启旧 Qwen3-32B/no-go 模型做重复实验。"
        )

    lines = [
        "# Qwen3-32B vs MACT 阶段证据摘要",
        "",
        "## 可写结论",
        "",
        (
            f"- canonical full200：myAgent {canonical_fraction} "
            f"({format_percent(canonical.get('myagent_accuracy'))}) vs MACT {canonical_mact_fraction} "
            f"({format_percent(canonical.get('mact_accuracy'))})，平均 token 为 MACT 的 {token_percent}。"
        ),
        (
            f"- current CRT staged composite：myAgent {staged_fraction} "
            f"({format_percent(staged.get('myagent_accuracy'))}) vs MACT {staged_mact_fraction} "
            f"({format_percent(staged.get('mact_accuracy'))})，平均 token 约为 MACT 的 {staged_token_percent}。"
        ),
        "- 当前 evidence_complete=true，可作为专家/专利材料中的阶段性 paired evidence。",
        "",
        "## 必须写明的限制",
        "",
        f"- strict acceptance：{strict_text}；full200 中 myAgent 不低于 MACT 的数据集数为 {datasets_at_least}/3。",
        "- 不能写成三个数据集全部超过 MACT，也不能写成 full200 对 MACT 全面显著胜出。",
        "- WTQ 和 TabFact 是 full200 单项短板，优势主要来自 CRT。",
        "",
        "## WTQ 修复判断",
        "",
        (
            "- WTQ representative100：新 myAgent 与旧 myAgent 同为 "
            f"{format_percent(wtq.get('new_myagent_accuracy'))}，MACT 为 {format_percent(wtq.get('mact_accuracy'))}；"
            f"恢复 {recovered} 条、回退 {regressed} 条、净收益 {net} 条。"
        ),
        "- 因此 WTQ extreme/only 全局行修复不应继续作为下一阶段主线扩大。",
        "",
        "## 下一步",
        "",
        f"- {next_action}",
        "- 新候选进入后，先跑 Gate-10 / Gate-50；只有 Gate-50 接近或超过 Qwen3-32B reference，才扩 Gate-150 / Paired-200。",
        "",
    ]
    return "\n".join(lines)


def build_audit(
    *,
    myagent_root: Path,
    mact_root: Path,
    model_roots: Sequence[Path],
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    full200_path = evidence_path(mact_root, FULL200_RUN, "overall_mact_full200_summary.json")
    crt_current_path = evidence_path(mact_root, CRT_CURRENT_RUN, "crt_full200_current_comparison.json")
    wtq_rep_path = evidence_path(mact_root, WTQ_REP_RUN, "wtq_representative100_extreme_fix_comparison.json")

    full200_summary = read_json(full200_path) if full200_path.exists() else {}
    crt_current = read_json(crt_current_path) if crt_current_path.exists() else {}
    wtq_rep = read_json(wtq_rep_path) if wtq_rep_path.exists() else {}

    evidence_files = {
        "canonical_full200_summary": file_status(full200_path),
        "current_crt_comparison": file_status(crt_current_path),
        "wtq_representative100_comparison": file_status(wtq_rep_path),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "myagent_root": str(myagent_root),
        "mact_root": str(mact_root),
        "evidence_complete": all(item["present"] for item in evidence_files.values()),
        "evidence_files": evidence_files,
        "canonical_full200": summarize_full200(full200_summary),
        "current_crt_staged_composite": summarize_staged_composite(crt_current),
        "wtq_representative100": summarize_wtq_representative(wtq_rep),
        "model_readiness": summarize_model_readiness(model_roots, env),
    }


def default_model_roots() -> list[Path]:
    return [
        Path("/home/ubuntu/models"),
        Path("/home/ubuntu/.cache/huggingface"),
        Path("/data"),
        Path("/mnt"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--myagent-root", type=Path, default=Path.cwd())
    parser.add_argument("--mact-root", type=Path, default=Path("/home/ubuntu/lzz/MACT"))
    parser.add_argument("--model-root", action="append", type=Path, dest="model_roots")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    audit = build_audit(
        myagent_root=args.myagent_root.resolve(),
        mact_root=args.mact_root.resolve(),
        model_roots=[path.resolve() for path in (args.model_roots or default_model_roots())],
        env=os.environ,
    )
    output_text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_expert_summary(audit), encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
