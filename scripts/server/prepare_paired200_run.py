#!/usr/bin/env python3
"""Prepare a MACT-hosted paired-200 expansion directory for final candidates."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WTQ_DATASET = "datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl"
DEFAULT_TABFACT_DATASET = "datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl"
DEFAULT_CRT_DATASET = "datasets_ready/blind_holdout_200_v1_2026-06-27/crt.jsonl"
DEFAULT_MACT_AVG_TOKENS = 11262.41
PAIRED_LIMIT = 200


@dataclass(frozen=True)
class Paired200Config:
    myagent_root: Path
    mact_root: Path
    gate_run_dir: Path
    run_dir: Path | None = None
    wtq_dataset: str = DEFAULT_WTQ_DATASET
    tabfact_dataset: str = DEFAULT_TABFACT_DATASET
    crt_dataset: str = DEFAULT_CRT_DATASET
    max_replan: int = 2
    mact_avg_tokens: float = DEFAULT_MACT_AVG_TOKENS


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def safe_model_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "model"


def default_run_dir(mact_root: Path, model_tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return mact_root / "outputs" / "server_runs" / f"{safe_slug(model_tag)}_paired200_{stamp}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def source_env_name(backend: str) -> str:
    return "api.env" if backend == "api" else "vllm.env"


def endpoints_for_manifest(manifest: Mapping[str, Any]) -> list[str]:
    endpoints = manifest.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("gate_run_manifest.json must contain a non-empty endpoints list")
    return [str(endpoint).rstrip("/") for endpoint in endpoints]


def render_source_line(backend: str) -> str:
    return f'source "$SOURCE_GATE_RUN_DIR/{source_env_name(backend)}"'


def render_myagent_script(config: Paired200Config, run_dir: Path, manifest: Mapping[str, Any]) -> str:
    backend = str(manifest.get("backend") or "local-vllm")
    endpoints = endpoints_for_manifest(manifest)
    if backend == "api":
        endpoints_arg = '"$API_BASE_URL"'
        api_key_arg = '"$API_KEY_ENV"'
    else:
        endpoints_arg = shell_quote(",".join(endpoints))
        api_key_arg = "LOCAL_VLLM_API_KEY"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"MYAGENT_ROOT={shell_quote(config.myagent_root)}",
            f"PAIRED_RUN_DIR={shell_quote(run_dir)}",
            f"SOURCE_GATE_RUN_DIR={shell_quote(config.gate_run_dir)}",
            'cd "$MYAGENT_ROOT"',
            "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh",
            "conda activate lzz-agent",
            render_source_line(backend),
            "python scripts/server/run_sharded_tqa.py \\",
            "  --repo-root . \\",
            "  --tasks wtq,tabfact,crt \\",
            f"  --wtq-dataset {shell_quote(config.wtq_dataset)} \\",
            f"  --tabfact-dataset {shell_quote(config.tabfact_dataset)} \\",
            f"  --crt-dataset {shell_quote(config.crt_dataset)} \\",
            f"  --endpoints {endpoints_arg} \\",
            '  --model "$SERVED_MODEL_NAME" \\',
            f"  --api-key-env {api_key_arg} \\",
            '  --output-root "$PAIRED_RUN_DIR/myagent_paired200" \\',
            f"  --limit-per-task {PAIRED_LIMIT} \\",
            f"  --max-replan {config.max_replan} \\",
            f"  --mact-avg-tokens {config.mact_avg_tokens} \\",
            "  --resume",
            "",
        ]
    )


def mact_task_for(dataset: str) -> str:
    return "scitab" if dataset == "tabfact" else dataset


def render_mact_script(
    config: Paired200Config,
    run_dir: Path,
    manifest: Mapping[str, Any],
    dataset: str,
    dataset_path: str,
) -> str:
    backend = str(manifest.get("backend") or "local-vllm")
    endpoints = endpoints_for_manifest(manifest)
    if backend == "api":
        api_base_arg = '"$API_BASE_URL"'
        api_key_arg = '"$API_KEY_ENV"'
    else:
        api_base_arg = shell_quote(endpoints[0])
        api_key_arg = "LOCAL_VLLM_API_KEY"
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"MYAGENT_ROOT={shell_quote(config.myagent_root)}",
            f"PAIRED_RUN_DIR={shell_quote(run_dir)}",
            f"SOURCE_GATE_RUN_DIR={shell_quote(config.gate_run_dir)}",
            'cd "$MYAGENT_ROOT"',
            "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh",
            "conda activate lzz-agent",
            render_source_line(backend),
            'mkdir -p "$PAIRED_RUN_DIR/mact" "$PAIRED_RUN_DIR/logs" "$PAIRED_RUN_DIR/tmp"',
            "python scripts/server/run_mact_one_by_one.py \\",
            f"  --mact-root {shell_quote(config.mact_root)} \\",
            f"  --dataset-path {shell_quote(dataset_path)} \\",
            f'  --output-path "$PAIRED_RUN_DIR/mact/{dataset}_mact_paired200.jsonl" \\',
            f'  --log-path "$PAIRED_RUN_DIR/logs/mact_{dataset}_paired200.log" \\',
            f"  --task {mact_task_for(dataset)} \\",
            '  --plan-model-name "$SERVED_MODEL_NAME" \\',
            '  --code-model-name "$SERVED_MODEL_NAME" \\',
            "  --model-provider openai_compatible \\",
            f"  --api-base {api_base_arg} \\",
            f"  --api-key-env {api_key_arg} \\",
            "  --thinking disabled \\",
            "  --temperature 0 \\",
            "  --max-tokens 2048 \\",
            "  --api-timeout 180 \\",
            "  --api-max-retries 5 \\",
            "  --plan-sample 1 \\",
            "  --code-sample 1 \\",
            "  --max-step 3 \\",
            "  --max-actual-step 3 \\",
            '  --temp-dir "$PAIRED_RUN_DIR/tmp" \\',
            f"  --limit {PAIRED_LIMIT} \\",
            "  --resume",
            "",
        ]
    )


def render_eval_compare_script(
    config: Paired200Config,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> str:
    model_name = safe_model_name(str(manifest.get("served_model_name") or manifest.get("model_tag") or "model"))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"MYAGENT_ROOT={shell_quote(config.myagent_root)}",
        f"PAIRED_RUN_DIR={shell_quote(run_dir)}",
        'cd "$MYAGENT_ROOT"',
        "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh",
        "conda activate lzz-agent",
        'mkdir -p "$PAIRED_RUN_DIR/eval"',
    ]
    for dataset in ("wtq", "tabfact", "crt"):
        lines.extend(
            [
                "python code/evaluate_results.py \\",
                f'  "$PAIRED_RUN_DIR/mact/{dataset}_mact_paired200.jsonl" \\',
                f'  --error_output "$PAIRED_RUN_DIR/eval/{dataset}_mact_paired200_errors.jsonl" \\',
                f'  > "$PAIRED_RUN_DIR/eval/{dataset}_mact_paired200_eval.json"',
            ]
        )
    lines.extend(
        [
            "python code/compare_blind_results.py \\",
            f'  --myagent_wtq "$PAIRED_RUN_DIR/myagent_paired200/merged/wtq_{model_name}.jsonl" \\',
            f'  --myagent_tabfact "$PAIRED_RUN_DIR/myagent_paired200/merged/tabfact_{model_name}.jsonl" \\',
            f'  --myagent_crt "$PAIRED_RUN_DIR/myagent_paired200/merged/crt_{model_name}.jsonl" \\',
            '  --mact_wtq "$PAIRED_RUN_DIR/mact/wtq_mact_paired200.jsonl" \\',
            '  --mact_tabfact "$PAIRED_RUN_DIR/mact/tabfact_mact_paired200.jsonl" \\',
            '  --mact_crt "$PAIRED_RUN_DIR/mact/crt_mact_paired200.jsonl" \\',
            '  --output "$PAIRED_RUN_DIR/paired200_summary.json"',
            "",
        ]
    )
    return "\n".join(lines)


def render_readme(config: Paired200Config, run_dir: Path, manifest: Mapping[str, Any]) -> str:
    api_key_env = manifest.get("api_key_env")
    api_key_lines = []
    if api_key_env:
        api_key_lines = [
            f"API key environment variable: `{api_key_env}`",
            "",
        ]
    return "\n".join(
        [
            f"# {manifest.get('model_tag', 'candidate')} Paired-200 Run",
            "",
            "This directory is for final-candidate same-ID paired-200 expansion after Gate-150.",
            "",
            *api_key_lines,
            "Run order:",
            "",
            "```bash",
            f"bash {run_dir}/run_myagent_paired200.sh",
            f"bash {run_dir}/run_mact_wtq_paired200.sh",
            f"bash {run_dir}/run_mact_tabfact_paired200.sh",
            f"bash {run_dir}/run_mact_crt_paired200.sh",
            f"bash {run_dir}/run_eval_and_compare.sh",
            "```",
            "",
            "Do not start this run unless Gate-150 still shows a competitive candidate. Keep all outputs under this MACT run directory and force-add it after each checkpoint.",
            "",
            "```bash",
            f"cd {config.mact_root}",
            f"git add -f {run_dir}",
            "```",
            "",
        ]
    )


def build_manifest(config: Paired200Config, run_dir: Path, gate_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "source_gate_run_dir": str(config.gate_run_dir),
        "myagent_root": str(config.myagent_root),
        "mact_root": str(config.mact_root),
        "backend": gate_manifest.get("backend") or "local-vllm",
        "model_tag": gate_manifest.get("model_tag"),
        "served_model_name": gate_manifest.get("served_model_name"),
        "endpoints": endpoints_for_manifest(gate_manifest),
        "api_key_env": gate_manifest.get("api_key_env"),
        "paired_limit": PAIRED_LIMIT,
        "datasets": {
            "wtq": config.wtq_dataset,
            "tabfact": config.tabfact_dataset,
            "crt": config.crt_dataset,
        },
        "writes_outputs_to_mact": True,
        "starts_services": False,
    }


def prepare_paired200_run(config: Paired200Config) -> dict[str, Any]:
    gate_manifest_path = config.gate_run_dir / "gate_run_manifest.json"
    if not gate_manifest_path.exists():
        raise FileNotFoundError(f"missing gate manifest: {gate_manifest_path}")
    gate_manifest = read_json(gate_manifest_path)
    backend = str(gate_manifest.get("backend") or "local-vllm")
    env_path = config.gate_run_dir / source_env_name(backend)
    if not env_path.exists():
        raise FileNotFoundError(f"missing source env for {backend}: {env_path}")

    model_tag = str(gate_manifest.get("model_tag") or "candidate")
    run_dir = config.run_dir or default_run_dir(config.mact_root, model_tag)
    run_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ("logs", "mact", "eval", "tmp"):
        (run_dir / subdir).mkdir()

    write_executable(run_dir / "run_myagent_paired200.sh", render_myagent_script(config, run_dir, gate_manifest))
    dataset_paths = {
        "wtq": config.wtq_dataset,
        "tabfact": config.tabfact_dataset,
        "crt": config.crt_dataset,
    }
    for dataset, dataset_path in dataset_paths.items():
        write_executable(
            run_dir / f"run_mact_{dataset}_paired200.sh",
            render_mact_script(config, run_dir, gate_manifest, dataset, dataset_path),
        )
    write_executable(run_dir / "run_eval_and_compare.sh", render_eval_compare_script(config, run_dir, gate_manifest))
    (run_dir / "README.md").write_text(render_readme(config, run_dir, gate_manifest), encoding="utf-8")
    manifest = build_manifest(config, run_dir, gate_manifest)
    (run_dir / "paired200_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--myagent-root", type=Path, default=Path("/home/ubuntu/lzz/MyAgent"))
    parser.add_argument("--mact-root", type=Path, default=Path("/home/ubuntu/lzz/MACT"))
    parser.add_argument("--gate-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    config = Paired200Config(
        myagent_root=args.myagent_root.resolve(),
        mact_root=args.mact_root.resolve(),
        gate_run_dir=args.gate_run_dir.resolve(),
        run_dir=args.run_dir.resolve() if args.run_dir else None,
    )
    manifest = prepare_paired200_run(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
