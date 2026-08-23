#!/usr/bin/env python3
"""Prepare a MACT-hosted P0 baseline experiment run directory."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


DATASETS = {
    "wtq": "datasets_ready/full/wtq_unseen.jsonl",
    "tabfact": "datasets_ready/full/tabfact_test.jsonl",
    "crt": "datasets_ready/full/crt.jsonl",
}
DEFAULT_ENDPOINTS = "http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"


def shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable and os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def git_head(repo: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def default_run_dir(mact_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return mact_root / "outputs" / "server_runs" / f"qwen3_32b_baseline_formal200_{stamp}"


def copy_inputs(myagent_root: Path, run_dir: Path, formal_limit: int, ablation_limit: int, smoke_limit: int) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"inputs": {}}
    for dataset, relative_path in DATASETS.items():
        source = myagent_root / relative_path
        rows = read_jsonl(source)
        slices = {
            "formal200": rows[:formal_limit],
            "ablation50": rows[:ablation_limit],
            "smoke5": rows[:smoke_limit],
        }
        manifest["inputs"][dataset] = {
            "source": str(source),
            "available_rows": len(rows),
        }
        for slice_name, slice_rows in slices.items():
            target = run_dir / "input" / slice_name / f"{dataset}.jsonl"
            write_jsonl(target, slice_rows)
            manifest["inputs"][dataset][slice_name] = {
                "path": str(target),
                "rows": len(slice_rows),
                "first_id": str(slice_rows[0].get("id", "")) if slice_rows else "",
                "last_id": str(slice_rows[-1].get("id", "")) if slice_rows else "",
            }
    return manifest


def header(myagent_root: Path, run_dir: Path) -> List[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"MYAGENT_ROOT={shell_quote(myagent_root)}",
        f"RUN_DIR={shell_quote(run_dir)}",
        'cd "$MYAGENT_ROOT"',
        "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh",
        "conda activate lzz-agent",
        'source "$RUN_DIR/env.sh"',
        "",
    ]


def render_env(served_model_name: str, endpoints: str, api_key_env: str) -> str:
    return "\n".join(
        [
            "# Baseline experiment runtime configuration.",
            f'export SERVED_MODEL_NAME="${{SERVED_MODEL_NAME:-{served_model_name}}}"',
            f'export BASELINE_ENDPOINTS="${{BASELINE_ENDPOINTS:-{endpoints}}}"',
            f'export API_KEY_ENV="${{API_KEY_ENV:-{api_key_env}}}"',
            f': "${{{api_key_env}:?set {api_key_env} before running online experiments}}"',
            "",
        ]
    )


def dataset_args(run_dir: Path, slice_name: str) -> List[str]:
    return [
        f'  --wtq-dataset "$RUN_DIR/input/{slice_name}/wtq.jsonl" \\',
        f'  --tabfact-dataset "$RUN_DIR/input/{slice_name}/tabfact.jsonl" \\',
        f'  --crt-dataset "$RUN_DIR/input/{slice_name}/crt.jsonl" \\',
    ]


def render_baseline_script(
    myagent_root: Path,
    run_dir: Path,
    *,
    baseline: str,
    slice_name: str,
    output_name: str,
    limit: int,
) -> str:
    lines = header(myagent_root, run_dir)
    lines.extend(
        [
            "python scripts/server/run_baseline_tqa.py \\",
            "  --repo-root . \\",
            f"  --baseline {baseline} \\",
            "  --tasks wtq,tabfact,crt \\",
        ]
    )
    lines.extend(dataset_args(run_dir, slice_name))
    lines.extend(
        [
            '  --endpoints "$BASELINE_ENDPOINTS" \\',
            '  --model "$SERVED_MODEL_NAME" \\',
            '  --api-key-env "$API_KEY_ENV" \\',
            f'  --output-root "$RUN_DIR/{output_name}" \\',
            f"  --limit-per-task {limit} \\",
            "  --thinking disabled \\",
            "  --temperature 0 \\",
            "  --max-tokens 1024 \\",
            "  --resume",
            "",
        ]
    )
    return "\n".join(lines)


def render_myagent_script(
    myagent_root: Path,
    run_dir: Path,
    *,
    slice_name: str,
    output_name: str,
    limit: int,
    extra_args: List[str] | None = None,
) -> str:
    lines = header(myagent_root, run_dir)
    lines.extend(
        [
            "python scripts/server/run_sharded_tqa.py \\",
            "  --repo-root . \\",
            "  --tasks wtq,tabfact,crt \\",
        ]
    )
    lines.extend(dataset_args(run_dir, slice_name))
    lines.extend(
        [
            '  --endpoints "$BASELINE_ENDPOINTS" \\',
            '  --model "$SERVED_MODEL_NAME" \\',
            '  --api-key-env "$API_KEY_ENV" \\',
            f'  --output-root "$RUN_DIR/{output_name}" \\',
            f"  --limit-per-task {limit} \\",
            "  --max-replan 3 \\",
            "  --mact-avg-tokens 47439.2633 \\",
            "  --thinking disabled \\",
            "  --temperature 0 \\",
            "  --max-tokens 2048 \\",
        ]
    )
    for arg in extra_args or []:
        lines.append(f"  {arg} \\")
    lines.append("  --resume")
    lines.append("")
    return "\n".join(lines)


def mact_task(dataset: str) -> str:
    return "scitab" if dataset == "tabfact" else dataset


def render_mact_script(myagent_root: Path, mact_root: Path, run_dir: Path, dataset: str, endpoint_index: int) -> str:
    endpoint_expr = f'$(echo "$BASELINE_ENDPOINTS" | cut -d, -f{endpoint_index + 1})'
    lines = header(myagent_root, run_dir)
    lines.extend(
        [
            'mkdir -p "$RUN_DIR/mact" "$RUN_DIR/logs" "$RUN_DIR/tmp"',
            "python scripts/server/run_mact_one_by_one.py \\",
            f"  --mact-root {shell_quote(mact_root)} \\",
            f'  --dataset-path "$RUN_DIR/input/formal200/{dataset}.jsonl" \\',
            f'  --output-path "$RUN_DIR/mact/{dataset}_mact_formal200.jsonl" \\',
            f'  --log-path "$RUN_DIR/logs/mact_{dataset}_formal200.log" \\',
            f"  --task {mact_task(dataset)} \\",
            '  --plan-model-name "$SERVED_MODEL_NAME" \\',
            '  --code-model-name "$SERVED_MODEL_NAME" \\',
            "  --model-provider openai_compatible \\",
            f"  --api-base {endpoint_expr} \\",
            '  --api-key-env "$API_KEY_ENV" \\',
            "  --thinking disabled \\",
            "  --temperature 0 \\",
            "  --max-tokens 2048 \\",
            "  --api-timeout 180 \\",
            "  --api-max-retries 5 \\",
            "  --plan-sample 1 \\",
            "  --code-sample 1 \\",
            "  --max-step 3 \\",
            "  --max-actual-step 3 \\",
            '  --temp-dir "$RUN_DIR/tmp" \\',
            "  --limit 200 \\",
            "  --resume",
            "",
        ]
    )
    return "\n".join(lines)


def render_healthcheck(myagent_root: Path, run_dir: Path) -> str:
    lines = header(myagent_root, run_dir)
    lines.extend(
        [
            'IFS="," read -r -a ENDPOINT_ARRAY <<< "$BASELINE_ENDPOINTS"',
            'for API_BASE in "${ENDPOINT_ARRAY[@]}"; do',
            '  echo "[healthcheck] $API_BASE"',
            '  curl -sS -H "Authorization: Bearer ${!API_KEY_ENV}" "$API_BASE/models" >/dev/null',
            "done",
            "",
        ]
    )
    return "\n".join(lines)


def render_eval_summary(myagent_root: Path, run_dir: Path) -> str:
    lines = header(myagent_root, run_dir)
    lines.extend(
        [
            'mkdir -p "$RUN_DIR/eval" "$RUN_DIR/summary"',
            "for DATASET in wtq tabfact crt; do",
            '  if [ -f "$RUN_DIR/mact/${DATASET}_mact_formal200.jsonl" ]; then',
            "    python code/evaluate_results.py \\",
            '      "$RUN_DIR/mact/${DATASET}_mact_formal200.jsonl" \\',
            '      --error_output "$RUN_DIR/eval/${DATASET}_mact_formal200_errors.jsonl" \\',
            '      > "$RUN_DIR/eval/${DATASET}_mact_formal200_eval.json"',
            "  fi",
            "done",
            "python scripts/server/summarize_baseline_experiment.py \\",
            '  --run-dir "$RUN_DIR"',
            "",
        ]
    )
    return "\n".join(lines)


def render_checkpoint(mact_root: Path, run_dir: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"MACT_ROOT={shell_quote(mact_root)}",
            f"RUN_DIR={shell_quote(run_dir)}",
            'MESSAGE="${1:-results: checkpoint baseline formal200 package}"',
            'cd "$MACT_ROOT"',
            'RUN_REL="${RUN_DIR#$MACT_ROOT/}"',
            'git add -f -- "$RUN_REL"',
            'git commit -m "$MESSAGE"',
            "git push origin HEAD",
            "",
        ]
    )


def render_readme(run_dir: Path) -> str:
    return f"""# Qwen3-32B Baseline Formal-200 Package

Prepared run directory: `{run_dir}`

This package is GPU-ready but does not start any model by itself. Wait until the Qwen3-32B endpoint(s) are available, then run scripts from this directory.

## Execution Order

1. `bash healthcheck_services.sh`
2. `bash run_smoke_direct_cot.sh`
3. `bash run_smoke_single_agent_pandas.sh`
4. Inspect smoke `merged/` and `eval/`; stop if row count is not exactly 5 per dataset.
5. `bash run_formal_myagent.sh`
6. `bash run_formal_direct_cot.sh`
7. `bash run_formal_single_agent_pandas.sh`
8. `bash run_mact_wtq_formal200.sh`
9. `bash run_mact_tabfact_formal200.sh`
10. `bash run_mact_crt_formal200.sh`
11. `bash run_eval_and_summary.sh`

## Ablation Scripts

Run these after the main Formal-200 table is stable:

- `bash run_ablation_legacy50.sh`
- `bash run_ablation_no_strong50.sh`
- `bash run_ablation_no_deterministic_shortcuts50.sh`
- `bash run_ablation_no_question_routing50.sh`
- `bash run_ablation_no_risk_scoring50.sh`
- `bash run_ablation_no_table_compression50.sh`

These scripts isolate the implemented command-line switches. If runtime is limited, run the three existing completed ablations first and then run the new routing/risk/compression ablations one at a time.

## Output Layout

- `input/formal200/`: fixed 200-row inputs per dataset.
- `input/smoke5/`: fixed 5-row smoke inputs per dataset.
- `input/ablation50/`: fixed 50-row ablation inputs per dataset.
- `myagent_formal200/`
- `direct_cot_formal200/`
- `single_agent_pandas_formal200/`
- `mact/`
- `ablation/`
- `summary/main_baseline_summary.md`
"""


def build_manifest(args: argparse.Namespace, run_dir: Path, input_manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "purpose": "P0 baseline experiment package for MyAgent vs MACT vs Direct-CoT vs Single-Agent Pandas",
        "myagent_root": str(Path(args.myagent_root).resolve()),
        "mact_root": str(Path(args.mact_root).resolve()),
        "run_dir": str(run_dir),
        "myagent_commit": git_head(Path(args.myagent_root).resolve()),
        "mact_commit": git_head(Path(args.mact_root).resolve()),
        "served_model_name": args.served_model_name,
        "endpoints": [item.strip() for item in args.endpoints.split(",") if item.strip()],
        "api_key_env": args.api_key_env,
        "formal_limit": args.formal_limit,
        "ablation_limit": args.ablation_limit,
        "smoke_limit": args.smoke_limit,
        **input_manifest,
        "p0_status": {
            "main_baselines": "prepared_not_run",
            "direct_cot_runner": "implemented",
            "single_agent_pandas_runner": "implemented",
            "myagent_ablation_existing_switches": [
                "collaboration_mode=legacy",
                "disable_strong_verification",
                "disable_deterministic_shortcuts",
                "disable_question_routing",
                "disable_risk_scoring",
                "disable_table_compression",
            ],
            "missing_ablation_switches": [],
        },
    }


def prepare(args: argparse.Namespace) -> Path:
    myagent_root = Path(args.myagent_root).resolve()
    mact_root = Path(args.mact_root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else default_run_dir(mact_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    input_manifest = copy_inputs(
        myagent_root,
        run_dir,
        formal_limit=args.formal_limit,
        ablation_limit=args.ablation_limit,
        smoke_limit=args.smoke_limit,
    )
    manifest = build_manifest(args, run_dir, input_manifest)
    write_text(run_dir / "baseline_run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(run_dir / "env.sh", render_env(args.served_model_name, args.endpoints, args.api_key_env))
    write_text(run_dir / "README.md", render_readme(run_dir))
    write_text(run_dir / "healthcheck_services.sh", render_healthcheck(myagent_root, run_dir), executable=True)
    write_text(
        run_dir / "run_smoke_direct_cot.sh",
        render_baseline_script(myagent_root, run_dir, baseline="direct_cot", slice_name="smoke5", output_name="direct_cot_smoke5", limit=args.smoke_limit),
        executable=True,
    )
    write_text(
        run_dir / "run_smoke_single_agent_pandas.sh",
        render_baseline_script(myagent_root, run_dir, baseline="single_agent_pandas", slice_name="smoke5", output_name="single_agent_pandas_smoke5", limit=args.smoke_limit),
        executable=True,
    )
    write_text(
        run_dir / "run_formal_direct_cot.sh",
        render_baseline_script(myagent_root, run_dir, baseline="direct_cot", slice_name="formal200", output_name="direct_cot_formal200", limit=args.formal_limit),
        executable=True,
    )
    write_text(
        run_dir / "run_formal_single_agent_pandas.sh",
        render_baseline_script(myagent_root, run_dir, baseline="single_agent_pandas", slice_name="formal200", output_name="single_agent_pandas_formal200", limit=args.formal_limit),
        executable=True,
    )
    write_text(
        run_dir / "run_formal_myagent.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="formal200", output_name="myagent_formal200", limit=args.formal_limit),
        executable=True,
    )
    write_text(
        run_dir / "run_ablation_legacy50.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="ablation50", output_name="ablation/legacy_gate50", limit=args.ablation_limit, extra_args=["--collaboration-mode legacy"]),
        executable=True,
    )
    write_text(
        run_dir / "run_ablation_no_strong50.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="ablation50", output_name="ablation/no_strong_gate50", limit=args.ablation_limit, extra_args=["--disable-strong-verification"]),
        executable=True,
    )
    write_text(
        run_dir / "run_ablation_no_deterministic_shortcuts50.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="ablation50", output_name="ablation/no_deterministic_shortcuts_gate50", limit=args.ablation_limit, extra_args=["--disable-deterministic-shortcuts"]),
        executable=True,
    )
    write_text(
        run_dir / "run_ablation_no_question_routing50.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="ablation50", output_name="ablation/no_question_routing_gate50", limit=args.ablation_limit, extra_args=["--disable-question-routing"]),
        executable=True,
    )
    write_text(
        run_dir / "run_ablation_no_risk_scoring50.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="ablation50", output_name="ablation/no_risk_scoring_gate50", limit=args.ablation_limit, extra_args=["--disable-risk-scoring"]),
        executable=True,
    )
    write_text(
        run_dir / "run_ablation_no_table_compression50.sh",
        render_myagent_script(myagent_root, run_dir, slice_name="ablation50", output_name="ablation/no_table_compression_gate50", limit=args.ablation_limit, extra_args=["--disable-table-compression"]),
        executable=True,
    )
    for index, dataset in enumerate(("wtq", "tabfact", "crt")):
        write_text(
            run_dir / f"run_mact_{dataset}_formal200.sh",
            render_mact_script(myagent_root, mact_root, run_dir, dataset, endpoint_index=index % 2),
            executable=True,
        )
    write_text(run_dir / "run_eval_and_summary.sh", render_eval_summary(myagent_root, run_dir), executable=True)
    write_text(run_dir / "checkpoint_to_git.sh", render_checkpoint(mact_root, run_dir), executable=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--myagent-root", default="/home/ubuntu/lzz/MyAgent")
    parser.add_argument("--mact-root", default="/home/ubuntu/lzz/MACT")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--served-model-name", default="qwen3-32b-local")
    parser.add_argument("--endpoints", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--api-key-env", default="LOCAL_VLLM_API_KEY")
    parser.add_argument("--formal-limit", type=int, default=200)
    parser.add_argument("--ablation-limit", type=int, default=50)
    parser.add_argument("--smoke-limit", type=int, default=5)
    return parser


def main() -> None:
    run_dir = prepare(build_parser().parse_args())
    print(run_dir)


if __name__ == "__main__":
    main()
