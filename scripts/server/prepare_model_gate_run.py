#!/usr/bin/env python3
"""Prepare a MACT-hosted run directory for new-model Gate-10/Gate-50 screening."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_WTQ_DATASET = "datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl"
DEFAULT_TABFACT_DATASET = "datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl"
DEFAULT_CRT_DATASET = "datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl"
DEFAULT_GPU_GROUPS = "4,5;6,7"
DEFAULT_BASE_PORT = 8000
DEFAULT_API_KEY = "local-vllm-key-change-me"
DEFAULT_MACT_AVG_TOKENS = 11262.41


@dataclass(frozen=True)
class GateRunConfig:
    myagent_root: Path
    mact_root: Path
    model_id: Path
    model_tag: str
    served_model_name: str
    run_dir: Path | None = None
    gpu_groups: str = DEFAULT_GPU_GROUPS
    base_port: int = DEFAULT_BASE_PORT
    vllm_api_key: str = DEFAULT_API_KEY
    vllm_max_model_len: int = 8192
    vllm_gpu_memory_utilization: float = 0.88
    vllm_dtype: str = "auto"
    vllm_extra_args: str = "--trust-remote-code"
    wtq_dataset: str = DEFAULT_WTQ_DATASET
    tabfact_dataset: str = DEFAULT_TABFACT_DATASET
    crt_dataset: str = DEFAULT_CRT_DATASET
    max_replan: int = 2
    mact_avg_tokens: float = DEFAULT_MACT_AVG_TOKENS


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def endpoints_for(gpu_groups: str, base_port: int) -> list[str]:
    groups = [group.strip() for group in gpu_groups.split(";") if group.strip()]
    if not groups:
        raise ValueError("gpu_groups must contain at least one group")
    return [f"http://127.0.0.1:{base_port + index}/v1" for index, _ in enumerate(groups)]


def default_run_dir(mact_root: Path, model_tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return mact_root / "outputs" / "server_runs" / f"{safe_slug(model_tag)}_gate50_{stamp}"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def render_vllm_env(config: GateRunConfig, run_dir: Path) -> str:
    lines = [
        f"export HF_HOME={shell_quote('/home/ubuntu/models')}",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        f"export MODEL_ID={shell_quote(config.model_id)}",
        f"export SERVED_MODEL_NAME={shell_quote(config.served_model_name)}",
        f"export GPU_GROUPS={shell_quote(config.gpu_groups)}",
        f"export BASE_PORT={config.base_port}",
        f"export VLLM_API_KEY={shell_quote(config.vllm_api_key)}",
        f"export VLLM_MAX_MODEL_LEN={config.vllm_max_model_len}",
        f"export VLLM_GPU_MEMORY_UTILIZATION={config.vllm_gpu_memory_utilization}",
        f"export VLLM_DTYPE={shell_quote(config.vllm_dtype)}",
        f"export VLLM_EXTRA_ARGS={shell_quote(config.vllm_extra_args)}",
        'export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"',
        f"export RUN_DIR={shell_quote(run_dir)}",
        "",
    ]
    return "\n".join(lines)


def render_service_script(config: GateRunConfig, run_dir: Path, action: str) -> str:
    if action == "start":
        command = 'bash scripts/server/start_vllm_pool.sh "$RUN_DIR/vllm.env"'
    elif action == "healthcheck":
        command = 'bash scripts/server/healthcheck_vllm_pool.sh "$RUN_DIR/vllm.env"'
    elif action == "stop":
        command = "bash scripts/server/stop_vllm_pool.sh"
    else:
        raise ValueError(f"unsupported action: {action}")
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"MYAGENT_ROOT={shell_quote(config.myagent_root)}",
            f"RUN_DIR={shell_quote(run_dir)}",
            'cd "$MYAGENT_ROOT"',
            "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh",
            "conda activate lzz-agent",
            command,
            "",
        ]
    )


def render_gate_script(config: GateRunConfig, run_dir: Path, gate_name: str, limit: int) -> str:
    endpoints = ",".join(endpoints_for(config.gpu_groups, config.base_port))
    output_root = f"$RUN_DIR/myagent_{gate_name}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"MYAGENT_ROOT={shell_quote(config.myagent_root)}",
        f"RUN_DIR={shell_quote(run_dir)}",
        'cd "$MYAGENT_ROOT"',
        "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh",
        "conda activate lzz-agent",
        'source "$RUN_DIR/vllm.env"',
        "python scripts/server/run_sharded_tqa.py \\",
        "  --repo-root . \\",
        "  --tasks wtq,tabfact,crt \\",
        f"  --wtq-dataset {shell_quote(config.wtq_dataset)} \\",
        f"  --tabfact-dataset {shell_quote(config.tabfact_dataset)} \\",
        f"  --crt-dataset {shell_quote(config.crt_dataset)} \\",
        f"  --endpoints {shell_quote(endpoints)} \\",
        '  --model "$SERVED_MODEL_NAME" \\',
        "  --api-key-env LOCAL_VLLM_API_KEY \\",
        f"  --output-root \"{output_root}\" \\",
        f"  --limit-per-task {limit} \\",
        f"  --max-replan {config.max_replan} \\",
        f"  --mact-avg-tokens {config.mact_avg_tokens} \\",
        "  --resume",
    ]
    if gate_name == "gate50":
        lines.extend(
            [
                "",
                "python scripts/server/summarize_model_gate_results.py \\",
                '  --gate-root "$RUN_DIR/myagent_gate50" \\',
                f"  --model-tag {shell_quote(config.model_tag)} \\",
                f"  --mact-avg-tokens {config.mact_avg_tokens} \\",
                '  --output "$RUN_DIR/gate50_summary.json" \\',
                '  --markdown-output "$RUN_DIR/gate50_summary.md"',
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_readme(config: GateRunConfig, run_dir: Path) -> str:
    return "\n".join(
        [
            f"# {config.model_tag} Gate Run",
            "",
            "This directory is for staged myAgent-only screening before any MACT paired expansion.",
            "",
            "Run order:",
            "",
            "```bash",
            f"bash {run_dir}/start_services.sh",
            f"bash {run_dir}/healthcheck_services.sh",
            f"bash {run_dir}/run_gate10.sh",
            f"bash {run_dir}/run_gate50.sh",
            f"bash {run_dir}/stop_services.sh",
            "```",
            "",
            "After Gate-50, inspect `gate50_summary.json` and `gate50_summary.md` before deciding whether to expand to Gate-150.",
            "",
            "Do not commit API keys. `vllm.env` contains only a local placeholder key by default.",
            "",
            "After a gate completes, force-add this ignored MACT output directory:",
            "",
            "```bash",
            f"cd {config.mact_root}",
            f"git add -f {run_dir}",
            "```",
            "",
        ]
    )


def build_manifest(config: GateRunConfig, run_dir: Path) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "myagent_root": str(config.myagent_root),
        "mact_root": str(config.mact_root),
        "model_id": str(config.model_id),
        "model_tag": config.model_tag,
        "served_model_name": config.served_model_name,
        "gpu_groups": config.gpu_groups,
        "base_port": config.base_port,
        "endpoints": endpoints_for(config.gpu_groups, config.base_port),
        "gate_limits": {"gate10": 10, "gate50": 50},
        "datasets": {
            "wtq": config.wtq_dataset,
            "tabfact": config.tabfact_dataset,
            "crt": config.crt_dataset,
        },
        "writes_outputs_to_mact": True,
        "starts_services": False,
    }


def prepare_gate_run(config: GateRunConfig) -> dict[str, Any]:
    run_dir = config.run_dir or default_run_dir(config.mact_root, config.model_tag)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()

    (run_dir / "vllm.env").write_text(render_vllm_env(config, run_dir), encoding="utf-8")
    write_executable(run_dir / "start_services.sh", render_service_script(config, run_dir, "start"))
    write_executable(run_dir / "healthcheck_services.sh", render_service_script(config, run_dir, "healthcheck"))
    write_executable(run_dir / "stop_services.sh", render_service_script(config, run_dir, "stop"))
    write_executable(run_dir / "run_gate10.sh", render_gate_script(config, run_dir, "gate10", 10))
    write_executable(run_dir / "run_gate50.sh", render_gate_script(config, run_dir, "gate50", 50))
    (run_dir / "README.md").write_text(render_readme(config, run_dir), encoding="utf-8")

    manifest = build_manifest(config, run_dir)
    (run_dir / "gate_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--myagent-root", type=Path, default=Path("/home/ubuntu/lzz/MyAgent"))
    parser.add_argument("--mact-root", type=Path, default=Path("/home/ubuntu/lzz/MACT"))
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--gpu-groups", default=DEFAULT_GPU_GROUPS)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    args = parser.parse_args()

    config = GateRunConfig(
        myagent_root=args.myagent_root.resolve(),
        mact_root=args.mact_root.resolve(),
        model_id=args.model_id.resolve(),
        model_tag=safe_slug(args.model_tag),
        served_model_name=args.served_model_name,
        run_dir=args.run_dir.resolve() if args.run_dir else None,
        gpu_groups=args.gpu_groups,
        base_port=args.base_port,
    )
    manifest = prepare_gate_run(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
