# Model Gate Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run small but comparable model-gate experiments to determine whether current myAgent can exceed MACT under the same model, data split, and evaluator before any costly formal run.

**Architecture:** Keep Qwen3-32B as the main evidence path and use frozen, table-diverse JSONL splits for paired myAgent-vs-MACT comparisons. Use smaller candidate models only as quick gates; a model only advances if it has complete outputs, low failure rate, and promising accuracy/token behavior.

**Tech Stack:** Python 3 in `lzz-agent`, vLLM OpenAI-compatible server, `scripts/server/run_sharded_tqa.py`, `code/evaluate_results.py`, `code/compare_blind_results.py`, MACT `code/tqa.py`, JSONL datasets under `datasets_ready`.

## Global Constraints

- Do not optimize further on TabFact before measuring cross-dataset behavior.
- Use same model, same frozen split, same temperature 0, same evaluator for myAgent and MACT comparisons.
- Preserve eval rows, merged rows, token usage, elapsed time, failed count, missing answer count, and command lines.
- Do not run full datasets during model screening; full myAgent is estimated near 75 h and full MACT near 569 h on current Qwen3 speed.
- Avoid committing ignored runtime env files, logs, pids, or `outputs/`.
- Do not overwrite existing user changes in `configs/server/qwen3_32b_2gpu_local.env.example`, `configs/server/qwen3_32b_2gpu_local.env.bak.20260709_191806`, or `restart_qwen3_context_try.sh`.

---

### Task 1: Confirm Current State and Available Models

**Files:**
- Read: `/home/ubuntu/lzz/MyAgent`
- Read: `/home/ubuntu/models`
- Read: `/home/ubuntu/lzz/MACT/outputs/server_runs`

**Interfaces:**
- Consumes: existing Git branch and server state.
- Produces: a confirmed model list and current baseline summary for later report sections.

- [x] **Step 1: Inspect branch and dirty worktree**

Run:

```bash
cd /home/ubuntu/lzz/MyAgent
git status --short --branch
git log --oneline -5
```

Expected: branch is `codex/selective-risk-collaboration`; only unrelated local config/restart files may remain dirty.

- [x] **Step 2: Inspect available local models**

Run:

```bash
find /home/ubuntu/models -maxdepth 2 -type f \( -name config.json -o -name tokenizer_config.json -o -name '*.safetensors' \) \
  | sed 's#/config.json##; s#/tokenizer_config.json##; s#/[^/]*\.safetensors##' \
  | sort -u
```

Expected: current local candidates are `/home/ubuntu/models/Qwen3-32B` and `/home/ubuntu/models/Qwen2.5-3B-Instruct`.

- [x] **Step 3: Inspect GPU/port availability**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
python - <<'PY'
import socket
for port in [8000, 8010, 8011, 8020]:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect(("127.0.0.1", port))
        print(port, "open")
    except Exception:
        print(port, "closed")
    finally:
        sock.close()
PY
```

Expected: Qwen3 remains on port 8000 and GPU 5/6; port 8010 is available for Qwen2.5-3B.

### Task 2: Prepare Qwen2.5-3B Runtime Profile

**Files:**
- Create: `configs/server/qwen25_3b_1gpu_local.env.example`
- Create ignored local file: `configs/server/qwen25_3b_1gpu_local.env`

**Interfaces:**
- Consumes: `scripts/server/start_vllm_pool.sh` env contract.
- Produces: a reproducible example profile and a local ignored profile using port 8010.

- [x] **Step 1: Add a tracked example env profile**

Create `configs/server/qwen25_3b_1gpu_local.env.example` with:

```bash
export HF_HOME=/home/ubuntu/models
export HF_HUB_ENABLE_HF_TRANSFER=1
export MODEL_ID=/home/ubuntu/models/Qwen2.5-3B-Instruct
export SERVED_MODEL_NAME=qwen25-3b-local
export GPU_GROUPS="0"
export BASE_PORT=8010
export VLLM_API_KEY=local-vllm-key-change-me
export VLLM_MAX_MODEL_LEN=8192
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export VLLM_DTYPE=auto
export VLLM_EXTRA_ARGS="--trust-remote-code"
export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"
```

- [x] **Step 2: Add the ignored local env profile**

Create `configs/server/qwen25_3b_1gpu_local.env` with the same values. Confirm it is ignored by `.gitignore` and not staged.

### Task 3: Start and Verify Qwen2.5-3B Service

**Files:**
- Read: `logs/server/vllm_8010.log`
- Read: `pids/server/vllm_8010.pid`
- Modify: `scripts/server/start_vllm_pool.sh`
- Test: `tests/test_start_vllm_pool.py`

**Interfaces:**
- Consumes: `configs/server/qwen25_3b_1gpu_local.env`.
- Produces: an OpenAI-compatible endpoint at `http://127.0.0.1:8010/v1` serving `qwen25-3b-local`.

- [x] **Step 1: Start vLLM**

Run:

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
bash scripts/server/start_vllm_pool.sh configs/server/qwen25_3b_1gpu_local.env
```

Expected: pid is written to `pids/server/vllm_8010.pid`.

- [x] **Step 2: Wait for healthcheck**

Run:

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
until bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen25_3b_1gpu_local.env; do
  tail -40 logs/server/vllm_8010.log
  sleep 10
done
```

Expected: chat completion content is exactly `ok`.

- [x] **Step 3: Fix noninteractive process persistence**

The first start attempt proved that plain background `&`, and then `nohup` alone, did not keep 8010 alive after the Codex command returned. Add `tests/test_start_vllm_pool.py` and change `scripts/server/start_vllm_pool.sh` to launch with:

```bash
setsid nohup env ... vllm serve ... > "${log_file}" 2>&1 < /dev/null &
```

Run:

```bash
python tests/test_start_vllm_pool.py
bash -n scripts/server/start_vllm_pool.sh
```

Expected: both commands pass, and an independent healthcheck after the startup command returns still gets `ok` from port 8010.

### Task 4: Run Qwen2.5-3B myAgent Smoke

**Files:**
- Create ignored output directory: `outputs/server_runs/qwen25_3b_policy_v5_smoke5_all`

**Interfaces:**
- Consumes: `qwen25-3b-local` endpoint and full default datasets.
- Produces: 5 rows per dataset, eval JSON, and logs.

- [x] **Step 1: Run smoke**

Run:

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen25_3b_1gpu_local.env

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8010/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen25_3b_policy_v5_smoke5_all \
  --limit-per-task 5 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

Expected: each log ends with `Finished sample 5/5`.

- [x] **Step 2: Verify smoke outputs**

Run:

```bash
cd /home/ubuntu/lzz/MyAgent
wc -l outputs/server_runs/qwen25_3b_policy_v5_smoke5_all/raw/*/*_out.jsonl \
      outputs/server_runs/qwen25_3b_policy_v5_smoke5_all/merged/*.jsonl
python - <<'PY'
import glob, json
for path in sorted(glob.glob("outputs/server_runs/qwen25_3b_policy_v5_smoke5_all/eval/*_eval.json")):
    data = json.load(open(path))
    print(path, data["num_samples"], data["primary_accuracy"], data["avg_total_tokens"], data.get("num_failed_exec"), data.get("num_missing_answer"))
PY
```

Expected: raw and merged rows are all 5; failed/missing are recorded.

### Task 5: Decide Whether to Run Qwen2.5-3B 50-Sample Paired Gate

**Files:**
- Read: `outputs/server_runs/qwen25_3b_policy_v5_smoke5_all/eval/*_eval.json`
- Optionally create ignored outputs under `outputs/server_runs/qwen25_3b_policy_v5_50_all`
- Optionally create ignored MACT outputs under `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen25_3b_50`

**Interfaces:**
- Consumes: smoke eval metrics.
- Produces: a go/no-go decision for 50-sample paired gate.

- [x] **Step 1: Apply smoke gate**

Proceed only if:

```text
all three datasets produce complete 5/5 outputs
failed rate is not obviously pathological
model responses are parseable enough to compute primary_accuracy
avg_total_tokens does not exceed Qwen3 policy v5 by a large margin
```

- [x] **Step 2: If smoke passes, run myAgent 50**

Run the same command as Task 4 with:

```bash
--output-root outputs/server_runs/qwen25_3b_policy_v5_50_all
--limit-per-task 50
```

- [x] **Step 3: If myAgent 50 is not clearly hopeless, run MACT 50**

Decision: do not run Qwen2.5-3B MACT 50. myAgent 50 was complete but only 84/150 = 56.0% overall, with WTQ 38.0% and CRT 52.0%; this model is not a useful formal candidate.

### Task 6: Report and Commit

**Files:**
- Create or modify: `docs/server/server_codex_reports/2026-07-19-model-gate-results.md`
- Commit: tracked env example, plan, and report only.

**Interfaces:**
- Consumes: Qwen2.5-3B smoke or gate results.
- Produces: a clear recommendation for the next formal paired Qwen3/Qwen2.5 experiment.

- [x] **Step 1: Write report**

Report:

```text
docs/server/server_codex_reports/2026-07-19-model-gate-results.md
```

- [ ] **Step 2: Run verification**

Run:

```bash
cd /home/ubuntu/lzz/MyAgent
git diff --check
python tests/test_server_runner.py
python -m py_compile scripts/server/run_sharded_tqa.py
```

Expected: exit code 0.

- [ ] **Step 3: Commit and push**

Run:

```bash
git add configs/server/qwen25_3b_1gpu_local.env.example \
        docs/superpowers/plans/2026-07-19-model-gate-experiments.md \
        docs/server/server_codex_reports/2026-07-19-model-gate-results.md
git commit -m "Document model gate experiment setup"
git push origin codex/selective-risk-collaboration
```

Expected: ignored runtime env and outputs are not committed.

### Task 7: Prepare Frozen Qwen3 Formal Split

**Files:**
- Create: `datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl`
- Create: `datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl`
- Create: `datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl`
- Create: `datasets_ready/frozen_qwen3_eval_150_2026-07-19/manifest.json`
- Modify: `docs/server/server_codex_reports/2026-07-19-model-gate-results.md`

**Interfaces:**
- Consumes: full adapted datasets and historical outputs.
- Produces: a reproducible 150/数据集 table-diverse split with zero prior id/table overlap.

- [x] **Step 1: Freeze the split**

Run:

```bash
python code/freeze_blind_holdout.py \
  --wtq_input datasets_ready/full/wtq_unseen.jsonl \
  --tabfact_input datasets_ready/full/tabfact_test.jsonl \
  --crt_input datasets_ready/full/crt.jsonl \
  --output_dir datasets_ready/frozen_qwen3_eval_150_2026-07-19 \
  --history_root outputs \
  --history_root /home/ubuntu/lzz/MACT/outputs \
  --sample_size 150 \
  --seed 20260719
```

Expected: each dataset has 150 records and 150 unique tables; prior id/table overlap is 0.

- [x] **Step 2: Verify row counts and manifest**

Run:

```bash
wc -l datasets_ready/frozen_qwen3_eval_150_2026-07-19/*.jsonl
python -m json.tool datasets_ready/frozen_qwen3_eval_150_2026-07-19/manifest.json
```

Expected: each JSONL has 150 rows; manifest parses as JSON.

- [x] **Step 3: Verify runner dataset overrides with dry-run**

Run:

```bash
python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model qwen3-32b-local \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root /tmp/myagent_frozen150_dry_20260719 \
  --dry-run
```

Expected: dry-run prints WTQ, TabFact, and CRT commands using the temporary shard paths.
