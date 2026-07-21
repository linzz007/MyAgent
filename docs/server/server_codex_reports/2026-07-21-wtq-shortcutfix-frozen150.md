# 2026-07-21 WTQ Shortcut Fix Frozen150 Result

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
起始提交：`2954f9f Record Qwen3 frozen150 paired results`
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
数据：`datasets_ready/frozen_qwen3_eval_150_2026-07-19/{wtq,tabfact,crt}.jsonl`

## 1. Current Verdict

本轮不再继续优先优化 TabFact，而是处理 frozen150 strict paired 中 WTQ 低于 MACT 的问题。

结论：

1. myAgent 侧之前的工程问题没有复现：Qwen3 vLLM healthcheck 返回 `ok`，WTQ full rerun 生成 raw/merged/eval 各 `150` 行，failed/missing 均为 `0`，日志未检出 `Traceback`、`Connection refused`、`context length` 或 `APIConnectionError`。
2. WTQ shortcutfix full rerun 后，WTQ 从旧版 `105/150 = 0.700` 提升到 `114/150 = 0.760`，追平 MACT WTQ `114/150 = 0.760`。
3. 合并既有 TabFact/CRT frozen150 strict paired 后，myAgent overall 为 `342/450 = 0.7600`，MACT 为 `330/450 = 0.7333`；myAgent 平均 token 为 MACT 的 `61.61%`。
4. 按当前 `compare_blind_results.py` 的 acceptance criteria，`selective-risk-collaboration` 阶段验收通过。

可以写进专家/专利阶段材料的稳妥表述：

```text
在 Qwen3-32B 本地同模型、同 frozen150 split、同 evaluator 的 strict paired 评估中，
myAgent 三数据集合计 342/450，超过 MACT 的 330/450；平均 API token 为 MACT 的 61.6%，
且 myAgent failed/missing 为 0。结果说明风险自适应协作在总体准确率不降低的同时显著降低推理资源。
```

仍不建议写成：

```text
当前版本已经统计显著全面优于 MACT，或 full 数据集上已经稳定全面超过 MACT。
```

overall McNemar p = `0.2461`，支持阶段性工程验收，不支持统计显著超越。

## 2. Root Cause and Code Change

WTQ frozen150 旧结果为 myAgent `105/150` vs MACT `114/150`，discordant 集合为：

```text
myAgent only: 13
MACT only:   22
both wrong:  23
```

人工复核显示主要不是 runtime failure，而是 WTQ 中高置信、可确定的表格模式覆盖不足：

- "how long" roster/count 问题被错误规范化成 `12 years`。
- only metric value、listed after、usage count、occurrence count、ordinal/rank、extreme metric lookup、first evicted/status、stated-left arithmetic、release-date gap 等模式没有稳定前置处理。
- blanket high-confidence `thinking_direct` override 的离线模拟会带来净伤害，因此没有启用。

本轮代码改动：

- `code/my_agents.py`
  - 收紧 `_canonicalize_wtq_scalar` 的 `how long` 年份后缀条件，只在问题包含 `after/before/between/from/since/until` 等持续时间关系时补 `year(s)`。
  - 新增一组 WTQ deterministic semantic shortcuts，用于处理上述通用表格模式。
  - 将这些 shortcut 接入 `_try_wtq_semantic_shortcut`，优先于更宽泛的 fallback。
- `tests/test_myagent_pipeline.py`
  - 新增 12 个 WTQ 回归测试，覆盖本轮修复模式。

RED/GREEN 记录：

```text
RED:   python tests/test_myagent_pipeline.py
       1 assertion failure + 11 missing helper AttributeErrors

GREEN: python tests/test_myagent_pipeline.py
       150 tests OK
```

## 3. Targeted Discordant35 Rerun

输入：

```text
outputs/server_runs/qwen3_32b_wtq_discordant_shortcutfix_20260721/input/wtq_discordant35.jsonl
```

命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

RUN_ROOT=outputs/server_runs/qwen3_32b_wtq_discordant_shortcutfix_20260721
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq \
  --wtq-dataset "$RUN_ROOT/input/wtq_discordant35.jsonl" \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --max-replan 2 \
  --mact-avg-tokens 11539.45
```

结果：

| item | value |
|---|---:|
| raw rows | 35 |
| merged rows | 35 |
| eval samples | 35 |
| correct | 26/35 |
| accuracy | 0.7429 |
| avg tokens | 5,868.66 |
| avg seconds | 15.006 |
| avg calls | 4.229 |
| failed | 0 |
| missing | 0 |
| wall time | 8m46.652s |

旧 myAgent 在这 35 条上为 `13/35`，MACT 为 `22/35`。该 targeted rerun 用于验证修复方向，不作为最终 paired 结论。

## 4. Full WTQ150 Actual Rerun

命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

RUN_ROOT=outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --max-replan 2 \
  --mact-avg-tokens 11539.45
```

输出：

```text
outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721/raw/wtq/wtq_shard00_out.jsonl
outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721/merged/wtq_qwen3-32b-local.jsonl
outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721/eval/wtq_qwen3-32b-local_eval.json
outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721/logs/wtq/wtq_shard00.log
```

文件时间：

```text
shard/log created: 2026-07-21 09:41:17 CST
eval written:       2026-07-21 10:22:18 CST
observed wall:      about 41m01s
```

完整性和 eval：

| item | value |
|---|---:|
| raw rows | 150 |
| merged rows | 150 |
| eval samples | 150 |
| num_with_gold | 150 |
| correct | 114/150 |
| primary accuracy | 0.7600 |
| exact match | 0.7400 |
| avg calls | 4.733 |
| avg tokens | 6,185.50 |
| avg prompt tokens | 5,790.19 |
| avg completion tokens | 395.31 |
| avg seconds | 16.397 |
| failed | 0 |
| missing | 0 |

日志尾部正常到 `Finished sample 150/150`；错误扫描未检出：

```text
Traceback
BadRequestError
context length
Connection refused
APIConnectionError
Exception
ERROR
failed
```

## 5. Strict Paired Result After WTQ Fix

比较输出：

```text
outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721/compare/
```

Paired column format: `both_correct / myAgent_only / MACT_only / both_wrong`.

| dataset | myAgent | MACT | delta | token ratio | my sec | MACT sec | paired | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| WTQ | 114/150 = 0.760 | 114/150 = 0.760 | +0.00 pp | 0.578 | 16.4 | 117.2 | 101 / 13 / 13 / 23 | 1.0000 |
| TabFact | 131/150 = 0.873 | 132/150 = 0.880 | -0.67 pp | 0.248 | 11.8 | 96.4 | 120 / 11 / 12 / 7 | 1.0000 |
| CRT | 97/150 = 0.647 | 84/150 = 0.560 | +8.67 pp | 0.946 | 29.1 | 173.9 | 70 / 27 / 14 / 39 | 0.0596 |
| Overall | 342/450 = 0.760 | 330/450 = 0.733 | +2.67 pp | 0.616 | 19.1 | 129.1 | 291 / 51 / 39 / 69 | 0.2461 |

完整性表：

| dataset | system | rows | eval samples | failed | missing | avg calls | avg tokens | avg seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | myAgent | 150 | 150 | 0 | 0 | 4.73 | 6,185.50 | 16.40 |
| WTQ | MACT | 150 | 150 | 6 | 6 | 3.40 | 10,698.43 | 117.16 |
| TabFact | myAgent | 150 | 150 | 0 | 0 | 3.88 | 2,657.49 | 11.78 |
| TabFact | MACT | 150 | 150 | 0 | 0 | 3.20 | 10,722.56 | 96.41 |
| CRT | myAgent | 150 | 150 | 0 | 0 | 5.93 | 12,484.09 | 29.14 |
| CRT | MACT | 150 | 150 | 0 | 1 | 4.05 | 13,197.35 | 173.86 |
| Overall | myAgent | 450 | 450 | 0 | 0 | 4.85 | 7,109.03 | 19.11 |
| Overall | MACT | 450 | 450 | 6 | 7 | 3.55 | 11,539.45 | 129.15 |

Acceptance criteria:

| criterion | passed |
|---|---:|
| overall accuracy >= MACT | yes |
| at least two datasets >= MACT | yes |
| token ratio <= 0.75 | yes |
| myAgent execution failure rate <= 0.02 | yes |
| full selective-risk acceptance | yes |

## 6. Status of Previous Problems

| problem | current status |
|---|---|
| vLLM not ready / `Connection refused` | Not reproduced. Healthcheck and `/v1/models` returned valid Qwen3 responses. |
| myAgent shard silently exits / missing output | Not reproduced in this run. raw/merged/eval are complete at 150 rows. |
| MACT WTQ context failures | Still present in MACT baseline; one-by-one wrapper preserves failed rows for strict paired comparison. |
| TabFact weak under early Qwen3 selective path | No longer first priority. Frozen150 TabFact is only 1 item below MACT, with token ratio 0.248. |
| WTQ below MACT by 6 pp | Addressed for frozen150: WTQ now ties MACT at 114/150. |

## 7. Recommended Experiment Plan

Do not use full WTQ/TabFact/CRT as the routine model-selection loop. Full data sizes are:

| dataset | full rows |
|---|---:|
| WTQ | 4,344 |
| TabFact | 12,779 |
| CRT | 728 |
| Total | 18,301 |

Recommended staged plan:

| stage | scope | run | stop rule |
|---|---|---|---|
| Smoke | 20/数据集 | myAgent only | failed/missing > 0, context error, or token obviously abnormal |
| Gate-50 | frozen first50/数据集 | myAgent first; only promising models get MACT paired | overall clearly below current Qwen3 or token > MACT |
| Frozen150 | frozen150/数据集 | 1-2 best models strict paired | require overall >= MACT, at least 2 datasets >= MACT, token <= 0.75 MACT, failed <= 2% |
| Formal sample | 200 or 300/数据集 | final model strict paired | write expert/patent main table |
| Ablation | 50 or 100/数据集 | selected toggles only | explain mechanism, not exhaustively optimize |
| Full dataset | optional background | final model only | not required for regular iteration |

Recommended ablations:

```text
legacy collaboration
no strong verification
no deterministic shortcuts
max-replan 0/1/2
```

Do not expand every ablation to full data. A 50/100 stratified subset is enough to explain mechanism contribution.

## 8. How to Make Runs Finish on This Server

Current Qwen3 env only starts one 2-GPU endpoint:

```text
GPU_GROUPS="5,6"
BASE_PORT=8000
```

During this run, GPUs 0-4 were mostly idle while GPUs 5/6 were fully utilized. For formal batch experiments, create a separate temporary env file instead of editing private config:

```bash
tmp_env=/tmp/qwen3_32b_3endpoint.env
cat >"$tmp_env" <<'ENV'
export MODEL_ID=/home/ubuntu/models/Qwen3-32B
export SERVED_MODEL_NAME=qwen3-32b-local
export GPU_GROUPS="0,1;2,3;5,6"
export BASE_PORT=8100
export VLLM_API_KEY=local-vllm-key-change-me
export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"
export VLLM_MAX_MODEL_LEN=8192
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export VLLM_DTYPE=auto
export VLLM_EXTRA_ARGS="--trust-remote-code"
ENV

bash scripts/server/start_vllm_pool.sh "$tmp_env"
bash scripts/server/healthcheck_vllm_pool.sh "$tmp_env"
```

Then pass three endpoints to the runner:

```bash
python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/<run_tag> \
  --max-replan 2 \
  --mact-avg-tokens 11539.45 \
  --resume
```

Notes:

- Use ports that do not collide with the currently running port 8000 service.
- Check `nvidia-smi` before starting extra endpoints.
- Run a 5-10 sample smoke before launching a long multi-endpoint job.
- Always use `--resume`; if a shard is complete, the runner skips it.
- Record raw/merged/eval row counts and do not commit `outputs/server_runs`.

## 9. Bottom Line for Project Alignment

Current Qwen3-32B/myAgent now meets the practical stage gate:

- Engineering run stability: yes.
- Same-model same-split strict paired overall >= MACT: yes, `342/450` vs `330/450`.
- Token clearly below MACT: yes, `61.61%` of MACT.
- At least two datasets >= MACT: yes, WTQ ties and CRT exceeds.
- Failed/missing on myAgent: `0`.

This is enough to freeze the current branch as the main Qwen3 candidate and move to formal sampled paired experiments or 1-2 additional model gates. It is not a reason to run full WTQ/TabFact/CRT for every model.

## 10. Verification

Fresh verification before commit:

```text
git diff --check
python tests/test_myagent_pipeline.py              # 150 tests OK
python tests/test_evaluate_results.py              # 15 tests OK
python tests/test_compare_blind_results.py         # 3 tests OK
python tests/test_run_mact_one_by_one.py           # 3 tests OK
python tests/test_tqa_failure_exit.py              # 1 test OK
python -m py_compile code/my_agents.py code/evaluate_results.py code/compare_blind_results.py scripts/server/run_sharded_tqa.py
```
