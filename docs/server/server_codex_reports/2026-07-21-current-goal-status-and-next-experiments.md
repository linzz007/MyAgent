# 2026-07-21 Current Goal Status and Next Experiments

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
基线提交：`680f35b Record current experiment plan`
目标：继续排查当前实验问题，筛选可超过 MACT 的模型，并给出不跑 full 多天的正式实验路径。

## 1. Current Verdict

当前本地可用模型均已完成小样本筛选：

| model | local path | status | decision |
|---|---|---|---|
| Qwen3-32B | `/home/ubuntu/models/Qwen3-32B` | passed Gate-50 and frozen150 strict paired | current main model |
| Qwen3-14B-AWQ | `/home/ubuntu/models/Qwen3-14B-AWQ` | Gate-50 runs, but below MACT | no-go for expansion |
| Qwen2.5-14B-AWQ | `/home/ubuntu/models/Qwen2.5-14B-Instruct-AWQ` | Gate-50 runs, but below MACT | no-go for expansion |
| Qwen2.5-3B-Instruct | `/home/ubuntu/models/Qwen2.5-3B-Instruct` | Gate-50 runs, but clearly below MACT | no-go for expansion |

结论：

1. 当前没有未筛选的本地候选模型；继续多模型测试前，需要先下载或挂载新模型。
2. 当前 Qwen3-32B 服务健康，非沙箱 `healthcheck_vllm_pool.sh` 返回 `ok`。
3. 之前的问题没有复现：当前 blind200 / frozen150 guard 日志未检出 `Connection refused`、context length、APIConnectionError 或 shard 缺行。
4. 当前主证据足够支持阶段 gate：Qwen3-32B frozen150 strict paired 为 myAgent `342/450 = 0.7600` vs MACT `330/450 = 0.7333`，token ratio `0.6161`，myAgent failed/missing 为 `0`。
5. 不能写成“所有模型都超过 MACT”或“full dataset 已经稳定全面超过 MACT”；目前只有 Qwen3-32B 是主候选。
6. 本轮发现并修复了一个 vLLM 管理脚本风险：`pids/server/vllm_8000.pid` 已 stale，但 port `8000` 仍有真实 Qwen3-32B listener。`start_vllm_pool.sh` 已增加 stale pid + live port 检查，避免在已有服务上重复启动。
7. 已补跑 blind200 same-split MACT smoke5：WTQ/TabFact/CRT 各 5 条均完整落盘，wrapper returncode 全为 0，未检出连接、context length 或 API 错误。同 ID paired smoke 为 myAgent `14/15` vs MACT `10/15`，myAgent 平均 token 为 MACT 的 `0.7400`，平均耗时为 MACT 的 `0.1562`。这只能证明正式 paired 路径可跑，不能替代 full paired 结论。

## 2. Current Resource Constraints

磁盘状态：

```text
overlay 200G, used 197G, available 3.2G, use 99%
```

主要占用：

| path | size | note |
|---|---:|---|
| `/home/ubuntu/models` | 86G | main model storage |
| `/home/ubuntu/miniconda3` | 26G | environment |
| `/home/ubuntu/lzz` | 5.2G | repos and local datasets |
| `/home/ubuntu/.vscode-server` | 4.0G | editor runtime |
| `/home/ubuntu/.cache/vllm` | 1.9G | compile cache |
| `/home/ubuntu/lzz/MyAgent/outputs/server_runs` | 241M | experiment outputs, not the disk bottleneck |
| `/home/ubuntu/lzz/MACT/outputs` | 6.8M | MACT outputs, not the disk bottleneck |

模型目录大小：

| model dir | size |
|---|---:|
| `/home/ubuntu/models/Qwen3-32B` | 62G |
| `/home/ubuntu/models/Qwen3-14B-AWQ` | 9.4G |
| `/home/ubuntu/models/Qwen2.5-14B-Instruct-AWQ` | 9.4G |
| `/home/ubuntu/models/Qwen2.5-3B-Instruct` | 5.8G |

判断：

- 删除 `outputs/server_runs` 不能解决空间问题，只能释放约 241M。
- 如果要下载新模型，需要先删除或迁移 no-go 模型目录，或清理 vLLM compile cache；这是 destructive action，应先由用户确认。
- 保留 Qwen3-32B，因为它是当前主模型。

GPU 状态：

| gpu | memory used / total | note |
|---:|---:|---|
| 0 | 1209 / 49140 MiB | mostly free |
| 1 | 17433 / 49140 MiB | occupied by unknown process/cache |
| 2 | 17497 / 49140 MiB | occupied by unknown process/cache |
| 3 | 3 / 49140 MiB | free |
| 4 | 3 / 49140 MiB | free |
| 5 | 45815 / 49140 MiB | Qwen3-32B service |
| 6 | 45815 / 49140 MiB | Qwen3-32B service |

Qwen3-32B current service:

```text
MODEL_ID=/home/ubuntu/models/Qwen3-32B
SERVED_MODEL_NAME=qwen3-32b-local
GPU_GROUPS="5,6"
BASE_PORT=8000
VLLM_MAX_MODEL_LEN=8192
```

服务管理修复验证：

```text
before fix:
  pids/server/vllm_8000.pid = 47338
  /proc/47338 missing
  actual vLLM process = 17816, port 8000, Qwen3-32B

after fix:
  start_vllm_pool.sh removed stale pid 47338
  detected port 8000 already has a listener
  skipped duplicate start
  healthcheck_vllm_pool.sh returned ok
```

后续长跑前不要只信 pid 文件；必须跑 healthcheck。

## 3. Model Screening Summary

Strict Gate-50 MACT reference:

| dataset | MACT correct | avg tokens |
|---|---:|---:|
| WTQ | 41/50 | 10,346.36 |
| TabFact | 48/50 | 10,442.78 |
| CRT | 31/50 | 12,998.10 |
| Overall | 120/150 | 11,262.41 |

Qwen3-32B current strict Gate-50 subtotal comes from the latest WTQ shortcutfix, TabFact shortcutfix, and CRT strict paired runs:

| dataset | myAgent correct | MACT correct | myAgent avg tokens | token ratio | status |
|---|---:|---:|---:|---:|---|
| WTQ | 41/50 | 41/50 | 6,246.82 | 0.6038 | tied MACT |
| TabFact | 49/50 | 48/50 | 2,631.46 | 0.2520 | beats MACT |
| CRT | 34/50 | 31/50 | 12,920.72 | 0.9940 | beats MACT, token almost tied |
| Overall | 124/150 | 120/150 | 7,266.33 | 0.6452 | gate passed |

Other local models:

| model | WTQ | TabFact | CRT | overall | avg tokens | avg seconds | failed/missing | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-14B-AWQ | 37/50 | 44/50 | 27/50 | 108/150 = 0.720 | 7,344.51 | 6.50s | 0/0 | no-go |
| Qwen2.5-14B-AWQ | 34/50 | 45/50 | 28/50 | 107/150 = 0.713 | 7,308.35 | 6.92s | 0/0 | no-go |
| Qwen2.5-3B-Instruct | 28/50 | 40/50 | 21/50 | 89/150 = 0.593 | 7,298.51 | 4.26s | 0/0 | no-go |

The 14B and 3B models are faster, but they do not reduce average tokens versus Qwen3-32B myAgent and lose too much accuracy. They should not enter Gate-100/150/200.

## 4. Current Qwen3-32B Evidence

Strict paired main evidence:

| scope | myAgent | MACT | accuracy delta | token ratio | accepted |
|---|---:|---:|---:|---:|---|
| frozen150 strict paired | 342/450 = 0.7600 | 330/450 = 0.7333 | +2.67 pp | 0.6161 | yes |

Current-code blind200 stress test:

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ shortcutfix2 | 131/200 | 0.6550 | 6,226.93 | 15.939s | 0 | 0 |
| TabFact current | 185/200 | 0.9250 | 2,426.89 | 10.755s | 0 | 0 |
| CRT current | 137/200 | 0.6850 | 10,838.25 | 24.899s | 0 | 0 |
| Overall | 453/600 | 0.7550 | 6,497.36 | 17.198s | 0 | 0 |

限制：

- blind200 目前是 myAgent-only stress test，没有同 split Qwen3 MACT 输出。
- 因此 blind200 不能写成 strict paired 超过 MACT，只能写“current-code 稳定、总体性能可用、token 仍明显低于 frozen150 MACT 均值”。

## 5. Next Experiments

### Option A: No New Model, Formal Paired 200

如果不下载新模型，下一步最有价值的是补跑 blind200 同 split MACT，和已经完成的 myAgent blind200 组成 formal paired 200/数据集主表。

已有 myAgent 输出：

```text
outputs/server_runs/qwen3_32b_current_blind200_20260721/
outputs/server_runs/qwen3_32b_current_blind200_wtq200_shortcutfix2_20260721/
```

要补跑的 MACT 输入：

```text
datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl
datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl
datasets_ready/blind_holdout_200_v1_2026-06-27/crt.jsonl
```

预计成本：

| dataset | MACT frozen150 avg sec/sample | blind200 estimated wall |
|---|---:|---:|
| WTQ | 117.2s | about 6.5h |
| TabFact | 96.4s | about 5.4h |
| CRT | 173.9s | about 9.7h |
| Total sequential | - | about 21.6h |

这不是五天级别，但仍然是长跑。建议只在磁盘和服务状态确认后，用 `tmux` 或明确后台脚本执行，并全程 `--resume`。

本轮实际 MACT smoke5 重新估计：

| dataset | wall for 5 | row-level avg sec/sample | blind200 projected wall |
|---|---:|---:|---:|
| WTQ | 10m15s | 120.99s | about 6.8h |
| TabFact | 7m10s | 84.02s | about 4.8h |
| CRT | 12m08s | 143.48s | about 8.1h |
| Total sequential | 29m33s | 116.16s overall | about 19.7h |

如果只跑 blind50 paired，大约 4.9h；blind100 paired 大约 9.9h。多模型场景不应每个模型都跑 full paired，应先用 myAgent-only Gate-50/150 淘汰，再只给进入候选的模型补 MACT paired。

### Option B: New Model Screening

若用户提供或允许下载新模型，流程固定为：

1. 先释放空间：删除或迁移 no-go 模型目录，保留 Qwen3-32B。
2. 下载/挂载新模型。
3. 启动单 endpoint smoke。
4. 跑 frozen Gate-50 myAgent-only。
5. 只有 overall 接近 Qwen3-32B 或超过 MACT Gate-50，才跑 MACT paired。
6. 若 Gate-50 不过，直接记录 no-go，不进入扩大实验。

Gate-50 命令模板：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

export LOCAL_VLLM_API_KEY=local-vllm-key-change-me
RUN_ROOT=outputs/server_runs/<model_tag>_gate50_$(date +%Y%m%d_%H%M%S)

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:<port>/v1 \
  --model <served_model_name> \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11262.41 \
  --resume
```

Gate acceptance:

| criterion | threshold |
|---|---|
| raw/merged/eval completeness | 50 rows per dataset |
| failed/missing | 0 for smoke; <= 2% for larger gate |
| logs | no connection/context/API errors |
| overall accuracy | >= MACT Gate-50 or close to Qwen3-32B |
| datasets | at least 2 datasets not below MACT |
| token | <= 0.75 MACT preferred; CRT can be watched separately |

### Option C: Ablation for Patent Mechanism Explanation

Do not expand ablations to full data. Use 50 or 100 samples only:

| ablation | command switch | purpose |
|---|---|---|
| legacy collaboration | `--collaboration-mode legacy` | show selective-risk policy contribution |
| no strong verification | `--disable-strong-verification` | show verifier contribution/cost |
| no deterministic shortcuts | `--disable-deterministic-shortcuts` | show semantic shortcut contribution |
| replan budget | `--max-replan 0/1/2` | show adaptive correction effect |

Recommended first ablation table for experts:

```text
frozen Gate-50, Qwen3-32B, WTQ/TabFact/CRT, same evaluator.
```

## 6. Recommended Immediate Action

当前不建议继续盲目跑本地模型，因为没有未筛选候选。下一步按优先级：

1. 若目标是尽快写专家/专利主表：补跑 blind200 MACT strict paired，预计约 22h sequential。
2. 若目标是继续找更优模型：先释放模型磁盘空间，再下载一个新候选，只跑 Gate-50。
3. 若目标是解释机制贡献：先跑 Qwen3-32B frozen Gate-50 ablation，不跑 full。

当前阶段可写入专家材料：

```text
在本地可用模型筛选中，Qwen3-32B 是唯一通过 strict paired stage gate 的候选；
Qwen3-14B-AWQ、Qwen2.5-14B-AWQ、Qwen2.5-3B 均可稳定运行但准确率不足。
Qwen3-32B 在 frozen150 strict paired 中以 342/450 对 330/450 超过 MACT，
平均 API token 为 MACT 的 61.6%，且 failed/missing 为 0。
```

不应写：

```text
所有模型都超过 MACT。
blind200 已严格证明超过 MACT。
full 数据集已经完成。
```

## 7. Blind200 MACT Smoke5

目的：验证当前 Qwen3-32B 服务、MACT one-by-one wrapper、blind200 输入、同 ID 对齐和 evaluator 口径是否可以支撑后续 formal paired run。

运行目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_smoke5_20260721
```

输入：

```text
datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl
datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl
datasets_ready/blind_holdout_200_v1_2026-06-27/crt.jsonl
```

运行方式：`scripts/server/run_mact_one_by_one.py`，Qwen3-32B local vLLM，`--limit 5 --resume`，`--thinking disabled`，`--max-step 3 --max-actual-step 3`。

落盘完整性：

| output | rows |
|---|---:|
| `wtq_mact_smoke5.jsonl` | 5 |
| `tabfact_mact_smoke5.jsonl` | 5 |
| `crt_mact_smoke5.jsonl` | 5 |
| total | 15 |

MACT smoke evaluator summary:

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 4/5 | 0.8000 | 9,705.80 | 120.995s | 0 | 0 | 1 |
| TabFact | 4/5 | 0.8000 | 8,300.60 | 84.020s | 0 | 0 | 1 |
| CRT | 2/5 | 0.4000 | 11,089.60 | 143.478s | 0 | 0 | 3 |
| Overall | 10/15 | 0.6667 | 9,698.67 | 116.165s | 0 | 0 | 5 |

同 ID myAgent vs MACT smoke:

| dataset | ids | myAgent | MACT | myAgent avg tokens | MACT avg tokens | token ratio | myAgent avg sec | MACT avg sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | `nu-2923, nu-58, nu-1874, nu-4342, nu-2772` | 5/5 | 4/5 | 5,049.80 | 9,705.80 | 0.5203 | 11.335s | 120.995s |
| TabFact | `tabfact-test-6439, tabfact-test-11416, tabfact-test-3557, tabfact-test-1176, tabfact-test-2658` | 5/5 | 4/5 | 2,026.00 | 8,300.60 | 0.2441 | 9.859s | 84.020s |
| CRT | `crt-601, crt-419, crt-391, crt-387, crt-543` | 4/5 | 2/5 | 14,456.60 | 11,089.60 | 1.3036 | 33.246s | 143.478s |
| Overall | 15 matched rows | 14/15 | 10/15 | 7,177.47 | 9,698.67 | 0.7400 | 18.147s | 116.165s |

Paired disagreement:

| dataset | both correct | myAgent only | MACT only | neither |
|---|---:|---:|---:|---:|
| WTQ | 4 | 1 | 0 | 0 |
| TabFact | 4 | 1 | 0 | 0 |
| CRT | 2 | 2 | 0 | 1 |
| Overall | 10 | 4 | 0 | 1 |

Log scan:

```text
Traceback / context length / Connection refused / APIConnectionError / ERROR: none found
wrapper returncode: 15/15 are returncode=0
MACT internal trial halted: 2/15 rows (WTQ 1, CRT 1), with output rows still preserved
```

结论：

1. 之前的服务启动问题当前没有复现；Qwen3-32B endpoint 能连续支撑 MACT WTQ/TabFact/CRT smoke。
2. MACT formal paired path 可执行，建议正式长跑继续使用 one-by-one + `--resume`，并保留 wrapper returncode、internal halted、eval mismatch、failed/missing 四类诊断。
3. 15 条 smoke 上 myAgent 同时更准、更省 token、更快，但样本太小，只能作为 pipeline evidence 和成本估计，不应写成正式性能结论。

## 8. Practical Formal Experiment Plan

当前项目可以进入 staged formal evaluation，但不建议对每个模型直接 full 600 paired。推荐方案：

| stage | scope | expected cost | pass rule | output |
|---|---|---:|---|---|
| S0 service smoke | each model, 1-2 samples per dataset | minutes | healthcheck ok, no missing rows | service readiness |
| S1 myAgent Gate-50 | 50 per dataset, myAgent-only | short | no failed/missing; overall near or above MACT Gate-50 | model shortlist |
| S2 myAgent Gate-150 | frozen150, myAgent-only unless already done | medium | beats/near MACT with clear token advantage | candidate confirmation |
| S3 paired core | same IDs, 50 per dataset against MACT | about 4.9h for Qwen3-32B MACT | myAgent >= MACT and token lower | expert-ready paired table |
| S4 paired expansion | same IDs, 100 per dataset | about 9.9h | only if S3 passes and more evidence needed | stronger paired appendix |
| S5 full blind200 paired | 200 per dataset | about 19.7h | only final selected model | final main table if compute budget allows |

专家/专利材料建议写法：

```text
先以 Qwen3-32B frozen150 strict paired 作为当前主证据；
blind200 myAgent-only 作为稳定性和泛化压力测试；
blind200 MACT smoke5 作为正式 paired pipeline 可执行性的运行记录；
后续正式报告只对入围模型补 blind50/100 paired，而不是对所有模型 full paired。
```

下一次最稳的命令策略：

```text
1. healthcheck Qwen3-32B endpoint
2. run MACT one-by-one with --resume and per-dataset output/log
3. after each dataset: wc -l, evaluate_results.py, grep error patterns
4. paired compare only matched IDs
5. append report before expanding sample size
```

当前是否符合要求：

```text
符合“进入正式 staged evaluation”的要求：
- Qwen3-32B 是唯一已通过 strict paired stage gate 的本地候选；
- current myAgent blind200 三数据集 600 条已完整、无 failed/missing；
- MACT blind200 same-split smoke5 已验证可跑；
- token 优势在 frozen150 和 smoke15 总体仍明显存在。

尚不符合“blind200 strict paired 已正式证明超过 MACT”的要求：
- blind200 MACT 目前只跑了 smoke5，不是 full paired；
- 若专家材料需要 blind200 paired 主表，至少补 blind50/100 paired，再决定是否 full200。
```

## 9. Blind50 Core Paired Final

用户要求本轮流程文档实时更新，并且测试结果必须保存到 MACT 文件夹、防止服务器数据丢失。本轮已建立 MACT 侧 live ledger，并将中间 checkpoint 与最终结果多次推送到 MACT GitHub。

MACT run directory:

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core50_20260722
```

MACT live ledger:

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core50_20260722/LIVE_LEDGER.md
```

MACT GitHub final sync:

```text
repo: git@github.com:linzz007/MACT.git
branch: main
commit: d630184 Record Qwen3 blind core50 final results
```

核心产物：

| file | purpose |
|---|---|
| `wtq_mact_core50.jsonl` | WTQ first50 MACT raw outputs |
| `tabfact_mact_core50.jsonl` | TabFact first50 MACT raw outputs |
| `crt_mact_core50.jsonl` | CRT first50 MACT raw outputs |
| `*_mact_core50_eval.json` | per-dataset evaluator summary |
| `*_mact_core50_paired.json` | same-ID myAgent vs MACT paired details |
| `*_mact_core50_errors.jsonl` | evaluator anomaly rows, not only execution failures |
| `overall_mact_core50_summary.json` | final 150-row paired summary |
| `logs/*_mact_core50.log` | raw MACT logs and diagnostics |

运行完整性：

| dataset | rows | wall time | MACT failed | MACT missing | critical error |
|---|---:|---:|---:|---:|---|
| WTQ | 50/50 | 92m58s | 1 | 1 | `nu-4299` context length |
| TabFact | 50/50 | 79m09s | 0 | 0 | none |
| CRT | 50/50 | 125m58s | 0 | 0 | none |
| Overall | 150/150 | 4h58m04s | 1 | 1 | 1 row |

Per-dataset paired result:

| dataset | myAgent | MACT | accuracy delta | myAgent avg tokens | MACT avg tokens | token ratio |
|---|---:|---:|---:|---:|---:|---:|
| WTQ | 34/50 = 0.6800 | 41/50 = 0.8200 | -14.00 pp | 5,775.70 | 10,579.62 | 0.546 |
| TabFact | 48/50 = 0.9600 | 49/50 = 0.9800 | -2.00 pp | 2,445.44 | 10,441.52 | 0.234 |
| CRT | 42/50 = 0.8400 | 29/50 = 0.5800 | +26.00 pp | 12,774.78 | 12,538.42 | 1.019 |
| Overall | 124/150 = 0.8267 | 119/150 = 0.7933 | +3.33 pp | 6,998.64 | 11,186.52 | 0.626 |

Overall paired disagreement:

| both correct | myAgent only | MACT only | neither |
|---:|---:|---:|---:|
| 106 | 18 | 13 | 13 |

Diagnostics:

```text
MACT internal Halted: WTQ 5 rows, TabFact 4 rows, CRT 19 rows.
Critical log hits: 2 log lines, both from the same WTQ context length failure.
Context failure detail: 6145 input tokens + 2048 requested output tokens exceeds 8192 context by 1 token.
```

Stage verdict:

1. Blind50 core paired overall passes: myAgent is `+5/150` correct over MACT and uses `62.6%` of MACT tokens.
2. The result must not be overstated: WTQ and TabFact individually are below MACT on this blind50 slice; the overall win comes from CRT.
3. The next expert-facing table should include both per-dataset rows and the overall row.
4. For the next expansion, prefer blind100 paired before full blind200. If WTQ remains below MACT, do not claim dataset-wide dominance; claim overall selective-risk efficiency with per-task caveats.

## 10. Active Workflow Ledger

本章节作为当前目标的唯一流程文档。所有实时状态、阶段判断、下一步实验选择先写在这里；所有测试原始结果、日志、eval、paired summary 必须保存在 MACT run 目录，并用 `git add -f` 同步到 MACT GitHub。

当前目标：

```text
在不继续优先优化 TabFact 的前提下，验证当前 myAgent/Qwen3-32B 是否总体超过 MACT，
并建立一个服务器不稳定时可恢复、可同步、可扩展到后续模型筛选和正式实验的流程。
```

当前执行原则：

| rule | implementation |
|---|---|
| 流程只维护一份 | 本章节记录总体流程、阶段 verdict、下一步动作 |
| 测试结果保存到 MACT | MACT raw/eval/paired/log 都放在 `/home/ubuntu/lzz/MACT/outputs/server_runs/...` |
| 防止数据丢失 | 每个检查点分别 commit/push MyAgent 流程文档和 MACT 结果目录 |
| 不跑无意义 full | 先 blind50，再 blind100；只有入围模型才考虑 full blind200 |
| 不夸大结论 | 分数据集输赢和 overall 输赢同时写，不能把 overall win 写成所有数据集 win |

当前 MACT blind100 run：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722
```

对应 MACT live ledger：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/LIVE_LEDGER.md
```

GitHub 同步状态：

| repo | branch | last synced checkpoint |
|---|---|---|
| MyAgent | `codex/selective-risk-collaboration` | `adf915f` blind50 paired report |
| MACT | `main` | `7e7b5ae` blind100 seed |

2026-07-22 19:32:47 CST 实时检查：

| dataset | MACT rows | failed | missing | last id | note |
|---|---:|---:|---:|---|---|
| WTQ | 54/100 | 1 | 1 | `nu-484` | inherited `nu-4299` context failure; rows 51-54 completed |
| TabFact | 50/100 | 0 | 0 | `tabfact-test-11867` | seed from blind50 |
| CRT | 50/100 | 0 | 0 | `crt-279` | seed from blind50 |

进程状态：

```text
No active scripts/server/run_mact_one_by_one.py or MACT code/tqa.py process found.
The prior core100 runner session was interrupted/lost after WTQ row 54.
```

2026-07-22 21:21:35 CST 恢复脚本检查：

```text
Detached WTQ resume script first launch exited with status 127 before adding rows:
/usr/bin/time is not available on this server.
Fix: run_wtq_resume.sh now uses bash SECONDS for elapsed-time logging.
Rows remain WTQ 54/100, TabFact 50/100, CRT 50/100 before relaunch.
```

2026-07-23 00:08:45 CST WTQ resume checkpoint:

| dataset | MACT rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 61/100 | 1 | 1 | `nu-1990` | active pid `318083` |
| TabFact | 50/100 | 0 | 0 | `tabfact-test-11867` | not started for core100 tail |
| CRT | 50/100 | 0 | 0 | `crt-279` | not started for core100 tail |

2026-07-23 00:24:15 CST WTQ resume checkpoint:

| dataset | MACT rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 70/100 | 1 | 1 | `nu-2232` | active pid `318083` |
| TabFact | 50/100 | 0 | 0 | `tabfact-test-11867` | not started for core100 tail |
| CRT | 50/100 | 0 | 0 | `crt-279` | not started for core100 tail |

2026-07-23 00:44:32 CST WTQ resume checkpoint:

| dataset | MACT rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 81/100 | 1 | 1 | `nu-1125` | active pid `318083` |
| TabFact | 50/100 | 0 | 0 | `tabfact-test-11867` | not started for core100 tail |
| CRT | 50/100 | 0 | 0 | `crt-279` | not started for core100 tail |

2026-07-23 01:02:33 CST WTQ resume checkpoint:

| dataset | MACT rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 91/100 | 1 | 1 | `nu-2502` | active pid `318083` |
| TabFact | 50/100 | 0 | 0 | `tabfact-test-11867` | not started for core100 tail |
| CRT | 50/100 | 0 | 0 | `crt-279` | not started for core100 tail |

2026-07-23 01:22:10 CST WTQ final checkpoint:

| dataset | MACT rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 100/100 | 2 | 2 | `nu-216` | exited status 0 |
| TabFact | 50/100 | 0 | 0 | `tabfact-test-11867` | not started for core100 tail |
| CRT | 50/100 | 0 | 0 | `crt-279` | not started for core100 tail |

WTQ final diagnostics:

```text
Failed/missing IDs: nu-4299, nu-2633.
Both failures are MACT context length BadRequest: 6145 input tokens + 2048 output tokens > 8192.
Connection refused / APIConnectionError: 0.
Internal Halted: 1 count in WTQ log: 8.
WTQ resume elapsed after row54: 5,243 seconds.
```

2026-07-23 01:24:28 CST TabFact/CRT resume preparation:

```text
Qwen3 healthcheck: ok.
Active MACT runner before TabFact start: 0.
Created MACT-side resume scripts:
- run_tabfact_resume.sh
- run_crt_resume.sh
```

2026-07-23 08:43:38 CST TabFact resume checkpoint:

| dataset | MACT rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 100/100 | 2 | 2 | `nu-216` | complete |
| TabFact | 61/100 | 0 | 0 | `tabfact-test-10012` | active pid `334723` |
| CRT | 50/100 | 0 | 0 | `crt-279` | not started for core100 tail |

下一步恢复策略：

1. WTQ final checkpoint 已完成并准备同步。
2. 下一步跑 TabFact：`--limit 100 --resume`，从现有 50 行继续到 100。
3. TabFact 到 100 后立即更新 MACT ledger、commit/push MACT。
4. 再按同样方式跑 CRT 到 100，并同步。
5. 三个数据集到 100 后生成 `*_eval.json`、`*_paired.json`、`overall_mact_core100_summary.json`。
6. 最后把 blind100 paired result 写回本章节，并推送 MyAgent。

当前阶段可写入专家材料的结论仍以 blind50 为准：

```text
Qwen3-32B blind50 same-ID paired overall: myAgent 124/150 vs MACT 119/150,
myAgent 平均 token 为 MACT 的 62.6%。WTQ 和 TabFact 单项未超过 MACT，CRT 明显超过；
因此当前应写“总体超过且 token 显著更低”，不能写“三个数据集全部超过”。
```
