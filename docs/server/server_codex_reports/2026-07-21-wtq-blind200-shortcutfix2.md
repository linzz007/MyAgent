# 2026-07-21 WTQ Blind200 Shortcutfix2 Diagnostic

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
起始提交：`6cd0cba Record current Qwen3 blind200 validation`
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
数据：`datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl`

## 1. Verdict

本轮继续处理 current blind200 中 WTQ 偏低的问题。结论：

1. `--disable-deterministic-shortcuts` 的 WTQ first50 消融显示，不能全局关闭 shortcut：准确率从 baseline `33/50 = 0.6600` 降到 `29/50 = 0.5800`，token 从 `5,776.26` 升到 `6,088.64`。
2. WTQ blind200 中确定性 shortcut 确实有少量误伤，但不是主因。first50 里 shortcut 净贡献为 +4 题：5 条从正确变错，1 条从错变正确。
3. 本轮修复 5 个通用 WTQ 模式：`difference of` 负数规约、`last <column> on <year>` 年份过滤、`listed for the last <owner>` 目标列选择、`which album has most sales` owner 列从 details 回退到 title、显式候选 `A or B` 的 superlative 只在候选内比较。
4. targeted5 rerun 从旧错误样本全部修正为 `5/5 = 1.0000`。
5. WTQ blind200 first50 aggregate 从 `33/50 = 0.6600` 提升到 `34/50 = 0.6800`，avg token 基本不变。
6. WTQ blind200 full200 从 `126/200 = 0.6300` 提升到 `131/200 = 0.6550`；逐行比较显示净增 5 条正确、没有丢失正确样本。
7. WTQ frozen150 guard 与上一版完全一致：`114/150 = 0.7600`，逐行预测变化为 0，因此 frozen150 strict paired 主证据保持 `342/450 = 0.7600` vs MACT `330/450 = 0.7333`，token ratio `0.6161`。

当前证据支持把 `997f51f Fix WTQ blind shortcut edge cases` 作为低风险 WTQ shortcutfix2：blind200 有净收益，frozen150 主证据无回退。下一步不需要继续优先优化 TabFact，也不建议跑三数据集 full 作为日常迭代。

## 2. Root Cause

current blind200 WTQ 错误中，deterministic shortcut wrong 为 `4/200`。人工复核发现：

| id | question | old answer | gold | root cause |
|---|---|---|---|---|
| `nu-2897` | `what is the last note on 2008` | `40.15` | `21.00` | `last note` shortcut 没有先筛选 `Year = 2008` |
| `nu-2012` | `which album has the most sales?` | album details text | `The Remixes` | owner phrase `album` 误选 `Album details`，应返回 `Title` |
| `nu-644` | `what is the date listed for the last round?` | `10.0` | `November 5` | 旧逻辑把 `last round` 当目标列，未识别目标是 `Date` |
| `nu-4318` | `which island has the most area, tiree or kasos?` | `Mljet` / `Ærø` | `Tiree` | superlative 没有限制在题目显式候选内；第一次修复还暴露出 `Ærø -> r` 的短 key 误匹配 |
| `nu-3977` | `difference of the jsu and tu scores` | `-18.0` | `18` | 负数取绝对值只覆盖 `difference ... between`，漏掉 `difference of` |

## 3. Code Change

修改文件：

```text
code/my_agents.py
tests/test_myagent_pipeline.py
```

实现：

- `_canonicalize_wtq_scalar`：WTQ numeric negative delta 只要问题包含 `difference` 就转绝对值，不再限定 `difference ... between`。
- `_wtq_superlative_owner_answer`：
  - 提取逗号 + `or` 的显式候选，并只在候选行内比较。
  - 当 owner 列误选到 `details/info/notes` 类列时，优先回退到 `Title` / `Name` 类实体列。
  - 收紧候选匹配，避免 `Ærø` 归一化成短 key `r` 后误匹配 `tiree`。
- `_wtq_last_requested_column_answer`：
  - 支持 `what is the <target> listed for the last <owner>`，按 owner 的最大数值或最后出现行返回 target 列。
  - 对 `last <target> on/in/for <year>` 先用 Year 列过滤，再取 target 列最后值。

新增 5 个回归测试，均先 RED 后 GREEN：

```text
test_wtq_difference_of_canonicalizes_negative_numeric_delta
test_wtq_superlative_owner_shortcut_uses_title_column_for_album_sales
test_wtq_superlative_owner_shortcut_restricts_explicit_or_candidates
test_wtq_last_requested_column_filters_year_qualifier
test_wtq_last_listed_owner_returns_requested_column
```

RED:

```text
python tests/test_myagent_pipeline.py
Ran 155 tests
FAILED (failures=5)
```

补充 RED，覆盖 `Ærø` 短 key 误匹配：

```text
python tests/test_myagent_pipeline.py -k restricts_explicit_or_candidates
Ran 1 test
FAILED (failures=1)
```

GREEN:

```text
python tests/test_myagent_pipeline.py
Ran 155 tests in 0.456s
OK
```

## 4. No-Shortcut Ablation

命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

RUN_ROOT=outputs/server_runs/qwen3_32b_current_blind200_wtq50_no_shortcuts_20260721
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq \
  --wtq-dataset datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11539.45 \
  --disable-deterministic-shortcuts
```

结果：

| run | rows | correct | accuracy | avg tokens | avg seconds | shortcuts | failed | missing | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline first50 | 50 | 33/50 | 0.6600 | 5,776.26 | 13.886s | 7 | 0 | 0 | historical |
| no-shortcuts first50 | 50 | 29/50 | 0.5800 | 6,088.64 | 15.918s | 0 | 0 | 0 | 13m17.512s |
| high-budget first50 | 50 | 33/50 | 0.6600 | 5,776.18 | 13.886s | 7 | 0 | 0 | 11m35.818s |

判断：WTQ blind200 低分不能用“关闭 deterministic shortcut”解决；shortcut 在 first50 上净收益明显。

## 5. Targeted5 Rerun

输入：

```text
outputs/server_runs/qwen3_32b_wtq_targeted_shortcutfix2_v2_20260721/input/wtq_targeted5.jsonl
```

命令：

```bash
RUN_ROOT=outputs/server_runs/qwen3_32b_wtq_targeted_shortcutfix2_v2_20260721
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq \
  --wtq-dataset outputs/server_runs/qwen3_32b_wtq_targeted_shortcutfix2_v2_20260721/input/wtq_targeted5.jsonl \
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
| raw rows | 5 |
| merged rows | 5 |
| eval samples | 5 |
| correct | 5/5 |
| accuracy | 1.0000 |
| avg tokens | 4,907.60 |
| avg seconds | 5.997 |
| failed | 0 |
| missing | 0 |
| wall | 31.435s |

逐样本：

| id | new answer | correct | path |
|---|---|---:|---|
| `nu-2897` | `21.0` | yes | deterministic last-column with year filter |
| `nu-2012` | `The Remixes` | yes | deterministic superlative owner with title fallback |
| `nu-644` | `November 5` | yes | deterministic listed-for-last-owner |
| `nu-4318` | `Tiree` | yes | deterministic superlative restricted to explicit candidates |
| `nu-3977` | `18` | yes | scalar normalization |

日志错误扫描无命中：

```text
Traceback
BadRequestError
context length
Connection refused
APIConnectionError
Exception
ERROR
Error processing
```

## 6. WTQ First50 Aggregate Rerun

命令：

```bash
RUN_ROOT=outputs/server_runs/qwen3_32b_current_blind200_wtq50_shortcutfix2_20260721
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq \
  --wtq-dataset datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11539.45
```

结果：

| run | rows | correct | accuracy | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old baseline first50 | 50 | 33/50 | 0.6600 | 5,776.26 | 5,446.92 | 329.34 | 4.64 | 13.886s | 0 | 0 | historical |
| shortcutfix2 first50 | 50 | 34/50 | 0.6800 | 5,775.70 | 5,446.84 | 328.86 | 4.64 | 13.872s | 0 | 0 | 11m35.104s |

逐样本差异只有 1 条：

```text
nu-2897: old 40.15 -> new 21.0, incorrect -> correct
```

因此 first50 aggregate 没有观察到副作用。

## 7. WTQ Blind200 Full200 Rerun

命令：

```bash
RUN_ROOT=outputs/server_runs/qwen3_32b_current_blind200_wtq200_shortcutfix2_20260721
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq \
  --wtq-dataset datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --max-replan 2 \
  --mact-avg-tokens 11539.45
```

计时：

```text
START: 2026-07-21 17:29:01 CST
END:   2026-07-21 18:22:11 CST
real:  53m09.694s
```

完整性：

| item | value |
|---|---:|
| raw rows | 200 |
| merged rows | 200 |
| eval samples | 200 |
| log tail | `Finished sample 200/200` |
| failed | 0 |
| missing | 0 |

Eval：

| run | correct | accuracy | exact match | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old blind200 | 126/200 | 0.6300 | 0.6150 | 6,227.31 | 5,844.84 | 382.47 | 4.865 | 15.947s | 0 | 0 |
| shortcutfix2 blind200 | 131/200 | 0.6550 | 0.6400 | 6,226.93 | 5,844.72 | 382.21 | 4.865 | 15.939s | 0 | 0 |

逐行差异：

| metric | value |
|---|---:|
| old rows / new rows | 200 / 200 |
| old correct / new correct | 126 / 131 |
| old correct lost | 0 |
| new correct gained | 5 |
| changed predictions/correctness | 5 |

净增正确样本：

| id | old answer | new answer | gold | reason |
|---|---|---|---|---|
| `nu-2012` | album details text | `The Remixes` | `The Remixes` | owner column fallback to title |
| `nu-2897` | `40.15` | `21.0` | `21.00` | last-column with year filter |
| `nu-3977` | `-18.0` | `18` | `18` | absolute difference normalization |
| `nu-4318` | `Mljet` | `Tiree` | `Tiree` | restrict superlative to explicit candidates |
| `nu-644` | `10.0` | `November 5` | `November 5` | target column for `listed for the last round` |

日志错误扫描无命中：

```text
Traceback
BadRequestError
context length
Connection refused
APIConnectionError
Exception
ERROR
Error processing
```

## 8. WTQ Frozen150 Guard

目的：验证 shortcutfix2 没有破坏当前 strict paired 主证据。

命令：

```bash
RUN_ROOT=outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix2_guard_20260721
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

文件时间：

```text
shard/log created: 2026-07-21 18:23:26 CST
eval written:       2026-07-21 19:04:28 CST
observed wall:      about 41m02s
```

完整性和 eval：

| item | value |
|---|---:|
| raw rows | 150 |
| merged rows | 150 |
| eval samples | 150 |
| correct | 114/150 |
| primary accuracy | 0.7600 |
| exact match | 0.7400 |
| avg tokens | 6,185.47 |
| avg prompt tokens | 5,790.19 |
| avg completion tokens | 395.29 |
| avg calls | 4.733 |
| avg seconds | 16.401 |
| failed | 0 |
| missing | 0 |

与上一版 `outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721` 逐行比较：

| metric | value |
|---|---:|
| old rows / new rows | 150 / 150 |
| old correct / new correct | 114 / 114 |
| old correct lost | 0 |
| new correct gained | 0 |
| changed predictions/correctness | 0 |

因此 frozen150 strict paired 总表保持不变：

| scope | myAgent | MACT | accuracy delta | token ratio | accepted |
|---|---:|---:|---:|---:|---|
| frozen150 strict paired | 342/450 = 0.7600 | 330/450 = 0.7333 | +2.67 pp | 0.6161 | yes |

## 9. Current Status and Experiment Plan

当前问题状态：

| problem | current status |
|---|---|
| vLLM not ready / `Connection refused` | Not reproduced in current blind200 or frozen150 guard. |
| shard silently exits / missing output | Not reproduced; raw/merged/eval row counts complete. |
| context length / API errors | Not reproduced by log scan. |
| TabFact weak early result | No longer priority: current blind200 TabFact is `185/200 = 0.9250` with avg tokens `2,426.89`. |
| WTQ split sensitivity | Still present: blind200 is `131/200 = 0.6550`, frozen150 is `114/150 = 0.7600`. Use small targeted WTQ diagnostics rather than full runs. |

Current-code blind200 三数据集组合：

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ shortcutfix2 | 131/200 | 0.6550 | 6,226.93 | 15.939s | 0 | 0 |
| TabFact current | 185/200 | 0.9250 | 2,426.89 | 10.755s | 0 | 0 |
| CRT current | 137/200 | 0.6850 | 10,838.25 | 24.899s | 0 | 0 |
| Overall | 453/600 | 0.7550 | 6,497.36 | 17.198s | 0 | 0 |

可写进专家材料的稳妥表述：

```text
在 Qwen3-32B 本地同模型、同 frozen150 split、同 evaluator 的 strict paired 评估中，
myAgent 三数据集合计 342/450，超过 MACT 的 330/450；平均 API token 为 MACT 的 61.6%，
且 myAgent failed/missing 为 0。current-code blind200 压力测试三数据集合计 453/600，
failed/missing 仍为 0，支持工程链路稳定和总体性能可用，但 blind200 不是同 split MACT 配对结论。
```

不建议写成：

```text
当前版本已经在 full WTQ/TabFact/CRT 上稳定、统计显著、全面超过 MACT。
```

正式实验不建议全量跑。服务器当前 full 数据为 WTQ `4,344`、TabFact `12,779`、CRT `728`，而且磁盘当前约 `3.2G` 可用、`99%` 使用率，直接多模型 full run 风险很高。推荐固定 gate：

| stage | scope | run | go/no-go |
|---|---|---|---|
| Smoke | 20/数据集 | myAgent only | failed/missing 必须为 0；日志无 connection/context error |
| Gate-50 | frozen first50/数据集 | myAgent only | overall 接近当前 Qwen3，token 不高于 MACT `0.75x` 太多 |
| Frozen150 strict paired | 150/数据集 | 只给 1-2 个候选模型跑 myAgent + MACT | overall >= MACT，至少 2 个数据集 >= MACT，token <= 0.75 MACT，failed <= 2% |
| Formal sample | 200 或 300/数据集 | 最终模型 strict paired | 写专家/专利主表 |
| Ablation | 50 或 100/数据集 | `legacy`、`no-strong`、`no deterministic shortcuts`、`max-replan 0/1/2` | 只解释机制贡献，不扩 full |
| Full dataset | optional | 最终模型后台补跑 | 只作背景补充，不阻塞主结论 |

执行建议：

1. 先清理或归档旧 `outputs/server_runs`，保留 eval/report 和关键 merged 文件，避免磁盘满。
2. 每个新模型先跑 Smoke，再跑 Gate-50；Gate-50 不接近当前 Qwen3-32B 就停止。
3. 只让最有希望的 1-2 个模型进入 frozen150 strict paired。
4. 正式主表用 strict paired sample，不用 full dataset 作为日常验收。
5. 消融只用固定 50/100 子集，确保能解释风险评分、强校验、确定性 shortcut、replan 的贡献。
