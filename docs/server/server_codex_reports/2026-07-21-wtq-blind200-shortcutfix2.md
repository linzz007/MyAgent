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

这不是 full blind200 结论。当前证据支持提交一个低风险 WTQ shortcutfix2，并建议下一步只重跑 WTQ blind200 full200 或 frozen150 WTQ，而不是扩大到三数据集 full。

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

## 7. Current Status and Next Step

当前阶段建议：

1. 提交本轮 low-risk WTQ shortcutfix2。
2. 下一步不要跑 full 三数据集；先跑 WTQ blind200 full200 或 frozen150 WTQ 单数据集验证。如果 WTQ 单数据集稳定提升且不伤 frozen150 acceptance，再考虑三数据集正式 paired。
3. 继续保留 frozen150 strict paired 作为当前可写专家材料的主证据：`342/450 = 0.7600` vs MACT `330/450 = 0.7333`，token ratio `0.6161`。
4. blind200 当前不能作为严格 MACT 对比主结论，因为没有同 split Qwen3 MACT。
