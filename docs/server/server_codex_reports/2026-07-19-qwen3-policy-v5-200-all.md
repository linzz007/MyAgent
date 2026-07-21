# 2026-07-19 Qwen3 Policy v5 200-Sample All-Dataset Validation

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
代码基线：`33663ab42366000fd94bc884f0f3d200d71bf756` (`Improve TabFact deterministic verification`)
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
任务：当前 myAgent policy v5 在 WTQ / TabFact / CRT 各 200 条同口径验证

> 2026-07-20 update: 本报告中的 MACT 对比是相对已有 Qwen3 MACT 50/数据集参考，不是同 frozen split 的正式配对结论。后续 frozen WTQ100 paired 诊断显示当前 Qwen3 policy v5 不能直接宣称严格超过 MACT；见 `docs/server/server_codex_reports/2026-07-20-qwen3-paired-diagnostic-and-plan.md`。

> 2026-07-21 update: 本文件新增当前分支 HEAD 的 blind200 三数据集验证。当前 HEAD 为 `a1f60c4b865caa06022b852c7da9565a3cdc6de8`，包含用户指定的 `33663ab42366000fd94bc884f0f3d200d71bf756` 之后的 WTQ shortcutfix 和报告提交。若需要“精确 checkout 到 33663ab”的复现实验，应另起 worktree；本节回答的是“当前 myAgent 是否仍符合阶段要求”。

## 0. 2026-07-21 Current Blind200 Rerun

### 0.1 结论

本轮运行链路没有复现之前的工程问题：

| problem | current status | evidence |
|---|---|---|
| vLLM 未就绪 / `Connection refused` | 未复现 | `healthcheck_vllm_pool.sh` 返回 `ok` |
| shard 静默退出或缺输出 | 未复现 | WTQ / TabFact / CRT 的 raw、merged、eval 均完整到 200 条 |
| context length / API 连接异常 | 未复现 | 日志扫描无 `Traceback`、`BadRequestError`、`context length`、`Connection refused`、`APIConnectionError` |
| failed/missing | 未复现 | myAgent 三数据集 failed/missing 均为 0 |

性能判断要分两层：

1. **当前阶段验收的主证据仍是 frozen150 strict paired**：myAgent `342/450 = 0.7600`，MACT `330/450 = 0.7333`；myAgent 平均 token 为 MACT 的 `61.61%`，满足 `compare_blind_results.py` 的 acceptance criteria。
2. **本轮 blind200 只能作为 current-code 泛化压力测试**：myAgent 三数据集合计 `448/600 = 0.7467`，平均 total tokens `6,497.48`，failed/missing 为 0。由于服务器本地没有同一 blind200 split 的 Qwen3 MACT 输出，不能只基于 blind200 写成“严格同口径超过 MACT”。若和已完成 frozen150 MACT 均值粗略比较，blind200 myAgent token 为 MACT 的 `56.31%`，仍明显更低。

风险点：WTQ blind200 为 `126/200 = 0.6300`，低于 frozen150 shortcutfix 的 `114/150 = 0.7600`，说明 WTQ 仍有 split 敏感性。TabFact 和 CRT 当前没有优先级更高的问题：TabFact `185/200 = 0.9250`，CRT `137/200 = 0.6850`。

### 0.2 运行命令

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

export RUN_ROOT=outputs/server_runs/qwen3_32b_current_blind200_20260721
date '+START %F %T %Z'
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl \
  --tabfact-dataset datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl \
  --crt-dataset datasets_ready/blind_holdout_200_v1_2026-06-27/crt.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --max-replan 2 \
  --mact-avg-tokens 11539.45
date '+END %F %T %Z'
```

计时：

```text
START: 2026-07-21 13:26:14 CST
END:   2026-07-21 16:18:20 CST
real:  172m06.107s
```

### 0.3 输出完整性

输出根目录：

```text
outputs/server_runs/qwen3_32b_current_blind200_20260721/
```

| dataset | dataset path | raw rows | merged rows | eval samples | log tail | failed | missing |
|---|---|---:|---:|---:|---|---:|---:|
| WTQ | `datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl` | 200 | 200 | 200 | `Finished sample 200/200` | 0 | 0 |
| TabFact | `datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl` | 200 | 200 | 200 | `Finished sample 200/200` | 0 | 0 |
| CRT | `datasets_ready/blind_holdout_200_v1_2026-06-27/crt.jsonl` | 200 | 200 | 200 | `Finished sample 200/200` | 0 | 0 |

错误扫描命令：

```bash
rg -n "Traceback|BadRequestError|context length|Connection refused|APIConnectionError|Exception|ERROR|Error processing" \
  outputs/server_runs/qwen3_32b_current_blind200_20260721/logs
```

结果：无命中。

### 0.4 Eval、token、耗时

| dataset | correct | primary accuracy | exact match | avg total tokens | avg prompt | avg completion | avg calls | avg elapsed | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 126/200 | 0.6300 | 0.6150 | 6,227.31 | 5,844.84 | 382.47 | 4.865 | 15.947s | 0 | 0 |
| TabFact | 185/200 | 0.9250 | 0.9250 | 2,426.89 | 2,151.77 | 275.12 | 3.685 | 10.755s | 0 | 0 |
| CRT | 137/200 | 0.6850 | 0.6900 | 10,838.25 | 10,252.10 | 586.15 | 5.625 | 24.899s | 0 | 0 |
| Overall | 448/600 | 0.7467 | - | 6,497.48 | 6,082.90 | 414.58 | 4.725 | 17.200s | 0 | 0 |

### 0.5 与 MACT 的当前判断

已有严格配对主表来自：

```text
outputs/server_runs/qwen3_32b_wtq_frozen150_shortcutfix_20260721/compare/compare_all_shortcutfix_vs_mact.json
docs/server/server_codex_reports/2026-07-21-wtq-shortcutfix-frozen150.md
```

| scope | myAgent | MACT | accuracy delta | token ratio | accepted |
|---|---:|---:|---:|---:|---|
| frozen150 strict paired | 342/450 = 0.7600 | 330/450 = 0.7333 | +2.67 pp | 0.6161 | yes |
| current blind200 myAgent only | 448/600 = 0.7467 | no same-split MACT | not strict | 0.5631 vs frozen150 MACT avg | not applicable |

可以写进专家材料的稳妥版本：

```text
在 Qwen3-32B 本地同模型、同 frozen150 split、同 evaluator 的 strict paired 评估中，
myAgent 三数据集合计 342/450，超过 MACT 的 330/450；平均 API token 为 MACT 的 61.6%，
且 myAgent failed/missing 为 0。随后 current-code blind200 压力测试三数据集合计 448/600，
failed/missing 仍为 0，说明当前工程链路稳定，但 WTQ 在不同 split 上仍存在波动。
```

不建议写：

```text
blind200 已经严格证明 myAgent 超过 MACT。
```

因为本地没有 Qwen3 MACT blind200 同 split 结果。

### 0.6 WTQ blind200 诊断

WTQ blind200 当前是主要风险：

| split/run | WTQ accuracy | avg tokens | note |
|---|---:|---:|---|
| current blind200 | 126/200 = 0.6300 | 6,227.31 | 本轮压力测试 |
| shortcutfix frozen150 | 114/150 = 0.7600 | 6,185.50 | 当前 strict paired 主证据 |
| old first200 policy v5 | 134/200 = 0.6700 | 11,317.45 | 旧 first-N run，不是 blind200 |

本轮额外验证了一个预算假设：把 WTQ blind200 first50 的 `--mact-avg-tokens` 从 `11539.45` 提高到 `47439.26` 后，结果没有变化。

| run | scope | correct | accuracy | avg tokens | avg elapsed | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal blind200 first50 | 50 | 33/50 | 0.6600 | 5,776.3 | 13.9s | 0 | 0 |
| high-budget blind200 first50 | 50 | 33/50 | 0.6600 | 5,776.2 | 13.9s | 0 | 0 |

因此当前 WTQ 低分不像是 `mact_avg_tokens` 过低导致的预算截断，更像是 WTQ solver/shortcut/finalization 对 blind split 的泛化问题。已观察到的错误类型包括：

- deterministic shortcut 误触发，例如候选比较题没有限制在题目给出的候选实体内。
- `last` / superlative 题的目标列和 owner 列选择不稳。
- 部分答案可通过 normalization/finalization 挽回，例如实体后缀、时长、绝对差值、表格单元抽取。

这部分不建议直接用大跑解决。下一步应只做 WTQ 小样本或 discordant subset 的根因分析，先证明通用规则有净收益，再重跑 frozen150 或 blind200。

### 0.7 不跑五天的实验方案

服务器上 full 数据规模不适合作为日常迭代：

| dataset | full rows |
|---|---:|
| WTQ | 4,344 |
| TabFact | 12,779 |
| CRT | 728 |

建议固定分阶段 gate：

| stage | scope | 运行内容 | stop rule | 用途 |
|---|---|---|---|---|
| Smoke | 20/数据集 | myAgent only | failed/missing > 0、context error、token 明显异常 | 检查模型服务和工程链路 |
| Gate-50 | frozen first50/数据集 | myAgent only；明显有希望再跑 MACT | overall 明显低于当前 Qwen3 或 token > 0.75 MACT | 快速筛模型 |
| Frozen150 strict paired | 150/数据集 | 只给 1-2 个候选模型跑 myAgent + MACT | overall < MACT、少于两个数据集 >= MACT、token > 0.75、failed > 2% | 阶段验收 |
| Formal sample | 150 或 200/数据集 | 最终模型 strict paired | 作为专家/专利主表 | 正式报告 |
| Ablation | 50 或 100/数据集 | `legacy`、`no-strong`、`no deterministic shortcuts`、`max-replan 0/1/2` | 不扩到 full | 解释机制贡献 |
| Full dataset | optional | 最终模型后台跑 | 不作为常规筛选 | 背景补充，不阻塞主结论 |

推荐后续固定命令模板：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

RUN_ROOT=outputs/server_runs/<model_tag>_gate50_frozen_$(date +%Y%m%d_%H%M%S)
python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11539.45
```

只有 gate50 过线的模型才进入 MACT paired。MACT 单数据集 one-by-one 模板：

```bash
python scripts/server/run_mact_one_by_one.py \
  --mact-root /home/ubuntu/lzz/MACT \
  --task wtq \
  --dataset-path datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --output-path /home/ubuntu/lzz/MACT/outputs/server_runs/<model_tag>_frozen150/wtq_mact.jsonl \
  --log-path /home/ubuntu/lzz/MACT/outputs/server_runs/<model_tag>_frozen150/logs/wtq_mact.log \
  --plan-model-name "$SERVED_MODEL_NAME" \
  --code-model-name "$SERVED_MODEL_NAME" \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_VLLM_API_KEY \
  --thinking disabled \
  --temperature 0 \
  --max-tokens 2048 \
  --max-step 3 \
  --max-actual-step 3 \
  --limit 50
```

TabFact 在 MACT 中使用 `--task scitab`，CRT 使用 `--task crt`。正式配对时去掉 `--limit 50`，并分别跑三个任务；每个输出再用 `code/evaluate_results.py` 和 `code/compare_blind_results.py` 生成同 split 比较表。

### 0.8 当前行动建议

1. 当前项目可以继续作为主线推进：frozen150 strict paired 已过线，current blind200 运行链路稳定。
2. 不建议现在全量跑，也不建议继续把所有精力放在 TabFact。TabFact blind200 已到 `0.9250`，token 很低。
3. 若要继续提升正式把握，优先做 WTQ 小范围根因分析：从 blind200 wrong rows 和 frozen150 discordant rows 中抽 30-50 条，验证 shortcut 误触发、目标列选择、final answer normalization 是否能通用改进。
4. 候选模型筛选只跑 Gate-50。Qwen3-14B-AWQ、Qwen2.5-14B-AWQ、Qwen2.5-3B 已经 no-go；下一个候选模型必须先在 myAgent Gate-50 接近当前 Qwen3-32B，再值得跑 MACT。
5. 正式实验建议用 frozen150 或 frozen200 strict paired 作为主表，消融只跑 50/100，不跑 full。

## 1. 结论

当前 myAgent 在三数据集合计 600 条上的 primary accuracy 为 **429/600 = 71.50%**。相对当前可用的 Qwen3 MACT 50/数据集参考基线 **100/150 = 66.67%**，总体准确率高 **+4.83 个百分点**。

总体 avg total tokens 为 **7,912.67**，是 MACT 参考均值 **11,460.75** 的 **69.04%**，低 **30.96%**。按用户给出的“约 70% MACT token 可接受”尺度看，总体 token 仍明显低于 MACT。

需要保留限制：服务器当前没有同一 200 条样本的 Qwen3 MACT 结果；因此这里的“超过 MACT”是相对现有 Qwen3 MACT 50/数据集参考，不是严格同 200 样本配对结论。

## 2. 运行命令

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_policy_v5_200_all \
  --limit-per-task 200 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

计时：

```text
START: 2026-07-19 14:19:30 CST
END:   2026-07-19 17:25:51 CST
real:  186m20.796s
```

vLLM healthcheck 在实验前后均返回 `ok`，模型名为 `qwen3-32b-local`。

## 3. 输出完整性

输出根目录：

```text
outputs/server_runs/qwen3_32b_policy_v5_200_all/
```

行数核对：

| dataset | raw rows | merged rows | eval num_samples | log tail |
|---|---:|---:|---:|---|
| WTQ | 200 | 200 | 200 | `Finished sample 200/200` |
| TabFact | 200 | 200 | 200 | `Finished sample 200/200` |
| CRT | 200 | 200 | 200 | `Finished sample 200/200` |

日志扫描未检出：

```text
Traceback
Connection refused
APIConnectionError
Error processing
Exception
FAILED/failed
```

日志中存在 pandas `SettingWithCopyWarning`，但未导致 shard 失败或缺行。

## 4. myAgent Eval 结果

| dataset | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 134/200 | 0.670 | 11,317.45 | 10,847.90 | 469.55 | 6.000 | 20.860 | 0 | 0 |
| TabFact | 166/200 | 0.830 | 2,680.65 | 2,348.30 | 332.35 | 3.875 | 12.833 | 0 | 0 |
| CRT | 129/200 | 0.645 | 9,739.92 | 9,218.18 | 521.74 | 5.365 | 22.179 | 0 | 0 |
| Overall | 429/600 | 0.715 | 7,912.67 | 7,471.46 | 441.21 | 5.080 | 18.624 | 0 | 0 |

补充分布：

| dataset | route distribution | risk distribution | strong applied | shortcut |
|---|---|---|---:|---:|
| WTQ | `COMPLEX=178, SIMPLE=22` | `high=179, light=12, medium=8, fallback=1` | 180/200 | 7/200 |
| TabFact | `COMPLEX=197, SIMPLE=3` | `high=163, medium=37` | 0/200 | 15/200 |
| CRT | `COMPLEX=198, SIMPLE=2` | `high=132, medium=64, light=3, fallback=1` | 133/200 | 8/200 |

## 5. MACT Qwen3 参考基线

当前可用 MACT 基线来自：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_50/
```

| dataset | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 23/50 | 0.460 | 10,946.00 | 8,343.76 | 2,602.24 | 3.760 | 97.567 | 1 | 3 |
| TabFact | 44/50 | 0.880 | 11,051.98 | 7,869.24 | 3,182.74 | 3.280 | 118.543 | 0 | 0 |
| CRT | 33/50 | 0.660 | 12,384.26 | 8,330.22 | 4,054.04 | 3.840 | 151.058 | 0 | 0 |
| Overall | 100/150 | 0.667 | 11,460.75 | 8,181.07 | 3,279.67 | 3.627 | 122.389 | 1 | 3 |

## 6. 对比判断

| dataset | accuracy delta | token ratio vs MACT | token delta |
|---|---:|---:|---:|
| WTQ | +21.0 pp | 103.39% | +3.39% |
| TabFact | -5.0 pp | 24.25% | -75.75% |
| CRT | -1.5 pp | 78.65% | -21.35% |
| Overall | +4.83 pp | 69.04% | -30.96% |

判断：

1. 总体准确率相对现有 Qwen3 MACT 50 参考是超过的：71.50% vs 66.67%。
2. 总体 token 仍明显低于 MACT，达到 MACT 参考均值的 69.04%。
3. 分数据集看，优势不均衡：WTQ 准确率明显更高但 token 略高于 MACT；TabFact token 大幅下降但 accuracy 低于 MACT；CRT accuracy 略低于 MACT，token 低约 21.35%。
4. 因为 MACT 缺少同 200 条配对结果，不能写成“同 200 样本稳定全面超过 MACT”。更稳妥表述是：当前 myAgent 在 200/数据集验证上的总体准确率高于现有 Qwen3 MACT 50/数据集参考，同时总体 token 约为 MACT 的 69%。

## 7. 本轮提交范围

本轮只提交本报告文件：

```text
docs/server/server_codex_reports/2026-07-19-qwen3-policy-v5-200-all.md
```

不提交实验输出和既有本地改动：

```text
outputs/server_runs/*
configs/server/qwen3_32b_2gpu_local.env.example
configs/server/qwen3_32b_2gpu_local.env.bak.20260709_191806
restart_qwen3_context_try.sh
```
