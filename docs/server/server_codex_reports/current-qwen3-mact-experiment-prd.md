# 当前 Qwen3 vs MACT 实验 PRD

最后更新：2026-07-23 13:27:07 CST

## 1. 最大目标

验证当前 `myAgent` 在 Qwen3-32B 本地模型下，是否能在 WTQ / TabFact / CRT 三个数据集的同口径评测中总体超过 MACT，并且 token 成本明显低于 MACT；在此基础上形成可写入专家/专利材料的实验结论与正式实验方案。

这个目标不是继续单独优化 TabFact，而是优先判断整体方法是否成立：总体准确率是否超过 MACT、token 是否显著更低、运行链路是否可恢复、结果是否可审计。

## 2. 唯一文档规则

本文档是后续唯一维护的中文 PRD / 进度入口。

历史报告和 run ledger 保留作为证据，不再新增同类“目标进度说明”文档。后续阶段状态、结果文件位置、结论限制和下一步计划都更新到本文档。

## 3. 仓库和同步位置

| repo | path | branch | sync rule | role |
|---|---|---|---|---|
| MyAgent | `/home/ubuntu/lzz/MyAgent` | `codex/selective-risk-collaboration` | 以 GitHub 分支最新提交为准 | PRD、myAgent 输出、评估脚本 |
| MACT | `/home/ubuntu/lzz/MACT` | `main` | 以 GitHub 分支最新提交为准 | MACT raw/log/eval/paired 实验结果 |

关键入口：

```text
PRD:
/home/ubuntu/lzz/MyAgent/docs/server/server_codex_reports/current-qwen3-mact-experiment-prd.md

当前 core100 MACT run:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722

当前 core100 live ledger:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/LIVE_LEDGER.md

当前 full200 MACT run:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723

当前 full200 live ledger:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/LIVE_LEDGER.md
```

## 4. 当前实验口径

| item | value |
|---|---|
| model | Qwen3-32B local |
| served name | `qwen3-32b-local` |
| endpoint | `http://127.0.0.1:8000/v1` |
| GPU | `5,6` |
| max model length | `8192` |
| decoding | `temperature=0`, thinking disabled |
| MACT command | `scripts/server/run_mact_one_by_one.py --limit 100 --resume` |
| paired scope | blind_holdout_200 first 100 rows per dataset |
| key rule | same-ID paired compare only; failed/missing rows must be preserved |

## 5. 子目标拆分和状态

| subgoal | status | evidence |
|---|---|---|
| 排查服务/环境问题 | completed | Qwen3 healthcheck 多次返回 `ok`；无 Connection refused / APIConnectionError |
| 筛选本地模型 | completed | Qwen3-32B 通过；Qwen3-14B-AWQ、Qwen2.5-14B-AWQ、Qwen2.5-3B 未过 gate |
| myAgent blind200 三数据集运行 | completed | WTQ/TabFact/CRT 共 600 条，failed/missing 为 0 |
| MACT blind smoke5 路径验证 | completed | 15 条同 ID smoke 可跑通，用于验证 MACT pipeline |
| MACT blind core50 paired | completed | myAgent `124/150` vs MACT `119/150`，token ratio `0.626` |
| MACT blind core100 paired raw/log | completed | WTQ 100/100，TabFact 100/100，CRT 100/100 |
| core100 eval/paired/summary | completed | overall myAgent `237/300` vs MACT `227/300`，token ratio `0.5913` |
| MACT blind full200 seeded run | in_progress | full200 目录已从 core100 seed，每个数据集 100/200；待补 tail100 |
| 专家/专利正式实验方案 | pending | 基于 core100 结果决定是否扩到 blind200 或改跑新模型 gate |

## 6. 当前 core100 实时状态

截至 2026-07-23 13:11:42 CST：

| dataset | rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 100/100 | 2 | 2 | `nu-216` | complete |
| TabFact | 100/100 | 0 | 0 | `tabfact-test-6673` | complete |
| CRT | 100/100 | 0 | 0 | `crt-620` | complete |

WTQ failed/missing IDs：

```text
nu-4299
nu-2633
```

二者都是 MACT prompt 超过 Qwen3 8192 context 的 BadRequest，不是服务连接错误。

## 6.1 当前 full200 扩样状态

截至 2026-07-23 13:27:07 CST：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723
```

| dataset | rows | status | recovery script |
|---|---:|---|---|
| WTQ | 100/200 | seeded; tail100 pending | `run_wtq_resume.sh` |
| TabFact | 100/200 | seeded; tail100 pending | `run_tabfact_resume.sh` |
| CRT | 100/200 | seeded; tail100 pending | `run_crt_resume.sh` |

full200 的 100 行 seed 来自 core100 raw/log/summary；后续用 `--limit 200 --resume` 只补第 101-200 行，不重跑前 100。

## 7. 已完成结果文件

### 7.1 MACT core100 当前结果

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/
```

已存在文件：

| file | current content |
|---|---|
| `LIVE_LEDGER.md` | core100 实时 ledger |
| `wtq_mact_core100.jsonl` | WTQ 100/100 raw |
| `tabfact_mact_core100.jsonl` | TabFact 100/100 raw |
| `crt_mact_core100.jsonl` | CRT 100/100 raw |
| `logs/wtq_mact_core100.log` | WTQ MACT log |
| `logs/tabfact_mact_core100.log` | TabFact MACT log |
| `logs/crt_mact_core100.log` | CRT MACT log |
| `logs/wtq_core100_resume_stdout.log` | WTQ detached stdout |
| `logs/tabfact_core100_resume_stdout.log` | TabFact detached stdout |
| `logs/crt_core100_resume_stdout.log` | CRT detached stdout |
| `run_wtq_resume.sh` | WTQ recovery script |
| `run_tabfact_resume.sh` | TabFact recovery script |
| `run_crt_resume.sh` | CRT recovery script |
| `*_mact_core100_eval.json` | per-dataset MACT eval |
| `*_mact_core100_errors.jsonl` | per-dataset anomaly rows |
| `*_mact_core100_paired.json` | same-ID myAgent vs MACT paired details |
| `overall_mact_core100_summary.json` | core100 final summary |

core100 raw/log/eval/paired/summary 已同步到 MACT `main`，可作为当前 staged paired 主证据。

core100 final result:

| dataset | myAgent | MACT | delta | token ratio |
|---|---:|---:|---:|---:|
| WTQ | 69/100 | 79/100 | -10 | 0.574 |
| TabFact | 95/100 | 93/100 | +2 | 0.233 |
| CRT | 73/100 | 55/100 | +18 | 0.913 |
| Overall | 237/300 | 227/300 | +10 | 0.591 |

### 7.2 MACT core50 final

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core50_20260722
```

结论：

| dataset | myAgent | MACT | token ratio |
|---|---:|---:|---:|
| WTQ | 34/50 | 41/50 | 0.546 |
| TabFact | 48/50 | 49/50 | 0.234 |
| CRT | 42/50 | 29/50 | 1.019 |
| Overall | 124/150 | 119/150 | 0.626 |

### 7.3 myAgent blind200 outputs

```text
/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen3_32b_current_blind200_wtq200_shortcutfix2_20260721/merged/wtq_qwen3-32b-local.jsonl
/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen3_32b_current_blind200_20260721/merged/tabfact_qwen3-32b-local.jsonl
/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen3_32b_current_blind200_20260721/merged/crt_qwen3-32b-local.jsonl
```

myAgent blind200 stress result：

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ | 131/200 | 0.6550 | 6,226.93 | 15.939s | 0 | 0 |
| TabFact | 185/200 | 0.9250 | 2,426.89 | 10.755s | 0 | 0 |
| CRT | 137/200 | 0.6850 | 10,838.25 | 24.899s | 0 | 0 |
| Overall | 453/600 | 0.7550 | 6,497.36 | 17.198s | 0 | 0 |

### 7.4 多模型 Gate-50 结果位置

这些是已经完成的 myAgent-only Gate-50 筛选。结论是三个非主模型都不进入扩大实验。

| model | output dir | WTQ | TabFact | CRT | overall | avg tokens | decision |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-14B-AWQ | `/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen3_14b_awq_gate50_20260721/` | 37/50 | 44/50 | 27/50 | 108/150 | 7,344.51 | no-go |
| Qwen2.5-14B-AWQ | `/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen25_14b_awq_gate50_20260721/` | 34/50 | 45/50 | 28/50 | 107/150 | 7,308.35 | no-go |
| Qwen2.5-3B-Instruct | `/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen25_3b_current_frozen_gate50_20260720/` | 28/50 | 40/50 | 21/50 | 89/150 | 7,298.51 | no-go |

这些模型没有进入 MACT paired 扩样，因为 Gate-50 已明显低于当前 Qwen3-32B 主模型，也低于 MACT Gate-50 reference。

## 8. 已做优化和修复

| item | status | impact |
|---|---|---|
| 不再继续优先优化 TabFact | done | 目标转为三数据集总体是否超过 MACT |
| WTQ shortcutfix2 输出作为当前 WTQ blind200 myAgent 结果 | done | 保留 current-code WTQ blind200 131/200 结果 |
| vLLM stale pid / live port 检查 | done | 避免 pid stale 时重复启动服务 |
| MACT one-by-one + `--resume` | done | 单条失败不丢整批，服务器中断后可继续 |
| detached resume scripts | done | 防止 Codex 前台 session 断开导致长跑停止 |
| 周期性 Git checkpoint | ongoing | 已按 row55/61/70/80/90/final 等节点推送 |
| context length failure 保留为 failed/missing | done | WTQ 中 MACT 的 `nu-4299`、`nu-2633` 被严格计入失败 |

## 9. 当前可以写的结论

可以写：

```text
在 Qwen3-32B 本地模型、same-ID paired 的 blind50 core 实验中，
myAgent 总体准确率高于 MACT：124/150 vs 119/150，
平均 token 为 MACT 的 62.6%。

在 blind100 core 实验中，myAgent 总体准确率继续高于 MACT：
237/300 vs 227/300，平均 token 为 MACT 的 59.1%。
```

必须带限制：

```text
WTQ 在 blind100 单项上仍低于 MACT：69/100 vs 79/100。
TabFact 小幅超过 MACT：95/100 vs 93/100。
CRT 明显超过 MACT：73/100 vs 55/100。
因此当前可以写“总体超过且 token 明显更低”，不能写“三个数据集全部超过”。
```

不能写：

```text
三个数据集全部超过 MACT。
blind200 strict paired 已完成。
full dataset 已完成。
所有本地模型都超过 MACT。
```

## 10. 下一步小目标

| priority | task | output |
|---:|---|---|
| P0 | 同步 core100 eval/paired/summary final checkpoint | done: MACT `main` |
| P0 | 更新并同步本文档的 core100 结论 | done: 本 PRD |
| P1 | 扩到 blind200 | in progress: full200 目录已 seed，准备补 MACT tail100 |
| P1 | 新模型筛选 | 当前本地 3 个非主模型已 no-go；除非新增/挂载模型，否则不继续跑 |
| P2 | 正式实验方案定稿 | 控制时间成本，避免所有模型 full run |

## 11. 当前决策建议

core100 结果已经满足“总体超过 MACT 且 token 明显更低”的阶段目标：

```text
myAgent: 237/300 = 0.7900
MACT:    227/300 = 0.7567
token ratio: 0.5913
```

但 WTQ 仍显著低于 MACT：

```text
WTQ: myAgent 69/100 vs MACT 79/100
```

因此建议：

1. 专家/专利材料可以先使用 `blind50 + blind100` 作为 staged paired evidence。
2. 不要写“三个数据集全部超过”；只写“总体超过，TabFact/CRT 贡献主要优势，WTQ 仍为短板”。
3. 如果服务器清空前还有稳定时间，当前正在按 seeded full200 方案只对 Qwen3-32B 补 blind200 tail100 MACT，不给 no-go 模型跑 full。
4. 若换新模型，先跑 myAgent-only Gate-50/Gate-150；只有接近或超过 Qwen3-32B 的模型才补 MACT paired。
5. 正式实验建议采用“分阶段抽样 + 最终候选扩样”，不是全模型全数据集暴力跑。

## 12. 如果服务器清空后的恢复方式

1. 重新 clone / pull 两个仓库：

```bash
git clone git@github.com:linzz007/MyAgent.git
git clone git@github.com:linzz007/MACT.git
```

2. 切分支：

```bash
cd MyAgent
git checkout codex/selective-risk-collaboration

cd ../MACT
git checkout main
```

3. 查看本文档和 MACT ledger：

```text
MyAgent/docs/server/server_codex_reports/current-qwen3-mact-experiment-prd.md
MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/LIVE_LEDGER.md
```

4. 如果要继续扩到 blind200，需要新建或恢复对应的 blind200 run 目录，并使用 `--resume --limit 200` 从已保存行数继续。core100 目录中的脚本只用于复现当前 100 行阶段：

```bash
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/run_crt_resume.sh
```

这些脚本使用 `--resume --limit 100`，会从已有 `*_mact_core100.jsonl` 行数继续，不会重跑已保存样本。
