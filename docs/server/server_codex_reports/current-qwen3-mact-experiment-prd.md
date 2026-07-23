# 当前 Qwen3 vs MACT 实验 PRD

最后更新：2026-07-23 11:43:41 CST

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
| MACT blind core100 paired | in progress | WTQ 100/100，TabFact 100/100，CRT 71/100 本地进行中 |
| core100 eval/paired/summary | pending | 等 CRT 到 100 后生成 |
| 专家/专利正式实验方案 | pending | 等 core100 结果决定是否扩到 blind200 或改跑新模型 gate |

## 6. 当前 core100 实时状态

截至 2026-07-23 11:43:41 CST：

| dataset | rows | failed | missing | last id | runner |
|---|---:|---:|---:|---|---|
| WTQ | 100/100 | 2 | 2 | `nu-216` | complete |
| TabFact | 100/100 | 0 | 0 | `tabfact-test-6673` | complete |
| CRT | 71/100 | 0 | 0 | `crt-427` | active pid `346671` |

WTQ failed/missing IDs：

```text
nu-4299
nu-2633
```

二者都是 MACT prompt 超过 Qwen3 8192 context 的 BadRequest，不是服务连接错误。

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
| `crt_mact_core100.jsonl` | CRT 71/100 raw，本地继续增长 |
| `logs/wtq_mact_core100.log` | WTQ MACT log |
| `logs/tabfact_mact_core100.log` | TabFact MACT log |
| `logs/crt_mact_core100.log` | CRT MACT log，本地继续增长 |
| `logs/wtq_core100_resume_stdout.log` | WTQ detached stdout |
| `logs/tabfact_core100_resume_stdout.log` | TabFact detached stdout |
| `logs/crt_core100_resume_stdout.log` | CRT detached stdout，本地继续增长 |
| `run_wtq_resume.sh` | WTQ recovery script |
| `run_tabfact_resume.sh` | TabFact recovery script |
| `run_crt_resume.sh` | CRT recovery script |

最新 checkpoint 会继续推送到 MACT `main`。当前准备同步 CRT row71。

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
```

必须带限制：

```text
WTQ 和 TabFact 在 blind50 单项上低于 MACT，overall 优势主要来自 CRT。
blind100 还在跑 CRT，不能提前写 blind100 已完成或已证明总体超过。
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
| P0 | CRT core100 从 71/100 跑到 100/100 | `crt_mact_core100.jsonl`、CRT logs |
| P0 | 每 5-10 行同步 MACT checkpoint | MACT commits on `main` |
| P1 | 生成 core100 per-dataset eval | `*_mact_core100_eval.json` |
| P1 | 生成 same-ID paired compare | `*_mact_core100_paired.json` |
| P1 | 生成 overall summary | `overall_mact_core100_summary.json` |
| P1 | 更新本文档的 core100 结论 | 本 PRD |
| P2 | 判断是否扩到 blind200 | 若 core100 overall 仍超过 MACT 且 token 优势明显 |
| P2 | 新模型筛选 | 只跑 Gate-50/Gate-150，不直接 full |

## 11. 如果服务器清空后的恢复方式

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

4. 如果 CRT 没完成，用已同步的脚本继续：

```bash
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/run_crt_resume.sh
```

脚本使用 `--resume --limit 100`，会从已有 `crt_mact_core100.jsonl` 行数继续，不会重跑已保存样本。
