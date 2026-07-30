# 当前 Qwen3 vs MACT 实验 PRD

最后更新：2026-07-30 20:24:40 CST

## 0. 下一次启动先看这里

服务器扩容/清空后，先恢复两个仓库，再看本文档这一节，不要新建第二份 PRD。

当前最后同步状态：

| item | status |
|---|---|
| 已完成并同步的 full200 MACT 数据集 | WTQ `200/200`，TabFact `200/200`，CRT `200/200` |
| 暂停的数据集 | 无；按用户 2026-07-30 最新要求，当前 MyAgent 的 CRT full200 已补跑完成 |
| 当前进程状态 | 本轮 vLLM、`run_sharded_tqa.py`、`code/tqa.py` 均已关停；2026-07-30 20:24 复核无匹配模型/评测进程，`nvidia-smi` compute apps 为空 |
| 下次本地模型服务资源 | 用户 2026-07-30 20:20 再次确认当前服务器卡还够，可使用 GPU `4,5` 和 GPU `6,7` 各启动一个模型服务；默认端口 `8000/8001` |
| 当前本机模型候选 | 2026-07-30 19:53 复扫 `/home/ubuntu/models`、`/home/ubuntu/.cache/huggingface`、`/data`、`/mnt` 后只发现 Qwen3-32B、Qwen3-14B-AWQ、Qwen2.5-14B-AWQ、Qwen2.5-3B-Instruct；除 Qwen3-32B 外均已 Gate-50 no-go |
| 当前外部 API 候选 | 2026-07-30 19:53 环境变量未发现 OpenAI / DeepSeek / DashScope / Anthropic / SiliconFlow / Moonshot / Zhipu / Gemini 可用 key |
| 当前主证据 | core100：myAgent `237/300` vs MACT `227/300`，token ratio `0.5913` |
| full200 阶段证据 | 原 full200：myAgent `453/600` vs MACT `450/600`，token ratio `0.5708`；替换为 2026-07-30 当前 CRT 复跑后：myAgent `456/600` vs MACT `450/600`，token ratio `0.5708` |
| 最新机器审计产物 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_experiment_readiness_audit.json` |
| 最新专家证据摘要 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_expert_evidence_summary.md` |
| 最新恢复就绪审计 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_recovery_readiness_audit.md` |
| canonical myAgent full200 raw artifacts | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_canonical_myagent_full200_raw_artifacts_20260730_2008/` |
| 完整 MyAgent server_runs 恢复包 | `/home/ubuntu/lzz/MACT/outputs/server_runs/myagent_server_runs_archive_20260730_2020/`；含 43 个 run 目录、433 个文件、源目录约 241M，压缩包约 27M |
| 多模型 Gate-50 汇总 | `/home/ubuntu/lzz/MACT/outputs/server_runs/multimodel_gate50_summaries_20260730_1948/` |
| 多模型 Gate-50 raw artifacts | `/home/ubuntu/lzz/MACT/outputs/server_runs/multimodel_gate50_raw_artifacts_20260730_2002/` |
| full200 问题诊断 | 诊断文件、WTQ 50 条 discordant 调试子集、压缩桶、gold 行列覆盖、候选修复收益估计、extreme/only 离线检查和 debug50 实测已保存到 MACT |
| 本地临时文件处理 | 2026-07-30 19:58 已确认 `restart_qwen3_context_try.sh` 和 `configs/server/*.env.bak.*` 是本地临时/备份文件，已加入 `.gitignore`；Qwen3-32B 单服务 example 对齐为 GPU `4,5` |
| 下一步建议 | 结果已校验并关停进程；当前不要重启旧 Qwen3-32B/no-go 模型做重复实验。只有新增/挂载候选模型或提供外部 API key 后，才按第 14 节启动双服务 Gate-10/Gate-50 |

下一次恢复命令入口：

```bash
cd /home/ubuntu/lzz/MyAgent
git checkout codex/selective-risk-collaboration
git pull

cd /home/ubuntu/lzz/MACT
git checkout main
git pull

wc -l /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/*_mact_full200.jsonl
python scripts/server/audit_qwen3_experiment_state.py \
  --myagent-root /home/ubuntu/lzz/MyAgent \
  --mact-root /home/ubuntu/lzz/MACT \
  --model-root /home/ubuntu/models \
  --model-root /home/ubuntu/.cache/huggingface \
  --model-root /data \
  --model-root /mnt \
  --output /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_experiment_readiness_audit.json \
  --markdown-output /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_expert_evidence_summary.md
```

当前 full200 已完成；恢复后优先复核这些结果文件，不要再启动 `run_crt_resume.sh`：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/crt_mact_full200_eval.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/crt_mact_full200_paired.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/overall_mact_full200_summary.json
```

不要重跑 WTQ/TabFact，除非明确创建新的 run 目录和新的实验口径。

新增模型时，所有新实验产物默认写到 MACT：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/<model_tag>_gate50_<date>/
```

新增本地 vLLM 模型时，默认按当前服务器资源开两个服务：GPU `4,5` 绑定端口 `8000`，GPU `6,7` 绑定端口 `8001`；Gate-10 / Gate-50 直接传两个 endpoint 并行跑。

MyAgent 仓库只负责代码、脚本和本文档；除非临时调试，不再把新实验主结果分散写到 MyAgent 的 `outputs/server_runs/`。

当前可复核审计脚本：

```text
/home/ubuntu/lzz/MyAgent/scripts/server/audit_qwen3_experiment_state.py
/home/ubuntu/lzz/MyAgent/scripts/server/prepare_model_gate_run.py
/home/ubuntu/lzz/MyAgent/scripts/server/summarize_model_gate_results.py
```

作用：`audit_qwen3_experiment_state.py` 从 MACT 已保存结果生成机器可读 JSON 和中文专家证据摘要，快速回答“证据是否完整、总体/token 阶段条件是否达成、是否有新候选值得启动 Gate-10/Gate-50”。`prepare_model_gate_run.py` 在新增本地模型后自动生成 MACT run 目录、双服务 vLLM env、Gate-10/Gate-50 runner 和 manifest，但不启动服务。`summarize_model_gate_results.py` 读取 Gate-50 三个 eval JSON，输出 `gate50_summary.json/md` 和 no-go/Gate-150 决策。

## 1. 最大目标

验证当前 `myAgent` 在 Qwen3-32B 本地模型下，是否能在 WTQ / TabFact / CRT 三个数据集的同口径评测中总体超过 MACT，并且 token 成本明显低于 MACT；在此基础上形成可写入专家/专利材料的实验结论与正式实验方案。

这个目标不是继续单独优化 TabFact，而是优先判断整体方法是否成立：总体准确率是否超过 MACT、token 是否显著更低、运行链路是否可恢复、结果是否可审计。

## 1.1 当前阶段验收判断

| question | current answer |
|---|---|
| 总体准确率是否超过 MACT | 是。canonical full200 为 `453/600` vs `450/600`；替换当前 CRT 复跑后 staged composite 为 `456/600` vs `450/600` |
| token 是否仍明显低于 MACT | 是。full200 token ratio 为 `0.5708`，约为 MACT 的 `57.1%` |
| 三个数据集是否都超过 MACT | 否。WTQ 和 TabFact 在 full200 单项仍低于 MACT，优势主要来自 CRT |
| 当前项目是否可作为阶段证据 | 可以作为 staged evidence；不能写成 full dataset 全量完成或全面显著胜出 |
| 现在是否继续跑旧本地模型 | 不建议。现有非主模型均已 Gate-50 no-go；下一轮等待新增模型或可用外部 API key |
| 下一步实验策略 | 使用 Gate-10 / Gate-50 / Gate-150 / Paired-200 漏斗，只扩大最终候选，避免全模型全数据集暴力跑 |

## 2. 唯一文档规则

本文档是后续唯一维护的中文 PRD / 进度入口。

历史报告和 run ledger 保留作为证据，不再新增同类“目标进度说明”文档。后续阶段状态、结果文件位置、结论限制和下一步计划都更新到本文档。

本文档负责回答五个问题：

1. 最大目标是什么。
2. 拆成哪些小目标，每个小目标是否完成。
3. 结果文件保存在哪里，哪些已经同步到 GitHub。
4. 做过哪些优化或流程修复。
5. 接下来继续跑什么、停止跑什么、怎样避免一次实验跑五天。

MACT run 目录里的 `LIVE_LEDGER.md` 只作为运行证据账本存在，不替代本文档，也不再新增第二份 PRD。

## 3. 仓库和同步位置

| repo | path | branch | sync rule | role |
|---|---|---|---|---|
| MyAgent | `/home/ubuntu/lzz/MyAgent` | `codex/selective-risk-collaboration` | 以 GitHub 分支最新提交为准 | PRD、评估脚本、历史 myAgent 输出 |
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

WTQ extreme/only 修复 debug50 实测 run:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_extreme_fix_debug50_20260730_173740

WTQ extreme/only 修复代表性 WTQ100 回归 run:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_extreme_fix_representative100_20260730_1805

当前 MyAgent CRT full200 复跑 run:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_crt_full200_current_20260730_1822
```

## 4. 当前实验口径

| item | value |
|---|---|
| model | Qwen3-32B local |
| served name | `qwen3-32b-local` |
| endpoint | 默认 `http://127.0.0.1:8000/v1`；CRT tail 并行 shard 使用 `8000/8001` |
| GPU | 默认 `5,6`；CRT tail 并行 shard 使用 `4,5;6,7` |
| max model length | `8192` |
| decoding | `temperature=0`, thinking disabled |
| MACT command | core100 使用 `--limit 100 --resume`；full200 使用 `--limit 200 --resume`，CRT tail 后 80 条用两个独立 shard 输出后按 ID 合并 |
| paired scope | core100 三数据集已完成；full200 WTQ/TabFact/CRT 三数据集已完成 |
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
| MACT blind full200 seeded run | completed | full200 目录已完成 WTQ/TabFact/CRT 各 `200/200` raw/eval/paired；overall myAgent `453/600` vs MACT `450/600` |
| full200 分歧诊断 | completed | `full200_disagreement_diagnostics.md/json` 已保存到 MACT full200 run 目录；WTQ net `-17`，TabFact net `-4`，CRT net `+24` |
| WTQ discordant 调试子集 | completed | `wtq_discordant_debug_subset_50.jsonl/md` 已保存到 MACT；40 条 `mact_only` + 10 条优先 `neither` |
| WTQ 压缩桶诊断 | completed | `wtq_compression_bucket_diagnostics.md/json` 已保存到 MACT；MACT-only 中位压缩比例 `0.25`，有 7 条 not-found-like、3 条 header-prediction 信号 |
| WTQ gold 行列覆盖诊断 | completed | `wtq_gold_rowcol_loss_diagnostics.md/json` 已保存到 MACT；50 条中 22 条 gold cell 已保留，19 条 literal-gold 被行/列压缩丢失，9 条为非字面计数/计算 |
| WTQ 候选修复收益估计 | completed | `wtq_fix_candidate_coverage_estimate.md/json` 已保存到 MACT；优先验证 WTQ extreme/only 全局行策略，其次验证行匹配扫描全行 |
| WTQ extreme/only 全局行最小修复 | completed measured debug50 | MyAgent 已加入 `only/top/first/last/earliest/latest` global-row 触发词；debug50 从旧 myAgent `0/50` 提升到新 myAgent `14/50`，18 条新触发全行里 `10/18` 正确，10 条 strict recoverable 里 `7/10` 正确 |
| WTQ extreme/only 代表性 WTQ100 回归 | completed measured | 新 myAgent `69/100`，旧 myAgent `69/100`，MACT `79/100`；new/MACT token ratio `0.5790`，new/old token ratio `1.0091`；恢复 3 条、回退 3 条，无净提升 |
| 当前 MyAgent CRT full200 复跑 | completed measured | 新 myAgent `140/200`，旧 myAgent `137/200`，MACT `113/200`；new/MACT token ratio `0.8461`，new/old token ratio `1.0001`；failed/missing 为 0 |
| numpy array runtime 边界修复 | completed | debug50 的 `nu-4299` 暴露 `verification_gap` 和 JSON 序列化对 numpy array 的崩溃；已加单测并修复 |
| 当前本机模型候选盘点 | completed | 仅发现 4 个本地模型目录；3 个非主模型已 no-go；未发现可直接使用的 DeepSeek/OpenAI/DashScope API key |
| 2026-07-30 19:23 继续执行审计 | completed | 两仓库已同步到远端；GPU 0-7 空闲；无 vLLM/runner 进程；常见模型目录和缓存未发现新候选；环境变量未发现可用外部 API key |
| 多模型 Gate-50 汇总入 MACT | completed | Qwen3-14B-AWQ、Qwen2.5-14B-AWQ、Qwen2.5-3B 的统一 no-go summary 已保存到 MACT `multimodel_gate50_summaries_20260730_1948` |
| 2026-07-30 19:53 继续执行审计 | completed | 复核 GPU/进程/模型/API key；GPU 空闲但无新增候选，审计 JSON 已刷新到 MACT full200 run，未启动重复 Gate |
| 本地未同步文件归类 | completed | `restart_qwen3_context_try.sh` 是早期 context-length 试启动脚本，当前由 `prepare_model_gate_run.py` 和 run-specific `vllm.env` 覆盖；`.env.bak.*` 是备份文件，二者已被 ignore，避免误提交或恢复时误用 |
| 多模型 Gate-50 raw artifacts 迁移到 MACT | completed | 三个历史 myAgent-only Gate-50 run 的 `raw/merged/eval/compare/shards/logs` 已复制到 MACT `multimodel_gate50_raw_artifacts_20260730_2002`；共 59 个源文件、约 20 MB，便于服务器清空后恢复审计 |
| canonical myAgent full200 raw artifacts 迁移到 MACT | completed | canonical full200 的 myAgent 源不是单一目录：WTQ 使用 `qwen3_32b_current_blind200_wtq200_shortcutfix2_20260721`，TabFact/CRT 使用 `qwen3_32b_current_blind200_20260721`；三项 raw/merged/eval/shards/logs 已复制到 MACT `qwen3_32b_canonical_myagent_full200_raw_artifacts_20260730_2008` |
| 完整 MyAgent server_runs 归档到 MACT | completed | `/home/ubuntu/lzz/MyAgent/outputs/server_runs` 已压缩到 MACT `myagent_server_runs_archive_20260730_2020`；覆盖 43 个 run 目录、433 个文件、源目录 241M、压缩包约 27M，归档前敏感信息扫描无命中 |
| 恢复就绪审计 | completed | `latest_recovery_readiness_audit.md` 已保存到 MACT full200 run；确认关键 PRD、summary、paired、diagnostics、Gate summaries/raw artifacts 均可从 Git 恢复，full200 本地 extra 仅为 tmp/pid |
| 专家/专利正式实验方案 | ready for drafting | full200 总体略超 MACT 且 token 显著更低，但 dataset-level 只有 CRT 超过；正式实验仍建议 gate 后只扩最终候选 |

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

截至 2026-07-30 16:50:48 CST：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723
```

| dataset | rows | status | recovery script |
|---|---:|---|---|
| WTQ | 200/200 | complete; MACT `148/200`, myAgent `131/200`, token ratio `0.5926`; 5 context overflow failures | `run_wtq_resume.sh` |
| TabFact | 200/200 | complete; MACT `189/200`, myAgent `185/200`, token ratio `0.2241`; 0 failures | `run_tabfact_resume.sh` |
| CRT | 200/200 | original complete: MACT `113/200`, old myAgent `137/200`, token ratio `0.8461`; 2026-07-30 current myAgent rerun: `140/200`, token ratio `0.8461`, failed/missing `0/0` | MACT baseline: `run_crt_resume.sh`; current myAgent rerun: `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_crt_full200_current_20260730_1822` |

full200 的 100 行 seed 来自 core100 raw/log/summary；后续用 `--limit 200 --resume` 只补第 101-200 行，不重跑前 100。

CRT tail 本次恢复过程：

```text
START_CRT_FULL200 2026-07-30 13:55:20 CST
single service: GPU 5,6 on port 8000
single runner reached 120/200; last id crt-187
user allowed two services on GPU 4,5 and 6,7
two-service profile: qwen3_32b_4gpu_2svc.env
shard 121-160: run_crt_shard_121_160.sh, port 8000, 40/40 complete, END 2026-07-30 16:38:30 CST
shard 161-200: run_crt_shard_161_200.sh, port 8001, 40/40 complete, END 2026-07-30 16:44:11 CST
merged canonical: crt_mact_full200.jsonl, 200/200, id order verified against blind_holdout_200 CRT
backup prefix: crt_mact_full200_prefix120_before_shard_merge.jsonl
```

WTQ tail100 已在主机级后台启动：

```text
START_WTQ_FULL200 2026-07-23 13:33:25 CST
runner: run_wtq_resume.sh
stdout log: logs/wtq_full200_resume_stdout.log
first tail row: 101/200, id=nu-4099, ok, 8735 tokens, 84.3s
latest checkpoint: 104/200, id=nu-3939, ok, 16520 tokens, 173.7s
latest checkpoint: 109/200, id=nu-712, ok, 7551 tokens, 98.8s
latest checkpoint: 116/200, id=nu-2903, ok, 11749 tokens, 132.6s
latest checkpoint: 120/200, id=nu-3399, ok, 8017 tokens, 102.6s
latest checkpoint: 124/200, id=nu-3290, failed, context overflow 6145+2048 > 8192
latest checkpoint: 125/200, id=nu-3922, ok, 8291 tokens, 116.7s
latest checkpoint: 126/200, id=nu-3527, ok, 7678 tokens, 76.6s
latest checkpoint: 131/200, id=nu-1298, ok, 6794 tokens, 67.1s
latest checkpoint: 133/200, id=nu-3139, failed, context overflow 6145+2048 > 8192
latest checkpoint: 138/200, id=nu-824, ok, 19077 tokens, 307.0s
latest checkpoint: 144/200, id=nu-976, ok, 11989 tokens, 130.1s
latest checkpoint: 145/200, id=nu-2426, ok, 10617 tokens, 111.2s
latest checkpoint: 150/200, id=nu-4291, ok, 7621 tokens, 58.9s
latest checkpoint: 151/200, id=nu-2973, ok, 8857 tokens, 100.5s
latest checkpoint: 152/200, id=nu-1313, ok, 11224 tokens, 98.6s
latest checkpoint: 153/200, id=nu-3488, ok, 5876 tokens, 43.4s
latest checkpoint: 154/200, id=nu-1030, ok, 28688 tokens, 286.6s
latest checkpoint: 155/200, id=nu-3027, ok, 8149 tokens, 77.4s
latest checkpoint: 156/200, id=nu-3487, failed, context overflow 6145+2048 > 8192
latest checkpoint: 157/200, id=nu-1424, ok, 12536 tokens, 132.3s
latest checkpoint: 158/200, id=nu-4318, ok, 12200 tokens, 43.5s
latest checkpoint: 159/200, id=nu-2547, ok, 6919 tokens, 54.6s
latest checkpoint: 160/200, id=nu-2981, ok, 6861 tokens, 56.8s
latest checkpoint: 161/200, id=nu-2032, ok, 11062 tokens, 191.8s
latest checkpoint: 162/200, id=nu-1761, ok, 9027 tokens, 124.9s
latest checkpoint: 163/200, id=nu-3074, ok, 6026 tokens, 53.1s
latest checkpoint: 164/200, id=nu-2506, ok, 5781 tokens, 44.1s
latest checkpoint: 165/200, id=nu-65, ok, 15989 tokens, 114.2s
latest checkpoint: 166/200, id=nu-1446, ok, 21356 tokens, 284.3s
latest checkpoint: 167/200, id=nu-1246, ok, 8842 tokens, 94.9s
latest checkpoint: 168/200, id=nu-267, ok, 7529 tokens, 87.4s
latest checkpoint: 169/200, id=nu-4092, ok, 6851 tokens, 73.9s
latest checkpoint: 170/200, id=nu-107, ok, 7174 tokens, 48.0s
latest checkpoint: 171/200, id=nu-543, ok, 9311 tokens, 102.5s
latest checkpoint: 172/200, id=nu-3982, ok, 7084 tokens, 97.7s
latest checkpoint: 173/200, id=nu-721, ok, 8239 tokens, 97.7s
latest checkpoint: 174/200, id=nu-4162, ok, 15162 tokens, 286.9s
latest checkpoint: 175/200, id=nu-641, ok, 16995 tokens, 202.7s
latest checkpoint: 176/200, id=nu-1615, ok, 6561 tokens, 68.9s
latest checkpoint: 177/200, id=nu-1845, ok, 8050 tokens, 86.1s
latest checkpoint: 178/200, id=nu-888, ok, 2259 tokens, 21.3s
latest checkpoint: 179/200, id=nu-1498, ok, 17523 tokens, 220.9s
latest checkpoint: 180/200, id=nu-53, ok, 10299 tokens, 114.9s
latest checkpoint: 181/200, id=nu-2144, ok, 10515 tokens, 108.9s
latest checkpoint: 182/200, id=nu-717, ok, 5679 tokens, 28.6s
latest checkpoint: 183/200, id=nu-1970, ok, 7539 tokens, 93.1s
latest checkpoint: 184/200, id=nu-1915, ok, 8304 tokens, 106.3s
latest checkpoint: 185/200, id=nu-2183, ok, 7935 tokens, 100.8s
latest checkpoint: 186/200, id=nu-889, ok, 2344 tokens, 14.9s
latest checkpoint: 187/200, id=nu-3891, ok, 8200 tokens, 103.5s
latest checkpoint: 188/200, id=nu-996, ok, 15058 tokens, 117.3s
latest checkpoint: 189/200, id=nu-1317, ok, 6105 tokens, 44.1s
latest checkpoint: 190/200, id=nu-1421, ok, 9670 tokens, 66.2s
latest checkpoint: 191/200, id=nu-2934, ok, 7751 tokens, 57.6s
latest checkpoint: 192/200, id=nu-207, ok, 20731 tokens, 285.9s
latest checkpoint: 193/200, id=nu-1686, ok, 6364 tokens, 47.0s
latest checkpoint: 194/200, id=nu-2172, ok, 9746 tokens, 62.2s
latest checkpoint: 195/200, id=nu-2483, ok, 9850 tokens, 99.2s
latest checkpoint: 196/200, id=nu-2120, ok, 16805 tokens, 192.2s
latest checkpoint: 197/200, id=nu-1657, ok, 6708 tokens, 56.9s
latest checkpoint: 198/200, id=nu-983, ok, 23692 tokens, 138.0s
latest checkpoint: 199/200, id=nu-3200, ok, 6841 tokens, 72.4s
latest checkpoint: 200/200, id=nu-2332, ok, 10496 tokens, 78.9s
END_WTQ_FULL200 2026-07-23 16:45:33 CST
```

TabFact tail100 已在主机级后台启动：

```text
START_TABFACT_FULL200 2026-07-23 19:01:42 CST
runner: run_tabfact_resume.sh
stdout log: logs/tabfact_full200_resume_stdout.log
latest checkpoint: 116/200, id=tabfact-test-12699, ok, 9102 tokens, 77.5s
latest checkpoint: 117/200, id=tabfact-test-1988, ok, 10138 tokens, 119.1s
latest checkpoint: 118/200, id=tabfact-test-3835, ok, 9649 tokens, 95.4s
latest checkpoint: 119/200, id=tabfact-test-5093, ok, 13539 tokens, 168.0s
latest checkpoint: 120/200, id=tabfact-test-12773, ok, 8836 tokens, 57.1s
latest checkpoint: 160/200, id=tabfact-test-10808, ok, 16011 tokens, 138.5s
latest checkpoint: 180/200, id=tabfact-test-5916, ok, 9359 tokens, 98.5s
latest checkpoint: 200/200, id=tabfact-test-6414, ok, 18308 tokens, 254.0s
END_TABFACT_FULL200 2026-07-23 21:55:21 CST
```

错误扫描截至本次更新看到五个 WTQ context length BadRequest：

```text
nu-4299
nu-2633
nu-3290
nu-3139
nu-3487
```

没有新的 Connection refused / APIConnectionError。

`nu-3290`、`nu-3139` 和 `nu-3487` 是 full200 tail 中新增的 MACT context overflow。原因和前两个一致：MACT 请求 `max_tokens=2048` 时，vLLM 只允许 input <= 6144，但这些样本 input 为 6145。runner 已将失败样本保留为一行 `exec_error`，并继续运行。后续如果要提升 MACT baseline 严谨性，应设计单独的 failed-row repair 步骤，例如用 `max_tokens=2047` 重跑这些失败 ID，并在 paired 结果中明确标注 repaired 口径；不能静默覆盖。

WTQ full200 same-ID paired 结果：

| item | value |
|---|---:|
| matched rows | 200/200 |
| myAgent correct | 131/200 |
| MACT correct | 148/200 |
| token ratio | 0.5926 |
| paired both correct | 108 |
| myAgent only | 23 |
| MACT only | 40 |
| neither | 29 |

TabFact full200 same-ID paired 结果：

| item | value |
|---|---:|
| matched rows | 200/200 |
| myAgent correct | 185/200 |
| MACT correct | 189/200 |
| token ratio | 0.2241 |
| paired both correct | 178 |
| myAgent only | 7 |
| MACT only | 11 |
| neither | 4 |

full200 已完成 WTQ+TabFact 两项的阶段合计：

| item | value |
|---|---:|
| matched rows | 400/400 |
| myAgent correct | 316/400 |
| MACT correct | 337/400 |
| myAgent accuracy | 0.7900 |
| MACT accuracy | 0.8425 |
| token ratio | 0.4055 |

CRT full200 same-ID paired 结果：

| item | value |
|---|---:|
| matched rows | 200/200 |
| myAgent correct | 137/200 |
| MACT correct | 113/200 |
| token ratio | 0.8461 |
| paired both correct | 101 |
| myAgent only | 36 |
| MACT only | 12 |
| neither | 51 |

full200 三数据集最终合计：

| item | value |
|---|---:|
| matched rows | 600/600 |
| myAgent correct | 453/600 |
| MACT correct | 450/600 |
| myAgent accuracy | 0.7550 |
| MACT accuracy | 0.7500 |
| token ratio | 0.5708 |

这个阶段结果确认：full200 三数据集总体 myAgent 略高于 MACT，平均 token 仍明显更低。但 dataset-level 只有 CRT 超过，WTQ 和 TabFact 仍低于 MACT；因此不能写“三个数据集全部超过”，也不能把 full200 结论写成强显著全面胜出。

## 6.2 full200 分歧诊断

诊断文件：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/full200_disagreement_diagnostics.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/full200_disagreement_diagnostics.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_discordant_debug_subset_50.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_discordant_debug_subset_50.jsonl
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_compression_bucket_diagnostics.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_compression_bucket_diagnostics.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_gold_rowcol_loss_diagnostics.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_gold_rowcol_loss_diagnostics.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_compressor_hypothesis_diagnostics.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_compressor_hypothesis_diagnostics.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_fix_candidate_coverage_estimate.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_fix_candidate_coverage_estimate.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_extreme_only_global_rows_offline_check.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/wtq_extreme_only_global_rows_offline_check.json
```

same-ID paired 分歧净贡献：

| dataset | both_correct | myAgent-only | MACT-only | neither | net |
|---|---:|---:|---:|---:|---:|
| WTQ | 108 | 23 | 40 | 29 | -17 |
| TabFact | 178 | 7 | 11 | 4 | -4 |
| CRT | 101 | 36 | 12 | 51 | +24 |
| Overall | 387 | 66 | 63 | 84 | +3 |

问题定位：

1. WTQ 是 full200 没有通过 dataset-level acceptance 的主要负贡献，MACT-only 40 行集中在 temporal、count、superlative、negation_logic 标签；后续若做算法修复，应优先做 WTQ discordant subset，而不是继续 TabFact。
2. TabFact full200 只净落后 4 行，且没有 MACT execution failure；当前不值得再优先局部优化。
3. CRT 是主要正贡献，myAgent-only 36 行、MACT-only 12 行；说明 selective-risk pipeline 的强项主要体现在 CRT 复杂比较/闭集问答。
4. MACT 的 5 个 WTQ context overflow 已严格保留为失败行；如果做 repaired baseline，必须新建口径，不能覆盖 canonical full200。
5. WTQ MACT-only 行的压缩比例中位数为 `0.25`，低于 both-correct 的 `0.375`；同时出现 7 条 not-found-like prediction 和 3 条 header-prediction。当前合理假设是检索/压缩后的信息定位不足叠加计数、时间边界错误，需要用 WTQ 子集继续验证，不能直接大改。
6. WTQ 50 条 debug 子集的 gold 行列覆盖结果：22 条 gold cell 已在压缩表内但仍答错，19 条 literal-gold 被压缩丢失，9 条是计数/计算答案不适合字面覆盖判断。MACT-only 的 literal-gold 丢失主要是行丢失：12 条 `gold_col_kept_row_dropped`，4 条 `gold_row_kept_col_dropped`。
7. 候选修复收益估计显示：`global_rows_for_wtq_extreme_or_only` 可覆盖 10 个 literal-gold 行丢失样本，其中 9 个是 MACT-only；`preserve_more_answer_columns_for_implicit_answer` 只覆盖 3 个样本。因此如果改代码，优先小范围验证 WTQ extreme/only 全局行策略，而不是先做大范围列保留。
8. 已完成第一步最小代码实验：`TableCompressor._needs_global_rows` 新增 `only/top/first/last/earliest/latest` 触发词。新增单测先失败后通过；离线检查显示 50 条 WTQ debug subset 中 18 条新触发全行，10 条 strict literal gold row-loss case 可恢复。该结果只证明压缩覆盖改善，不等价于模型准确率提升。
9. 已完成 debug50 模型实测：修复后 myAgent 在该 adversarial subset 上 `14/50`，旧 myAgent 为 `0/50`，MACT 为 `40/50`。新触发全行的 18 条里 `10/18` 正确，strict recoverable 的 10 条里 `7/10` 正确；平均 token 为 MACT 的 `0.6009`，相对旧 myAgent 增加约 `2.1%`。

当前问题排查结论：

1. 不要再用 debug50 代表总体准确率；它是从旧 myAgent 错例里抽出的 adversarial subset。
2. 代表性 WTQ100 回归已完成：新 myAgent `69/100`，旧 myAgent `69/100`，MACT `79/100`；恢复 3 条、回退 3 条，无净提升。
3. 因此不应把 WTQ extreme/only 全局行策略继续扩大成主线优化；它可保留为小范围修复，但不是当前总体提升来源。
4. 若用户明确继续投入 WTQ，下一步才考虑 `_match_rows` 从只扫前 3 个单元扩展为低噪声全行 token 扫描，并先跑小 gate；不能直接重跑 full200。
5. 只接受能跨样本解释问题的通用修复，不接受按 ID/table 硬编码；正式回归必须在小切片收益明确后再跑。

## 6.3 WTQ extreme/only debug50 实测

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_extreme_fix_debug50_20260730_173740
```

| metric | value |
|---|---:|
| rows | 50 |
| old myAgent correct | 0 |
| new myAgent correct | 14 |
| MACT correct | 40 |
| new exec failures | 0 |
| new missing answers | 0 |
| new avg total tokens | 6857.72 |
| MACT avg total tokens | 11412.10 |
| token ratio new myAgent / MACT | 0.6009 |
| token ratio new / old myAgent | 1.0213 |
| new avg elapsed seconds | 16.8523 |

细分结果：

| scope | rows | new myAgent correct | MACT correct | note |
|---|---:|---:|---:|---|
| newly global-triggered | 18 | 10 | 15 | 新触发 `only/top/first/last/earliest/latest` 全局行 |
| strict recoverable offline | 10 | 7 | 9 | 离线判断为 gold 列已保留、仅 gold 行丢失 |
| mact_only bucket | 40 | 14 | 40 | 该 bucket 本来就是旧 MACT-only |
| neither bucket | 10 | 0 | 0 | 双方旧结果都错，当前修复未改善 |

这次实测还暴露并修复了两个 runtime 边界问题：

1. `selective_collaboration.verification_gap` 遇到 numpy array execution result 时，不能用 `not in (None, "", [])` 做非空判断。
2. `tqa._to_serializable` 和 `_json_default` 遇到多元素 numpy array 时，不能直接 `.item()`，需要优先 `.tolist()`。

本节结论：extreme/only 全局行修复对 WTQ 目标错例有实际收益，但仍不足以在该 adversarial subset 上超过 MACT。后续代表性 WTQ100 回归已经证明该修复没有净总体收益，因此暂不把 WTQ 单点优化作为下一阶段主方向。

## 6.4 当前执行流程

当前流程按“先小样本判方向，再只给候选方案补 paired”的原则执行：

1. 服务健康检查：先确认 Qwen3-32B vLLM endpoint 可用，避免把服务失败误记为算法失败。
2. myAgent-only 压测：先跑 WTQ / TabFact / CRT 的 200 条，记录 eval、merged 行数、token、耗时、失败数。
3. MACT smoke：先用同 ID smoke5 验证 MACT pipeline、任务映射和输出格式。
4. MACT staged paired：先跑 core50，再跑 core100；只有 core100 总体超过且 token 明显更低，才扩到 full200。
5. full200 补样：复用 core100 seed，用 `--resume --limit 200` 只补每个数据集后 100 条。
6. 周期 checkpoint：raw、log、stdout、ledger 写入 MACT 并推送；PRD 写入 MyAgent 并推送。
7. 正式实验：不做全模型全数据集暴力跑，采用 gate 筛选后只扩最终候选。

## 7. 已完成结果文件

### 7.0 当前 full200 扩样结果

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/
```

当前已保存文件：

| file | current content |
|---|---|
| `LIVE_LEDGER.md` | full200 实时 ledger |
| `wtq_mact_full200.jsonl` | WTQ 200/200 raw，5 个 context overflow failure，runner complete |
| `wtq_mact_full200_eval.json` | WTQ MACT full200 eval：MACT `148/200 = 0.7400` |
| `wtq_mact_full200_errors.jsonl` | WTQ eval anomaly rows，52 行 |
| `wtq_mact_full200_paired.json` | WTQ full200 same-ID paired：myAgent `131/200` vs MACT `148/200` |
| `tabfact_mact_full200.jsonl` | TabFact 200/200 raw，0 failure，runner complete |
| `tabfact_mact_full200_eval.json` | TabFact MACT full200 eval：MACT `189/200 = 0.9450` |
| `tabfact_mact_full200_errors.jsonl` | TabFact eval anomaly rows，11 行 |
| `tabfact_mact_full200_paired.json` | TabFact full200 same-ID paired：myAgent `185/200` vs MACT `189/200` |
| `logs/tabfact_full200_resume_stdout.log` | TabFact detached stdout，19:01:42 CST 启动，21:55:21 CST complete |
| `crt_mact_full200.jsonl` | CRT 200/200 canonical raw，0 failure，ID 顺序已按 blind200 原始 dataset 校验 |
| `crt_mact_full200_prefix120_before_shard_merge.jsonl` | CRT 单 runner 到 120/200 后的 merge 前备份 |
| `crt_mact_full200_shard_121_160.jsonl` | CRT shard 121-160 raw，40/40 complete |
| `crt_mact_full200_shard_161_200.jsonl` | CRT shard 161-200 raw，40/40 complete |
| `crt_mact_full200_eval.json` | CRT MACT full200 eval：MACT `113/200 = 0.5650` |
| `crt_mact_full200_errors.jsonl` | CRT eval anomaly rows，87 行，均为 EM mismatch，不是 exec failure |
| `crt_mact_full200_paired.json` | CRT full200 same-ID paired：myAgent `137/200` vs MACT `113/200` |
| `overall_mact_full200_summary.json` | WTQ/TabFact/CRT full200 final summary：myAgent `453/600` vs MACT `450/600`，token ratio `0.5708` |
| `overall_mact_full200_summary.stdout.json` | 生成 overall 时保留的 stdout 镜像 |
| `latest_experiment_readiness_audit.json` | 由 MyAgent 审计脚本生成的机器可读状态：evidence complete、canonical full200/staged composite 指标、新模型 readiness |
| `latest_expert_evidence_summary.md` | 由同一审计脚本生成的中文专家/专利阶段证据摘要，含可写结论、限制和下一步 gate |
| `full200_disagreement_diagnostics.md` | full200 same-ID 分歧诊断；WTQ/TabFact/CRT 分歧净贡献和代表样本 |
| `full200_disagreement_diagnostics.json` | full200 same-ID 分歧诊断结构化结果，供后续 WTQ discordant subset 抽样 |
| `wtq_discordant_debug_subset_50.md` | WTQ 调试子集摘要：40 条 MACT-only + 10 条 prioritized neither |
| `wtq_discordant_debug_subset_50.jsonl` | WTQ 调试子集结构化输入，保留 table、gold、myAgent/MACT prediction、tags、metrics |
| `wtq_compression_bucket_diagnostics.md` | WTQ 按 paired bucket 的压缩比例、策略和 prediction signal 诊断摘要 |
| `wtq_compression_bucket_diagnostics.json` | WTQ 压缩桶结构化诊断，用于定位检索/压缩和答案类型问题 |
| `wtq_gold_rowcol_loss_diagnostics.md` | WTQ gold literal cell 在原表/压缩表中的行列覆盖诊断 |
| `wtq_gold_rowcol_loss_diagnostics.json` | WTQ gold 行列覆盖结构化诊断 |
| `wtq_compressor_hypothesis_diagnostics.md` | WTQ 压缩器与 planner/operation 错误假设映射 |
| `wtq_compressor_hypothesis_diagnostics.json` | WTQ 假设映射结构化结果 |
| `wtq_fix_candidate_coverage_estimate.md` | WTQ 候选修复方向的覆盖收益估计 |
| `wtq_fix_candidate_coverage_estimate.json` | WTQ 候选修复收益估计结构化结果 |
| `wtq_extreme_only_global_rows_offline_check.md` | WTQ extreme/only 最小修复离线覆盖检查；18/50 新触发全行，10 条 literal gold 行丢失可恢复 |
| `wtq_extreme_only_global_rows_offline_check.json` | WTQ extreme/only 离线覆盖检查结构化结果 |
| `qwen3_32b_4gpu_2svc.env` | CRT tail 并行 shard 使用的两服务 vLLM profile：GPU `4,5;6,7`，端口 `8000/8001` |
| `shards/crt_121_160.jsonl` | CRT shard input rows 121-160 |
| `shards/crt_161_200.jsonl` | CRT shard input rows 161-200 |
| `logs/crt_full200_resume_stdout.log` | CRT single-runner detached stdout，2026-07-30 13:55:20 CST 启动，到 120/200 后改用 shard |
| `logs/crt_shard_121_160_stdout.log` | CRT shard 121-160 detached stdout，16:38:30 CST complete |
| `logs/crt_shard_161_200_stdout.log` | CRT shard 161-200 detached stdout，16:44:11 CST complete |
| `logs/crt_mact_full200_shard_121_160.log` | CRT shard 121-160 MACT log |
| `logs/crt_mact_full200_shard_161_200.log` | CRT shard 161-200 MACT log |
| `logs/wtq_mact_full200.log` | WTQ MACT full200 log |
| `logs/wtq_full200_resume_stdout.log` | WTQ detached stdout |
| `run_wtq_resume.sh` | WTQ resume runner |
| `run_tabfact_resume.sh` | TabFact resume runner |
| `run_crt_resume.sh` | CRT resume runner |
| `run_crt_shard_121_160.sh` | CRT shard 121-160 runner |
| `run_crt_shard_161_200.sh` | CRT shard 161-200 runner |

这些文件按 checkpoint 强制加入 MACT Git，因为 MACT 默认忽略 `outputs/`。

### 7.0.1 当前 MyAgent CRT full200 复跑结果

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_crt_full200_current_20260730_1822/
```

| file | current content |
|---|---|
| `LIVE_LEDGER.md` | 当前 CRT full200 复跑 ledger，记录双服务启动、运行、评估和关停 |
| `input/crt_blind200.jsonl` | CRT blind200 输入副本 |
| `myagent_crt200/merged/crt_qwen3-32b-local.jsonl` | 当前代码 CRT full200 merged：`200/200` |
| `myagent_crt200/eval/crt_qwen3-32b-local_eval.json` | 当前代码 CRT eval：`140/200 = 0.7000`，0 failed，0 missing |
| `crt_full200_current_comparison.md` | 当前代码、旧 myAgent、MACT 的 CRT full200 对比摘要 |
| `crt_full200_current_comparison.json` | CRT full200 对比结构化结果，含 old/new transition 和 staged composite |
| `qwen3_32b_4gpu_2svc.env` | 本次复跑双服务 vLLM profile：GPU `4,5;6,7`，端口 `8000/8001` |

关键结论：

| metric | value |
|---|---:|
| current myAgent CRT | 140/200 |
| old myAgent CRT | 137/200 |
| MACT CRT | 113/200 |
| new / MACT token ratio | 0.8461 |
| new / old myAgent token ratio | 1.0001 |
| failed / missing | 0 / 0 |
| staged composite if replacing CRT only | myAgent `456/600 = 0.7600` vs MACT `450/600 = 0.7500` |

### 7.0.2 WTQ representative100 回归结果

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_extreme_fix_representative100_20260730_1805/
```

| file | current content |
|---|---|
| `LIVE_LEDGER.md` | WTQ representative100 回归 ledger |
| `input/wtq_blind200_first100.jsonl` | WTQ blind200 前 100 条代表性输入 |
| `myagent_wtq100/merged/wtq_qwen3-32b-local.jsonl` | 修复后 myAgent WTQ100 merged：`100/100` |
| `myagent_wtq100/eval/wtq_qwen3-32b-local_eval.json` | 修复后 WTQ100 eval：`69/100 = 0.6900`，0 failed，0 missing |
| `wtq_representative100_extreme_fix_comparison.md` | 新旧 myAgent 与 MACT 的 WTQ100 对比摘要 |
| `wtq_representative100_extreme_fix_comparison.json` | WTQ100 对比结构化结果 |
| `qwen3_32b_4gpu_2svc.env` | 本次双服务 vLLM profile：GPU `4,5;6,7`，端口 `8000/8001` |

关键结论：新 myAgent `69/100`、旧 myAgent `69/100`、MACT `79/100`；new/MACT token ratio `0.5790`，new/old token ratio `1.0091`。该代表性切片恢复 3 条、回退 3 条，无净提升，所以 WTQ extreme/only 全局行修复不应继续作为主线扩大。

### 7.1 WTQ extreme/only debug50 实测结果

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_extreme_fix_debug50_20260730_173740/
```

| file | current content |
|---|---|
| `LIVE_LEDGER.md` | debug50 实测 ledger，记录双服务启动、runtime bug、修复和最终指标 |
| `input/wtq_debug50_with_answer.jsonl` | 从 full200 WTQ debug subset 派生的 50 条输入，补齐 `answer` 字段 |
| `myagent_debug50/raw/wtq/wtq_shard00_out.jsonl` | shard00 25/25 raw |
| `myagent_debug50/raw/wtq/wtq_shard01_out.jsonl` | shard01 25/25 raw；前 21 条来自初始 runner，后 4 条通过 `--append_output` 追加 |
| `myagent_debug50/merged/wtq_qwen3-32b-local.jsonl` | 修复后 myAgent debug50 merged：50/50 |
| `myagent_debug50/eval/wtq_qwen3-32b-local_eval.json` | 修复后 WTQ debug50 eval：`14/50 = 0.28`，0 exec failure，0 missing answer |
| `wtq_debug50_extreme_fix_measured_comparison.md` | 新旧 myAgent 与 MACT 的同 ID debug50 measured comparison |
| `wtq_debug50_extreme_fix_measured_comparison.json` | debug50 measured comparison 结构化结果 |
| `qwen3_32b_4gpu_2svc.env` | 本次双服务 vLLM profile：GPU `4,5;6,7`，端口 `8000/8001` |

### 7.2 MACT core100 当前结果

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

### 7.3 MACT core50 final

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

### 7.4 myAgent blind200 outputs

canonical myAgent full200 的可恢复 MACT 镜像目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_canonical_myagent_full200_raw_artifacts_20260730_2008/
```

该目录保存 canonical full200 所需的 myAgent raw/merged/eval/shards/logs。WTQ 源自 shortcutfix2 目录，TabFact/CRT 源自 current blind200 目录。下面的 MyAgent 原始路径仅用于说明历史来源，服务器清空后的恢复以 MACT 镜像目录为准。

完整 MyAgent 本地探索性输出恢复包：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/myagent_server_runs_archive_20260730_2020/
```

该目录保存 `/home/ubuntu/lzz/MyAgent/outputs/server_runs` 的压缩归档和清单，覆盖 43 个 run 目录、433 个文件、源目录 241M、压缩包约 27M。它用于服务器清空后恢复 smoke、Gate-50、TabFact 消融、WTQ 调参和历史本地 raw 输出；专家/专利主证据仍以 canonical MACT 镜像目录和 full200 paired summary 为准。

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

2026-07-30 当前代码只复跑了 CRT full200，WTQ/TabFact 沿用上述 current blind200 结果时的 staged composite：

| dataset | correct | accuracy | avg tokens | failed | missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 131/200 | 0.6550 | 6,226.93 | 0 | 0 |
| TabFact | 185/200 | 0.9250 | 2,426.89 | 0 | 0 |
| CRT current rerun | 140/200 | 0.7000 | 10,839.17 | 0 | 0 |
| Overall staged composite | 456/600 | 0.7600 | 6,497.66 | 0 | 0 |

### 7.5 多模型 Gate-50 结果位置

这些是已经完成的 myAgent-only Gate-50 筛选。结论是三个非主模型都不进入扩大实验。

统一 MACT 汇总目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/multimodel_gate50_summaries_20260730_1948/
```

该目录包含三个模型的 `*_gate50_summary.json/md` 和 `README.md`。这些文件由 `scripts/server/summarize_model_gate_results.py` 从历史 MyAgent eval 派生，没有重跑任何样本。

统一 MACT raw artifact 镜像目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/multimodel_gate50_raw_artifacts_20260730_2002/
```

该目录复制了三个历史 Gate-50 run 的 `raw/merged/eval/compare/shards/logs` 可用文件，用于服务器清空后的证据恢复和复查。`qwen25_3b_current_frozen_gate50_20260720` 原始 MyAgent 目录没有历史 `compare/` 文件；其 no-go 决策以 summary 目录为准。

| model | output dir | WTQ | TabFact | CRT | overall | avg tokens | decision |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-14B-AWQ | `/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen3_14b_awq_gate50_20260721/` | 37/50 | 44/50 | 27/50 | 108/150 | 7,344.51 | no-go |
| Qwen2.5-14B-AWQ | `/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen25_14b_awq_gate50_20260721/` | 34/50 | 45/50 | 28/50 | 107/150 | 7,308.35 | no-go |
| Qwen2.5-3B-Instruct | `/home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen25_3b_current_frozen_gate50_20260720/` | 28/50 | 40/50 | 21/50 | 89/150 | 7,298.51 | no-go |

这些模型没有进入 MACT paired 扩样，因为 Gate-50 已明显低于当前 Qwen3-32B 主模型，也低于 MACT Gate-50 reference。

截至 2026-07-30 17:00:51 CST，本机可见模型目录只有：

```text
/home/ubuntu/models/Qwen2.5-14B-Instruct-AWQ
/home/ubuntu/models/Qwen2.5-3B-Instruct
/home/ubuntu/models/Qwen3-14B-AWQ
/home/ubuntu/models/Qwen3-32B
```

因此当前没有新的本地模型需要启动。继续跑旧模型只会重复已知 no-go 结论；下一轮模型实验需要先新增/挂载一个未测候选模型，或提供可直接调用的外部模型 API key。当前环境变量检查没有发现可用的 DeepSeek / OpenAI / DashScope API key。

## 8. 已做优化和修复

| item | status | impact |
|---|---|---|
| 不再继续优先优化 TabFact | done | 目标转为三数据集总体是否超过 MACT |
| WTQ shortcutfix2 输出作为当前 WTQ blind200 myAgent 结果 | done | 保留 current-code WTQ blind200 131/200 结果 |
| vLLM stale pid / live port 检查 | done | 避免 pid stale 时重复启动服务 |
| MACT one-by-one + `--resume` | done | 单条失败不丢整批，服务器中断后可继续 |
| detached resume scripts | done | 防止 Codex 前台 session 断开导致长跑停止 |
| 周期性 Git checkpoint | done for current stop point | core100、full200 WTQ/TabFact/CRT raw/eval/paired/overall、WTQ representative100、CRT current rerun 均已同步到 GitHub |
| context length failure 保留为 failed/missing | ongoing | WTQ 中 MACT 的 `nu-4299`、`nu-2633`、`nu-3290`、`nu-3139`、`nu-3487` 当前被严格计入失败；后续 repair 需显式标注 |
| full200 seed 复用 | done | 从 core100 复制前 100 行，full200 只补 tail100，避免重跑已完成样本 |
| CRT 双服务 shard 并行 | done | 用户确认 GPU 资源可用后，使用 `4,5` 和 `6,7` 两个 Qwen3-32B 服务，将 CRT 121-160/161-200 分文件运行并按 ID 合并，避免双 runner 抢写同一 jsonl |
| WTQ extreme/only 全局行触发 | done measured debug50 | `TableCompressor._needs_global_rows` 加入 `only/top/first/last/earliest/latest`；debug50 新 myAgent `14/50`，18 条新触发全行中 `10/18` 正确，strict recoverable 中 `7/10` 正确 |
| numpy array execution result 判断 | done | `verification_gap` 改为显式判断非空执行结果，避免 numpy array truth-value 崩溃 |
| numpy array 输出序列化 | done | `_to_serializable` 和 `_json_default` 优先使用 `.tolist()`，避免多元素 numpy array `.item()` 崩溃 |
| 机器审计脚本 | done | `scripts/server/audit_qwen3_experiment_state.py` 可从 MACT 结果生成 `latest_experiment_readiness_audit.json` 和 `latest_expert_evidence_summary.md`，防止下次恢复时人工误读 canonical/staged 口径或重复启动 no-go 模型 |
| 新模型 Gate run 准备脚本 | done | `scripts/server/prepare_model_gate_run.py` 可为新增本地模型生成 MACT run 目录、`vllm.env`、启动/健康检查/停止脚本、Gate-10/Gate-50 runner 和 `gate_run_manifest.json`；默认 GPU `4,5;6,7`、端口 `8000/8001` |
| Gate-50 自动决策脚本 | done | `scripts/server/summarize_model_gate_results.py` 汇总 WTQ/TabFact/CRT eval，按 reference `124/150`、failure <= `2%`、token ratio <= `0.75` 输出 `no-go` 或 `gate150` |
| 本地临时文件 ignore | done | `configs/server/*.env.bak.*` 和早期 context 试跑脚本 `restart_qwen3_context_try.sh` 不进入远端恢复路径；正式入口以 PRD 第 14 节和 `prepare_model_gate_run.py` 为准 |
| 完整 MyAgent 输出归档 | done | 2026-07-30 已将 MyAgent `outputs/server_runs` 完整压缩到 MACT `myagent_server_runs_archive_20260730_2020`，并生成 `SHA256SUMS`、`inventory.tsv`、`run_directories.txt`、`source_size.txt` 和 README |

## 9. 当前可以写的结论

可以写：

```text
在 Qwen3-32B 本地模型、same-ID paired 的 blind50 core 实验中，
myAgent 总体准确率高于 MACT：124/150 vs 119/150，
平均 token 为 MACT 的 62.6%。

在 blind100 core 实验中，myAgent 总体准确率继续高于 MACT：
237/300 vs 227/300，平均 token 为 MACT 的 59.1%。

在 canonical blind200 full200 三数据集 same-ID paired 实验中，
myAgent 总体准确率略高于 MACT：453/600 vs 450/600，
平均 token 为 MACT 的 57.1%。

在 2026-07-30 只替换当前 CRT full200 复跑结果的 staged composite 中，
myAgent 为 456/600 vs MACT 450/600，平均 token 仍约为 MACT 的 57.1%。
```

必须带限制：

```text
WTQ 在 blind100 单项上仍低于 MACT：69/100 vs 79/100。
TabFact 小幅超过 MACT：95/100 vs 93/100。
CRT 明显超过 MACT：73/100 vs 55/100。
因此当前可以写“总体超过且 token 明显更低”，不能写“三个数据集全部超过”。

canonical full200 中 WTQ 仍低于 MACT：131/200 vs 148/200。
canonical full200 中 TabFact 仍低于 MACT：185/200 vs 189/200。
canonical full200 中 CRT 明显超过 MACT：137/200 vs 113/200。
当前 CRT 复跑后 CRT 为 140/200 vs MACT 113/200。
因此 full200 可以写“总体略高且 token 明显更低”，但必须写明优势主要来自 CRT，WTQ/TabFact 仍是短板。
```

不能写：

```text
三个数据集全部超过 MACT。
官方完整 full dataset 已完成。
所有本地模型都超过 MACT。
full200 对 MACT 是全面显著胜出。
```

新增限制：full200 acceptance 的“至少两个数据集不低于 MACT”未通过，只有 CRT 单项超过；canonical full200 总体只是 `453/600` vs `450/600` 的小幅领先，current CRT staged composite 是 `456/600` vs `450/600`。两者都应作为 staged evidence，而不是最终强结论。

## 10. 下一步小目标

| priority | task | output |
|---:|---|---|
| P0 | 同步 core100 eval/paired/summary final checkpoint | done: MACT `main` |
| P0 | 更新并同步本文档的 core100 结论 | done: 本 PRD |
| P0 | WTQ full200 补到 200 并 checkpoint | done: WTQ `200/200` raw/log complete |
| P0 | WTQ context overflow 处置 | pending: 当前按 failure 保留；若做 repair，需单独记录 repaired 口径 |
| P0 | WTQ full200 完成后生成 eval/paired | done: WTQ full200 myAgent `131/200` vs MACT `148/200` |
| P1 | TabFact full200 tail100 | done: TabFact `200/200` raw/eval/paired complete |
| P1 | CRT full200 tail100 | done: CRT `200/200` raw/eval/paired complete；121-160/161-200 用双服务 shard 并行 |
| P1 | full200 分歧诊断 | done: 诊断文件保存到 MACT full200 run；确认 WTQ 是主要负贡献，TabFact 暂不优先 |
| P1 | WTQ discordant subset 根因分析 | ready: 50 条调试子集已保存，下一步先分类错误类型，再决定是否改代码 |
| P1 | WTQ 压缩/预测信号诊断 | done: MACT-only 中 not-found-like/header prediction 明显集中；下一步验证检索/压缩是否漏关键行列 |
| P1 | WTQ 行列覆盖与候选修复排序 | done: 优先级为 extreme/only 全局行策略，其次行匹配扫描全行；大范围列保留不是第一优先 |
| P1 | WTQ 最小修复实验 | done measured debug50: old myAgent `0/50` -> new `14/50`；但 MACT `40/50`，这是 adversarial subset，不能作为总体结论 |
| P1 | WTQ 代表性回归切片 | done measured: 新 myAgent `69/100`，旧 myAgent `69/100`，MACT `79/100`；恢复 3 条、回退 3 条，无净提升 |
| P1 | 新模型筛选 | waiting: 2026-07-30 19:53 已复扫模型目录/缓存和外部 API env，仍无新增候选；GPU `4,5` 和 `6,7` 可用于下个候选的双服务 Gate，但除非新增/挂载模型或提供外部 API key，否则不继续启动模型 |
| P2 | 正式实验方案定稿 | ready next: 本文第 13 节已给出 gate-based 方案；下一步只在新增模型/API 后执行，不做全模型全量枚举 |

## 11. 当前决策建议

core100、canonical full200、current CRT staged composite 都满足“总体超过 MACT 且 token 明显更低”的阶段目标，但 full200 领先幅度较小：

```text
core100:
myAgent: 237/300 = 0.7900
MACT:    227/300 = 0.7567
token ratio: 0.5913

canonical full200:
myAgent: 453/600 = 0.7550
MACT:    450/600 = 0.7500
token ratio: 0.5708

current CRT staged composite:
myAgent: 456/600 = 0.7600
MACT:    450/600 = 0.7500
token ratio: 0.5708
```

WTQ/TabFact 在 full200 单项仍低于 MACT，CRT 是主要正贡献：

```text
WTQ full200:            myAgent 131/200 vs MACT 148/200
TabFact full200:        myAgent 185/200 vs MACT 189/200
CRT canonical full200:  myAgent 137/200 vs MACT 113/200
CRT current rerun:      myAgent 140/200 vs MACT 113/200
```

因此建议：

1. 专家/专利材料可以使用 `blind50 + blind100 + canonical full200 + current CRT staged composite` 作为 staged evidence。
2. 不要写“三个数据集全部超过”；core100 可写“TabFact/CRT 贡献优势，WTQ 为短板”，full200 应写“优势主要来自 CRT，WTQ/TabFact 仍低于 MACT”。
3. 本轮 full200、WTQ representative100、CRT current rerun 都已同步；不再继续恢复旧 CRT runner。
4. 当前本机没有未测候选模型，不建议启动服务重跑 Qwen3-14B-AWQ、Qwen2.5-14B-AWQ 或 Qwen2.5-3B-Instruct。
5. 若新增模型，先跑 Gate-10 smoke 和 myAgent-only Gate-50；只有 overall 接近或超过 Qwen3-32B，且失败率不超过 2%，才扩 Gate-150。
6. 只有 Gate-150 仍有竞争力的最终候选，才补 MACT same-ID paired-200。
7. 当前代码已完成 WTQ extreme/only global-row 的第一步最小修复、debug50 模型实测和 representative100 回归；代表性切片没有净收益，因此暂不把 WTQ 单点优化作为下一阶段主方向。
8. 若用户后续仍要继续优化 WTQ，优先候选是 `_match_rows` 低噪声全行 token 扫描，但必须先过小 gate，再考虑 full200。

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
MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/LIVE_LEDGER.md
MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_recovery_readiness_audit.md
MACT/outputs/server_runs/qwen3_32b_canonical_myagent_full200_raw_artifacts_20260730_2008/README.md
MACT/outputs/server_runs/myagent_server_runs_archive_20260730_2020/README.md
```

4. full200 已完成。恢复服务器后先复核行数和 summary，不要自动启动任何 runner：

```bash
wc -l /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/*_mact_full200.jsonl
wc -l /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_canonical_myagent_full200_raw_artifacts_20260730_2008/*/merged/*.jsonl
cat /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/overall_mact_full200_summary.json
cd /home/ubuntu/lzz/MACT/outputs/server_runs/myagent_server_runs_archive_20260730_2020
sha256sum -c SHA256SUMS
```

如需恢复 MyAgent 本地探索性 `outputs/server_runs`：

```bash
tar -xzf /home/ubuntu/lzz/MACT/outputs/server_runs/myagent_server_runs_archive_20260730_2020/myagent_outputs_server_runs_20260730_2020.tar.gz -C /home/ubuntu/lzz/MyAgent
```

如需复现或 repair，必须新建实验口径或明确标注 repaired 口径。旧的 resume 脚本保留用于审计，不应在当前 canonical 上继续运行：

```bash
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/run_wtq_resume.sh
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/run_tabfact_resume.sh
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/run_crt_resume.sh
```

CRT 本次最终补样使用的双服务复现入口：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
bash scripts/server/start_vllm_pool.sh /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/qwen3_32b_4gpu_2svc.env
bash scripts/server/healthcheck_vllm_pool.sh /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/qwen3_32b_4gpu_2svc.env

setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/run_crt_shard_121_160.sh
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/run_crt_shard_161_200.sh
```

5. core100 目录中的脚本只用于复现当前 100 行阶段：

```bash
setsid -f bash /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_core100_20260722/run_crt_resume.sh
```

这些脚本使用 `--resume --limit 100`，会从已有 `*_mact_core100.jsonl` 行数继续，不会重跑已保存样本。

## 13. 正式实验策略

正式实验不建议全量枚举所有模型、所有数据集、所有样本。当前建议采用四层漏斗，目标是用小样本快速淘汰明显不行的模型，只把服务器时间花在最终候选上：

1. Gate-10 smoke：每个候选模型先跑 WTQ / TabFact / CRT 各 10 条，只验证服务、schema、token 统计、失败保留和评估链路；任一数据集出现系统性 schema/connectivity failure，先修环境，不进入 Gate-50。
2. Gate-50：所有候选模型跑 myAgent-only WTQ / TabFact / CRT 各 50 条；总体明显低于 Qwen3-32B reference `124/150` 或 token 明显失控的模型直接 no-go。
3. Gate-150：Gate-50 接近 Qwen3-32B 的模型再跑各 150 条 myAgent-only，并检查失败率、平均 token、平均耗时。
4. Paired-200：只给最终候选模型补 MACT same-ID paired 200 条；如果服务器时间不够，优先保留 Qwen3-32B 的 blind100 / blind200 staged evidence，不补所有候选的 MACT。

当前已经可以用于专家材料的证据链是 core50 + core100 + canonical full200 + current CRT staged composite。full200 支持“总体略高且 token 明显更低”，current CRT 复跑把 staged composite 提到 `456/600`；但 WTQ/TabFact 单项仍低于 MACT，正式材料应把该限制写清楚。

当前阶段不要求跑官方完整 full dataset。只有当最终候选在 Paired-200 上稳定超过 MACT，且服务器预算允许时，才考虑官方完整测试集或更大 blind sample；否则专家材料采用 staged evidence，更符合时间成本约束。

新增模型的实际执行规则：

1. 先创建独立 run 目录，命名包含模型名、样本规模和日期，避免覆盖当前 Qwen3-32B 结果。
2. 可选 Gate-10 smoke：WTQ / TabFact / CRT 各 10 条，只验证服务、schema、token 统计和失败处理。
3. Gate-50 必跑：三数据集各 50 条 myAgent-only；若 overall 明显低于 Qwen3-32B Gate-50 reference `124/150`，直接 no-go。
4. Gate-150 条件：Gate-50 overall 接近或超过 `124/150`，执行失败率 <= `2%`，平均 token 没有明显失控。
5. Paired-200 条件：Gate-150 仍接近或超过 Qwen3-32B，并且至少两个数据集不弱于当前 Qwen3-32B 或有明确论文/专利价值。
6. MACT paired 只在最终候选上跑；raw、eval、paired、summary 和 ledger 仍保存到 MACT run 目录并推送。
7. 本地 Qwen 系列大模型默认使用两个 vLLM 服务并行：GPU `4,5` -> port `8000`，GPU `6,7` -> port `8001`；runner 按 shard 写入不同 raw 文件，最后按原始 ID 顺序合并，避免两个进程抢写同一个 jsonl。

## 14. 下一次新增模型的执行模板

本节是扩容/清空后继续实验的最小可执行入口。只在出现新模型目录或新 API key 后使用；当前四个本地模型不要重跑。

### 14.1 本地 vLLM 候选模型

先用准备脚本在 MACT 下创建 run 目录和全部脚本。该命令只写文件，不启动模型：

```bash
MODEL_TAG=<model_tag>
MODEL_ID=/home/ubuntu/models/<model_dir>
SERVED_MODEL_NAME=<served_model_name>

cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

python scripts/server/prepare_model_gate_run.py \
  --myagent-root /home/ubuntu/lzz/MyAgent \
  --mact-root /home/ubuntu/lzz/MACT \
  --model-id "$MODEL_ID" \
  --model-tag "$MODEL_TAG" \
  --served-model-name "$SERVED_MODEL_NAME"
```

脚本会输出 `run_dir`。后续从 `gate_run_manifest.json` 或 stdout 取 `RUN_DIR`：

```bash
RUN_DIR=/home/ubuntu/lzz/MACT/outputs/server_runs/<model_tag>_gate50_<timestamp>
cat "$RUN_DIR/gate_run_manifest.json"
```

启动和健康检查：

```bash
bash "$RUN_DIR/start_services.sh"
bash "$RUN_DIR/healthcheck_services.sh"
```

Gate-10 / Gate-50：

```bash
bash "$RUN_DIR/run_gate10.sh"
# Gate-10 若有连接错误、缺行、schema 错误或明显 context 问题，先排查服务，不进入 Gate-50。
bash "$RUN_DIR/run_gate50.sh"
```

Gate-50 完成后必须检查：

```bash
wc -l "$RUN_DIR"/myagent_gate50/merged/*.jsonl
cat "$RUN_DIR"/myagent_gate50/eval/*_eval.json
cat "$RUN_DIR"/gate50_summary.json
cat "$RUN_DIR"/gate50_summary.md
rg -n "Connection refused|APIConnectionError|context length|BadRequest|Traceback" "$RUN_DIR"/myagent_gate50/logs || true
```

Gate-50 决策：

| decision | condition |
|---|---|
| no-go | overall 明显低于 Qwen3-32B Gate-50 reference `124/150`，或 failed/missing > `2%`，或 token 明显失控 |
| Gate-150 | overall 接近或超过 `124/150`，三数据集均完整，失败率 <= `2%` |
| paired-200 | Gate-150 后仍有竞争力，且值得为专家/专利主表补 MACT same-ID 对照 |

`run_gate50.sh` 会自动调用 `summarize_model_gate_results.py` 生成 `gate50_summary.json` 和 `gate50_summary.md`。下一步是否进入 Gate-150 以该 summary 的 `decision` 为准，人工只复核异常日志和数据行数。

每次阶段结束都同步：

```bash
cd /home/ubuntu/lzz/MACT
git add -f "$RUN_DIR"
git commit -m "Record ${MODEL_TAG} gate results"
git push origin main

cd /home/ubuntu/lzz/MyAgent
git add docs/server/server_codex_reports/current-qwen3-mact-experiment-prd.md
git commit -m "Update ${MODEL_TAG} gate status"
git push origin codex/selective-risk-collaboration
```

### 14.2 外部 API 候选模型

如果是 DeepSeek / OpenAI / DashScope 等外部模型，不启动 vLLM；只在 `RUN_DIR` 中保存一个不含 secret 的 `api_profile.md`，记录 provider、base URL、model name、temperature、max tokens 和样本口径。API key 只放环境变量，不写入 Git。

执行时把 `--endpoints` 指到外部 OpenAI-compatible base URL，并把 `--api-key-env` 改成对应环境变量。其它 Gate-10/Gate-50/Gate-150/paired-200 条件不变。
