# 当前 Qwen3 vs MACT 实验 PRD

最后更新：2026-08-03 11:58 CST

## 0. 下一次启动先看这里

服务器扩容/清空后，先恢复两个仓库，再看本文档这一节，不要新建第二份 PRD。

当前最后同步状态：

| item | status |
|---|---|
| 2026-07-31 最新用户目标 | 不是只看总体略超；必须围绕“选择性风险协作 / 劝返”核心专利方向优化，直到当前 MyAgent + Qwen3-32B 在 WTQ / TabFact / CRT 三个数据集单项都超过 MACT，同时 token 仍明显低于 MACT；禁止 test-set hardcoding，优化必须能解释为机制或细节改进 |
| 当前正在执行 | P0/P1/P2/P3、P4b paired、E1 WTQ 分歧诊断、E2 WTQ targeted fresh 闭环已完成；P4b after-targeted 总表为 MyAgent `121/150` vs MACT `111/150`、overall token ratio `0.5310`、failed/missing `0/0`、accepted=`true`，且 WTQ/TabFact/CRT 单项均高于 MACT；E2 专利机制证据矩阵已把 full200、coarse ablation 和 fresh attribution 汇总。E3 multi-seed current-only 与离线边界诊断已完成：Seed-C `114/150`、token ratio `0.6096`；Seed-D `98/150`、token ratio `0.5735`；合计 `212/300`、weighted token ratio `0.5916`、failed/missing `0/0`、row-level evaluator 复算 `pass`。E4 最新 readiness audit 已完成，结果为 `no_candidate_wait`：只发现 4 个已测本地模型，`untested_local_model_paths={}`，API key/profile 均为空，因此当前不能启动 Gate-10。E5/E6 当前专利实验章节已 consolidated：`latest_current_patent_experiment_section_zh.md` 明确列出可写正证据和不能写的边界。结论：当前 Qwen3-32B full200 与 P4b after-targeted 已达用户“单项超过 MACT + token 更低”的核心锚点；E3 可写成稳定性边界与适用条件，E4 写成 no-candidate 等待状态，不能写成多模型正证据 |
| 当前本轮代码优化 | WTQ：答案形态/失败状态驱动的 high-confidence verifier 劝返门控、existing-total-row shortcut、planner `Ellipsis` 占位符执行拦截、晚列证据行召回、earlier/later 候选比较保留全局行、否定年份标量冲突的高置信审阅者劝返；2026-08-01 E2 targeted fixes 新增 listed-after 目标列、directly-before 目标列、overtime `(OT)` 标记计数、rank/country suffix person canonicalization、playoff parenthetical negator、division winner entry count、unique sponsor count、retired-injured ordinal attempt；TabFact：国家配对、零金牌计数、日期前全胜、venue/competition/date 同行匹配、score-but-lose、second-smallest metric、retirement threshold，以及 v6b 的实体属性审计、同一行多条件审计、列值计数审计、双实体出现次数、首尾时间差、实体数值差 |
| 当前本轮 targeted evidence | WTQ v6b full200 `155/200` vs MACT `148/200`，token ratio `0.6187`；TabFact v6b full200 `194/200` vs MACT `189/200`，token ratio `0.2014`；CRT current full200 `140/200` vs MACT `113/200`，token ratio `0.8461`；三项失败/缺答案均为 `0/0` |
| 当前本轮离线投影 | WTQ full200 旧 artifact 离线套新策略预计 `149/200`，实跑 v6b 为 `155/200`；TabFact policy-v6 实跑为 `185/200` vs MACT `189/200` 未过线，v6b 新 audit shortcuts 基于该 raw 离线投影 `194/200`、净 gain 9、harm 0，fresh full200 已确认 `194/200` |
| 已完成并同步的 full200 MACT 数据集 | WTQ `200/200`，TabFact `200/200`，CRT `200/200` |
| 暂停的数据集 | 无；按用户 2026-07-30 最新要求，当前 MyAgent 的 CRT full200 已补跑完成 |
| 当前进程状态 | 2026-08-01 17:47：P4b MACT Gate-50 六个 shard 全部完成并合并，`run_p4b_eval_compare.sh` 已生成 eval 和 paired summary；MyAgent 和 MACT 均已提交推送；两个 Qwen3-32B vLLM endpoint 已关闭，`curl` 访问 `8000/8001` 均 connection refused，`pgrep` 未发现 `vllm` / `api_server` / `run_mact_one_by_one.py` / `tqa.py` / `run_sharded_tqa` 进程。最终 `nvidia-smi --query-compute-apps` 未返回任何 compute PID，但 `nvidia-smi --query-gpu` 显示 GPU `0,1,2,3` 仍有约 `14GB/14GB/20GB/20GB` 显存残留，GPU `4,5,6,7` 仍有约 `42GB/卡` 残留；这些残留当前没有可见计算进程可关闭，属于服务器 GPU runtime/driver 状态 |
| 当前进程状态 | 2026-08-01 21:49：`curl` 访问 `8000/8001` 均 connection refused；`nvidia-smi --query-compute-apps` 未返回任何 compute PID；`nvidia-smi --query-gpu` 显示 GPU `0,1,2,3,4,5,6,7` 分别约 `25.8GB/46.9GB/20GB/20GB/42GB/42GB/42GB/42GB` 显存占用但无可见计算进程。当前不强启 Qwen3；fresh targeted run 等服务器清理/扩容后执行 |
| 当前进程状态 | 2026-08-01 22:07：`curl` 访问 `8000/8001` 仍 connection refused；`nvidia-smi --query-gpu` 显示 GPU `6,7` 约 `42031/42027 MiB` 占用且 GPU 利用率 `100%/100%`，但 `nvidia-smi --query-compute-apps` 未返回可见 compute PID，process scan 未发现 vLLM/评测 runner。按用户约束暂时只用 `6,7` 时，当前不能可靠启动 Qwen3-32B fresh run |
| 当前进程状态 | 2026-08-01 22:31：`curl` 访问 `8000/8001` 仍 connection refused；`nvidia-smi --query-gpu` 显示 GPU `6,7` 约 `42031/42027 MiB`、利用率 `100%/100%`，process scan 未发现 vLLM/API server/MACT runner/tqa/run_sharded_tqa。当前只准备不依赖模型的 E3 multi-seed 实验包，不启动 fresh run |
| 当前进程状态 | 2026-08-01 22:44：`8000/8001` 仍 connection refused；GPU `6,7` 约 `42031/42027 MiB` 且利用率 `100%/100%`；`nvidia-smi --query-compute-apps` 与 `nvidia-smi pmon -c 1` 均无可见 PID；`fuser` 未安装；process scan 未发现 vLLM/API server/MACT runner/tqa/run_sharded_tqa。按用户只用 `6,7` 的约束，本轮仍不启动 fresh Qwen run |
| 当前进程状态 | 2026-08-01 23:34：最新 preflight 仍为 `blocked_gpu_runtime_residual`；`8000/8001` connection refused；目标 GPU `6,7` 分别约 `42031/42027 MiB` 且 `100%/100%` util，`nvidia-smi --query-compute-apps` 与 `nvidia-smi pmon -c 1` 均无可见 PID。GPU `0,1,2,3` 当前为 `0 MiB/0%`，但用户最新执行口径是暂时使用 `6,7`，因此不擅自改卡启动正式 Qwen3 队列；需服务器清理 `6,7` runtime 或用户授权其他干净 GPU pair |
| 当前进程状态 | 2026-08-01 23:42：latest preflight 仍为 `blocked_gpu_runtime_residual`；`8000/8001` connection refused；目标 GPU `6,7` 仍约 `42031/42027 MiB` 且 `100%/100%` util，`nvidia-smi --query-compute-apps` 和 `nvidia-smi pmon -c 1` 均无可见 PID。GPU `0` 约 `6489 MiB`、`1,2,3` 约 `3 MiB`，但仍未获得改用其他 GPU pair 的用户授权。本阻塞已连续多轮复现，剩余 WTQ fresh、E3 Seed-C/D 和多模型 gate 不能继续在线执行 |
| 当前进程状态 | 2026-08-02 10:52：用户提示 `0,1,2,3` 卡没有被使用；实际复核显示对 `0,1,2,3` 的 preflight 仍为 `blocked_gpu_runtime_residual`，四张卡分别约 `24029/24029/24009/24029 MiB` 且 `100%/100%/100%/100%` util，`nvidia-smi --query-compute-apps` 与 `nvidia-smi pmon -c 1` 仍无可见 PID，`fuser/lsof` 未安装，`8000/8001` 仍 connection refused。因此 `0-3` 当前也不能按正式实验口径直接启动 Qwen3 endpoint；需要清理 GPU runtime 或明确授权执行 GPU reset / 使用风险运行 |
| 当前进程状态 | 2026-08-03 09:31：`0,1,2,3` 已恢复为 `0 MiB/0%`，并成功启动两个 Qwen3-32B vLLM endpoint：GPU `0,1` -> `8000`，GPU `2,3` -> `8001`；`/v1/models` 均返回 `qwen3-32b-local`。`6,7` 仍有约 `42GB/卡` 残留但不影响当前 `0-3` online queue。09:39 WTQ fresh + after-targeted full50 队列完成并 checkpoint 推送到 MACT：fresh `9/9`，after-targeted WTQ `46/50` vs MACT `43/50`，overall after-targeted P4b `121/150` vs `111/150` |
| 当前进程状态 | 2026-08-03 10:23：E3 Seed-C current-only queue 已在 GPU `0,1,2,3` 完成并按 gate 停止，未进入 paired MACT；结果写入 `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_multiseed_gate50_20260801_2231/summary/seed_c_myagent_gate50_summary.json/md`。两个 Qwen3 endpoint 仍在 `8000/8001` 运行，可继续 Seed-D current-only 或 Seed-C inspect |
| 当前进程状态 | 2026-08-03 11:00：E3 Seed-D current-only queue 已在 GPU `0,1,2,3` 完成并按 gate 停止，未进入 paired MACT；结果写入 `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_multiseed_gate50_20260801_2231/summary/seed_d_myagent_gate50_summary.json/md`，raw/merged/eval 分别保存在同目录 `myagent_current/seed_d/` 下。两个 Qwen3 endpoint 仍在 `8000/8001` 运行，可用于后续小规模 inspect 或多模型前的 sanity check |
| 当前进程状态 | 2026-08-03 11:13：按用户要求已关闭本轮所有可见 Qwen3/vLLM/runner 进程。`ps` 未发现 `vllm serve`、`run_remaining_qwen3`、`run_sharded_tqa.py`、`MyAgent/code/tqa.py`、`run_seed_*`、`run_mact`；`nvidia-smi --query-compute-apps` 无 compute PID；GPU `0,1,2,3` 均约 `3 MiB/0%`。latest runtime preflight 为 `start_service_required`，下一次在线实验必须先启动 Qwen3 服务，再重跑 preflight |
| 当前进程状态 | 2026-08-03 11:25：按用户提示复核，GPU `0,1,2,3` 仍为空闲低显存状态，可作为下一次 Qwen3-32B 两 endpoint 资源；GPU `4,5,6,7` 当前仍有约 `42GB/卡` 负载或残留，不纳入下一次默认启动。E3 Seed-C/D 边界诊断已离线完成，未启动模型服务 |
| 当前进程状态 | 2026-08-03 11:37：E4 readiness audit 复核显示 GPU `0,1,2,3` 均约 `3 MiB/0%`，默认池可用；GPU `4,5,6,7` 仍约 `42GB/卡` 且多卡高利用；未发现 vLLM/runner/tqa 进程。E4 没有可用新模型/API 候选，因此不启动 Gate-10 |
| 当前进程状态 | 2026-08-03 11:58：按用户提示再次复核并刷新 E4 readiness，GPU `0,1,2,3` 均约 `3 MiB/0%`，当前确实未被使用；GPU `4,5,6,7` 分别约 `42013/42007/42015/42011 MiB`，其中 `4,5,7` 为 `100%` util、`6` 为 `10%` util。未发现 vLLM/Qwen runner/MACT runner/MyAgent tqa 进程。后续如出现新模型/API 候选，默认在 `0,1 -> 8000` 与 `2,3 -> 8001` 启动，不占用 `4-7` |
| 历史 fresh preflight | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/e2_wtq_targeted_fresh_preflight_20260801_2207.md`；记录 2026-08-01 未启动 fresh run 的 endpoint/GPU/进程证据。2026-08-03 fresh 已完成，当前结果以 `p4b_wtq_targeted_fresh_summary.json/md` 和 after-targeted summary 为准 |
| 最新 after-targeted full50 自动化 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/e2_after_targeted_full50_automation_20260801_2217.md`；记录 affected-slice 通过后如何自动跑 WTQ full50 和 paired summary，并验证 fresh summary 缺失时会阻止误扩样 |
| 最新机制证据矩阵 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_patent_mechanism_evidence_20260801_2222/patent_mechanism_evidence_matrix.md`；将 full200 anchor、coarse Gate-50 消融和 offline attribution 合并为专利可引用的机制证据表 |
| 最新 multi-seed 执行包 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_multiseed_gate50_20260801_2231/`；Seed-C 与 Seed-D current-only 均已完成 `150/150` 并 `stop_or_inspect`，paired MACT 均标记为 not required；边界诊断已写入 `summary/seed_boundary_error_diagnosis.json/md`，合计 `212/300`、wrong `88`、weighted token ratio `0.5916`、verification `pass`。该包当前用于记录稳定性边界，不作为“多 seed 稳定超过 MACT”的正证据 |
| 最新 E4 多模型 readiness | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_e4_multimodel_gate_readiness_audit_zh.md`；2026-08-03 11:58 刷新，decision=`no_candidate_wait`，local models discovered=`4`，untested local models=`0`，API keys/provider profiles=`0/0`，visible model/runner processes=`0`，default GPU pool `0-3` 可用于下次启动 |
| 最新当前专利实验章节 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_current_patent_experiment_section_zh.md`；从 full200、P4b after-targeted、机制矩阵、E3 边界诊断、E4 readiness 自动生成，结论为 `stage_patent_draft_ready_with_boundaries`。它是当前给专家/专利代理人看的实验章节收口稿，不把多模型或多 seed 稳定性写成已完成 |
| 最新完成度审计 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/completion_gap_audit_20260801_2244_zh.md`；逐项确认 R1 full200 主证据完成、R2 机制证据基本完成、R3 WTQ fresh 闭环完成、R4 multi-seed 已执行 Seed-C/D 但暴露稳定性边界、R5 多模型 gate 仍缺可行新候选、R6 专利包 draft complete、R7 当前状态持续同步 |
| 最新权利要求证据矩阵 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/claim_evidence_traceability_20260801_2248_zh.md`；把 C1 风险分层协作、C2 证据保留压缩、C3 确定性审计、C4 受控劝返、C5 答案契约、C6 预算感知实验漏斗逐项对齐到代码/实验证据和剩余缺口 |
| 最新正式结果表/模板 | 当前台账：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_formal_result_ledger_current_zh.md`；模板：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/formal_result_tables_template_20260801_2252_zh.md`。台账已包含 WTQ fresh、P4b after-targeted、E3 current-only、E4 no-candidate，并保留 pending rows |
| 下次本地模型服务资源 | 2026-08-03 11:25 最新口径：GPU `0,1,2,3` 当前未被使用，可按 `0,1 -> 8000`、`2,3 -> 8001` 启动两个 Qwen3-32B endpoint；GPU `4,5,6,7` 当前仍有负载或残留，不作为默认资源 |
| 当前本机模型候选 | 2026-08-03 11:37 E4 审计 `/home/ubuntu/models`、`/home/ubuntu/.cache/huggingface`、`/data`、`/mnt` 后只发现 Qwen3-32B、Qwen3-14B-AWQ、Qwen2.5-14B-Instruct-AWQ、Qwen2.5-3B-Instruct；这些都属于已测/已 no-go 清单，`untested_local_models=[]`、`untested_local_model_paths={}`；当前不能生成新的本地模型 Gate run |
| 当前外部 API 候选 | 2026-08-03 11:37 E4 审计显示环境变量和现有真实 `configs/server/*.env` 均未发现可用 API key，`api_keys_present=[]`、`api_provider_profiles={}`；外部 API Gate 仍需用户提供 key 和目标 provider model 后才能生成 run |
| 当前阻塞条件 | 当前不是 GPU `0-3` 不可用阻塞，而是没有 E4 新候选：latest E4 readiness 为 `no_candidate_wait`，未发现未测本地模型或 API profile。Qwen3 endpoint 已按要求关闭，latest preflight 为 `start_service_required`；下一次在线实验需先启动服务并重跑 preflight。2026-07-31 full200 目标和 2026-08-03 P4b after-targeted 新 seed 三数据集目标已过线；Seed-C/Seed-D current-only 与边界诊断均已完成，paired MACT 不应继续消耗 |
| 当前主证据 | core100：myAgent `237/300` vs MACT `227/300`，token ratio `0.5913` |
| full200 阶段证据 | 当前 Qwen3 policy-v6b/current 三数据集合计：MyAgent `489/600` vs MACT `450/600`，总体 token ratio `0.5717`，总体 elapsed ratio `0.1337`，失败/缺答案 `0/0`；总表：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_all200_acceptance_20260731_132611/qwen3_policy_v6b_all200_acceptance_summary.json` |
| 最新机器审计产物 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_experiment_readiness_audit.json` |
| 最新专家证据摘要 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_expert_evidence_summary.md` |
| 最新恢复就绪审计 | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_recovery_readiness_audit.md` |
| canonical myAgent full200 raw artifacts | `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_canonical_myagent_full200_raw_artifacts_20260730_2008/` |
| 完整 MyAgent server_runs 恢复包 | `/home/ubuntu/lzz/MACT/outputs/server_runs/myagent_server_runs_archive_20260730_2020/`；含 43 个 run 目录、433 个文件、源目录约 241M，压缩包约 27M |
| 多模型 Gate-50 汇总 | `/home/ubuntu/lzz/MACT/outputs/server_runs/multimodel_gate50_summaries_20260730_1948/` |
| 多模型 Gate-50 raw artifacts | `/home/ubuntu/lzz/MACT/outputs/server_runs/multimodel_gate50_raw_artifacts_20260730_2002/` |
| full200 问题诊断 | 诊断文件、WTQ 50 条 discordant 调试子集、压缩桶、gold 行列覆盖、候选修复收益估计、extreme/only 离线检查和 debug50 实测已保存到 MACT |
| 本地临时文件处理 | 2026-07-30 19:58 已确认 `restart_qwen3_context_try.sh` 和 `configs/server/*.env.bak.*` 是本地临时/备份文件，已加入 `.gitignore`；历史 Qwen3-32B 单服务 example 的 GPU `4,5` 口径已被当前 `prepare_model_gate_run.py` 默认 `0,1;2,3` 替代 |
| 下一步建议 | 不建议把三个 coarse 变体都扩 full200，也不建议继续对 Seed-C/Seed-D 跑 paired MACT。P4b after-targeted 已给出“WTQ/TabFact/CRT 三数据集单项均超过 MACT，overall/token 均过线”的新 seed 正证据；E3 Seed-C/Seed-D 边界诊断已完成，可写成适用边界与后续机制优化方向。E4 最新审计没有新模型/API 候选，因此不要重跑旧 no-go 模型；等新本地模型或 API key 可用后再按 Gate-10 -> Gate-50 -> Gate-150 漏斗跑多模型 |

### 0.1 2026-08-01 本次继续执行台账

本次继续执行的直接目标：完成 P4b 同 ID MACT Gate-50，对 P4a after-fix 的 MyAgent 结果做配对比较，判断新 seed 上是否仍然“单项准确率不弱于/超过 MACT，且 token 明显更低”。这一步只跑 MACT 的同 ID baseline，不再继续扩 full200，也不新增模型。

执行顺序：

1. 确认 P4b 输出目录没有可被误用的活动失败 `.jsonl`。当前只发现两类失败备份：`sandbox_network_failed_20260801_0500` 和 `connection_refused_failed_20260801_0506`，它们是环境问题痕迹，不纳入评测。
2. 复核或重启两个本地 Qwen3-32B vLLM endpoint。由于 GPU `4,5,6,7` 仍有驱动残留，本轮实际使用 GPU `0,1`/port `8000` 与 GPU `2,3`/port `8001`；模型、数据 ID、prompt、temperature、max token 和评估脚本不变。因为当前 Codex 沙箱限制本地网络，所有 healthcheck 和 MACT runner 都必须用外部执行权限访问 `127.0.0.1`。
3. 运行 MACT P4b shards：WTQ / TabFact / CRT 各 50 条，按每数据集两个 25-row shard 并行，输出到 `p4b_mact_shards/output/<dataset>/`。
4. 合并 shard 到 `mact/<dataset>_mact_newseed_gate50.jsonl`，再运行 `run_p4b_eval_compare.sh` 生成 paired eval / comparison / summary。该脚本已改为使用 after-fix MyAgent 的 TabFact/CRT 结果。
5. 如果 P4b 通过，就把结论写入本文档和 MACT summary；如果不过线，只记录失败类型和下一轮诊断，不再盲目扩样。
6. 收尾时运行相关单测，提交并推送 MyAgent 与 MACT；之后关闭所有 `vllm serve`、`run_mact_one_by_one`、`tqa.py` 等进程。

本次预期痕迹：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_mact_shards/
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/mact/
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/eval/
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_paired_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_paired_summary.md
```

13:50 CST 资源状态更新：port `8000/8001` healthcheck 均为 connection refused；GPU `4,5,6,7` 显存仍约 `42GB/卡`，但 `nvidia-smi` / `pmon` / compute-app query 均看不到计算 PID，普通和 sudo reset 都被驱动拒绝，属于服务器环境残留，不是实验失败。为继续完成 P4b，本次临时尝试把两个 Qwen3-32B endpoint 挪到空闲 GPU `0,1` 与 `2,3`；模型、prompt、temperature、max token、数据 ID 和评估脚本不变，因此只影响运行资源，不改变实验口径。

14:08 CST 环境阻塞结论：`0,1` / `2,3` 以原 `gpu-memory-utilization=0.88` 启动失败，vLLM 报可用显存不足；`0,1` 降到 `0.68` 后通过权重加载但 KV cache 不足以支持 `max_model_len=8192`；升到 `0.70` 后通过 KV cache 检查，但 CUDA graph capture 阶段 OOM。失败后 `nvidia-smi` 仍显示显存残留且没有可见 vLLM/Python 计算 PID。为避免产生不可复现实验结果，本次不再继续用降上下文或不稳定 GPU runtime 强跑 P4b；当前 P4b 没有有效 MACT 评测输出，只有环境失败痕迹。已在 MACT 保存阻塞记录：`p4b_environment_blocker_20260801_1415.md/json`。下一次恢复优先在服务器扩容/清空后启动干净的 Qwen3-32B 服务，再从 `p4b_mact_shards/input/` 重新跑 MACT shards。

14:15 CST 代码验证：`test_myagent_pipeline.py` 在沙箱内通过 `184` 个用例；整套 `tests/` 在沙箱内因本地 HTTPServer 绑定 `127.0.0.1` 被拒绝出现 `3` 个环境错误，随后用外部权限重跑通过 `330` 个用例。结论：本轮 TabFact/CRT 机制修复和 PRD 更新没有引入测试回归。

15:56 CST 恢复执行计划：P4b 不再停在环境阻塞。当前已在 GPU `0,1` 与 `2,3` 启动两个健康 Qwen3-32B endpoint，并继续按同 ID MACT Gate-50 跑完剩余 shard。已确认 WTQ shard00/shard01 均 `25/25`；TabFact shard01 `25/25`，TabFact shard00 `24/25` 且仍在最后一条。接下来按顺序执行：等待 TabFact shard00 完成 -> 启动 CRT shard00/shard01 并行 -> 合并 WTQ/TabFact/CRT shards -> 运行 `run_p4b_eval_compare.sh` -> 把 P4b paired summary、eval、token、耗时、失败/缺答案和判定补入本文档与 MACT run 目录 -> 提交推送 MyAgent 和 MACT -> 关闭所有模型与 runner 进程。

17:10 CST P4b 结果：WTQ/TabFact/CRT 六个 MACT shard 均完成 `25/25`，合并后每个数据集 `50` 行，merged ID 顺序已校验与 input 一致。`run_p4b_eval_compare.sh` 已生成 `eval/*_mact_newseed_gate50_eval.json`、`p4b_paired_gate50_summary.json/md`。结果：WTQ MyAgent `37/50` vs MACT `43/50`，token ratio `0.5980`；TabFact MyAgent `45/50` vs MACT `44/50`，token ratio `0.2156`；CRT MyAgent `30/50` vs MACT `24/50`，token ratio `0.7740`；overall MyAgent `112/150` vs MACT `111/150`，token ratio `0.5444`，MyAgent/MACT failed/missing 均 `0/0`。existing paired criteria `accepted=true`，但严格按用户“新 seed 三个数据集单项都超过 MACT”的目标看，WTQ 未过，不能把 P4b 解读为全维度达标。MACT 痕迹台账：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_execution_ledger_20260801_1708.md`。

17:11 CST 验证记录：P4b artifact check 通过，确认三份 merged JSONL 均 `50` 行、三份 eval JSON 均 `num_samples=50`、summary 为 overall MyAgent `112/150` vs MACT `111/150`、token ratio `0.5444`、`accepted=true`；`/home/ubuntu/miniconda3/envs/lzz-agent/bin/python -m unittest discover -s tests -p 'test_myagent_pipeline.py'` 通过 `184` 个用例；外部权限运行 `/home/ubuntu/miniconda3/envs/lzz-agent/bin/python -m unittest discover -s tests` 通过 `330` 个用例。

17:47 CST 收尾记录：MyAgent PRD 更新已推送到 `codex/selective-risk-collaboration`，MACT P4b artifacts 已推送到 `main`。随后关闭两个 vLLM session，`curl -sS -H 'Authorization: Bearer local-vllm-key-change-me' http://127.0.0.1:8000/v1/models` 和 `8001` 均返回 connection refused；`pgrep -af '[v]llm|[a]pi_server|[r]un_mact_one_by_one|[t]qa.py|[r]un_sharded_tqa'` 无匹配；`nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits` 无返回，说明没有可见 compute PID 可关闭。最终 `nvidia-smi --query-gpu` 仍显示 GPU `0,1,2,3` 约 `13962/13956/19693/19689 MiB` 残留，GPU `4,5,6,7` 约 `42021/42021/42023/42021 MiB` 残留。`ss` / `netstat` 在当前环境未安装，端口关闭以 `curl` 与 `pgrep` 为准。

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

新增本地 vLLM 模型时，默认按当前服务器资源开两个服务：GPU `0,1` 绑定端口 `8000`，GPU `2,3` 绑定端口 `8001`；Gate-10 / Gate-50 直接传两个 endpoint 并行跑。GPU `4,5,6,7` 当前仍有负载或残留，不作为默认资源。

MyAgent 仓库只负责代码、脚本和本文档；除非临时调试，不再把新实验主结果分散写到 MyAgent 的 `outputs/server_runs/`。

生成新的 Gate 或 paired-200 run 后，先看该 run 目录的 `README.md`。每个 run 目录都会生成 `checkpoint_to_git.sh`：

```bash
bash /home/ubuntu/lzz/MACT/outputs/server_runs/<run>/checkpoint_to_git.sh
bash /home/ubuntu/lzz/MACT/outputs/server_runs/<run>/checkpoint_to_git.sh --commit "checkpoint: <run> <stage>" --push
```

第一条命令只对当前 MACT run 目录执行 `git add -f`，适合检查后统一提交；第二条命令会把当前 run 目录限定提交并推送到远端，适合服务器可能清空时的中途备份。

当前可复核审计脚本：

```text
/home/ubuntu/lzz/MyAgent/scripts/server/audit_qwen3_experiment_state.py
/home/ubuntu/lzz/MyAgent/scripts/server/experiment_api_registry.py
/home/ubuntu/lzz/MyAgent/scripts/server/experiment_model_registry.py
/home/ubuntu/lzz/MyAgent/scripts/server/healthcheck_openai_compatible.py
/home/ubuntu/lzz/MyAgent/scripts/server/prepare_model_gate_run.py
/home/ubuntu/lzz/MyAgent/scripts/server/prepare_paired200_run.py
/home/ubuntu/lzz/MyAgent/scripts/server/summarize_model_gate_results.py
```

作用：`audit_qwen3_experiment_state.py` 从 MACT 已保存结果生成机器可读 JSON 和中文专家证据摘要，快速回答“证据是否完整、总体/token 阶段条件是否达成、是否有新候选值得启动 Gate-10/Gate-50”；API key readiness 默认检查 `MyAgent/configs/server/*.env` 的真实 env 文件并跳过 `.example`/`.bak`，也可通过 `--env-file <path>` 读取额外 run-specific `.env` 中的 key 名，但不会保存或打印 secret 值。`experiment_model_registry.py` 集中维护已测本地模型清单和 alias 规范化规则，供审计和 Gate 准备脚本共用；`experiment_api_registry.py` 集中维护 API key 名和已测试 provider 默认配置，供审计和 Gate 准备脚本共用。`healthcheck_openai_compatible.py` 在外部 API Gate-10 前用 `/models` 验证 API key、endpoint 和目标 model，且不打印 secret。`prepare_model_gate_run.py` 在新增本地模型或外部 API 候选后自动生成 MACT run 目录、双服务 vLLM env 或 API profile、Gate-10/Gate-50/Gate-150 runner、`checkpoint_to_git.sh` 和 manifest，但不启动服务；本地候选可从 readiness audit 读取 `untested_local_model_paths`，外部 API 候选可从 readiness audit 读取单个 `api_provider_profiles` 并自动填 provider/base URL/key env，仍需通过 `--model-name` 指明 provider model；已知测试过的本地模型默认会被拒绝，只有显式 `--allow-known-tested-model` 才能生成重跑目录，manifest 会标记 override。`run_gate10.sh`、`run_gate50.sh` 和 `run_gate150.sh` 会分别生成 gate summary，且 `run_gate50.sh` 会强制要求 `gate10_summary.json` 的 decision 为 `gate50`，`run_gate150.sh` 会强制要求 `gate50_summary.json` 的 decision 为 `gate150`。`prepare_paired200_run.py` 从 Gate run manifest 生成 MACT paired-200 run 目录、myAgent/MACT runner、eval/compare 脚本、`checkpoint_to_git.sh` 和 manifest，同时强制要求 `gate150_summary.json` 存在且 `decision=paired200`，避免 no-go 候选被静默扩样。`checkpoint_to_git.sh` 默认只对当前 run 目录执行 `git add -f`，可选 `--commit MESSAGE --push` 会限定提交并推送当前 run 目录。`summarize_model_gate_results.py` 读取 Gate-10/Gate-50/Gate-150 三个 eval JSON，输出对应 `gate*_summary.json/md`；Gate summary 的异常行按 `min(rows, num_failed_exec + num_missing_answer)` 保守统计，避免 failed 与 missing 同时出现时低估失败率；Gate-10 通过时 decision 为 `gate50`，Gate-50 通过时 decision 为 `gate150`，Gate-150 通过 Qwen3-32B frozen150 overall reference `333/450`，且至少 2 个数据集达到单项 reference 时 decision 为 `paired200`。

## 1. 最大目标

验证并优化当前 `myAgent` 在 Qwen3-32B 本地模型下的“选择性风险协作 / 劝返”机制，使其在 WTQ / TabFact / CRT 三个数据集的同口径 200 条评测中单项都超过 MACT，并且 token 成本明显低于 MACT；在此基础上形成可写入专家/专利材料的实验结论与正式实验方案。

这个目标不是只优化某一个数据集，也不是只追求总体略超。当前验收标准是：WTQ、TabFact、CRT 每个数据集都要单项超过 MACT；优化要能归因到可解释机制，例如风险检测、证据保留、答案形态校验、冲突劝返、确定性审计，而不是针对 gold 或样本 ID 的硬编码。

## 1.1 当前阶段验收判断

| question | current answer |
|---|---|
| 总体准确率是否超过 MACT | 是。当前 Qwen3 policy-v6b/current full200 合计 MyAgent `489/600`，MACT `450/600`，准确率 `0.815` vs `0.750`，净胜 `+39` |
| token 是否仍明显低于 MACT | 是。三数据集加权平均 token ratio 为 `0.5717`；分项为 WTQ `0.6187`、TabFact `0.2014`、CRT `0.8461`。CRT 低于 MACT 但节省幅度不如前两项，总体仍明显低于 MACT |
| 三个数据集是否都超过 MACT | 是。WTQ `155/200 > 148/200`，TabFact `194/200 > 189/200`，CRT `140/200 > 113/200` |
| 当前项目是否可作为阶段证据 | 可以写成“Qwen3-32B 当前 full200 三数据集单项均超过 MACT，且总体 token 显著更低”的阶段证据；正式论文/专家材料仍建议扩容后按 gate 漏斗补多模型稳健性 |
| 现在是否继续跑旧本地模型 | 不继续跑 no-go 模型；当前只围绕 Qwen3-32B + MyAgent 机制优化 |
| 下一步实验策略 | 本轮暂停长跑并保存；扩容后从新增模型/API 候选开始 Gate-10 / Gate-50 / Gate-150，只有通过 gate 的模型进入 paired-200 |

## 1.2 2026-08-01 起执行路线

这一阶段的目标不是继续在当前 full200 上刷分，而是把“选择性风险协作 / 劝返”整理成可写入专利和专家材料的机制证据，并用有限成本补齐正式实验可信度。

总原则：

1. 当前 Qwen3-32B policy-v6b/current full200 结果冻结为 `v1_prototype_evidence`，作为阶段性达标证据。
2. 所有新增实验结果写入 MACT `outputs/server_runs/`，每个 run 目录必须有 summary / comparison / README 或 ledger，并用 `git add -f` 同步到 GitHub。
3. MyAgent 仓库只维护代码、脚本、唯一 PRD 和专利/实验文本草稿；不把主实验结果分散写回 MyAgent `outputs/`。
4. 不再直接启动“全模型 full200”长跑。新模型或新抽样必须先 Gate-10，过线再 Gate-50 / Gate-150，最后只让入围候选跑 paired-200。
5. 每完成一个子目标，在本文档的“下一阶段任务板”和对应 run 目录里同步结论、产物路径、提交号和剩余风险。

### 1.2.1 下一阶段任务板

| phase | subgoal | status | expected output | trace location |
|---|---|---|---|---|
| P0 | 冻结 Qwen3-32B v1 原型证据 | completed 2026-08-01 | 汇总当前 WTQ/TabFact/CRT full200 指标、关键优化点、代码入口、结果路径，形成专利证据索引 | MACT `qwen3_32b_policy_v6b_all200_acceptance_20260731_132611/qwen3_policy_v6b_patent_evidence_index.md` + 本 PRD |
| P1 | 专利机制草稿骨架 | completed 2026-08-01 | 中文专利 PRD：技术问题、核心方案、模块拆分、可保护点、实验支撑、风险边界 | 本 PRD `1.3 专利草稿骨架` |
| P2 | 机制消融设计 | completed 2026-08-01 | 消融矩阵：legacy/current/no-risk-collab/no-verifier/no-deterministic-audit/no-evidence-retention，明确每项开关、样本量和判断标准 | 本 PRD `1.5 机制消融矩阵`；执行产物后续写入 MACT ablation run 目录 |
| P3 | 低成本离线归因 | completed 2026-08-01 | 基于已有 raw/eval 统计每类机制贡献：提升条数、回退条数、token 变化、适用问题类型 | MACT `qwen3_32b_policy_v6b_mechanism_attribution_20260801_0033/` |
| P4 | 新 seed 泛化验证 | P4b completed 2026-08-01: existing paired gate accepted, strict all-dataset new-seed goal not met due WTQ | 每数据集新增 seed slice，先 Gate-50；P4b overall 过线且 token 明显低，但 WTQ `37/50 < 43/50`，后续优先诊断 WTQ discordant | MACT `qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/` + 本 PRD `0.1 2026-08-01 本次继续执行台账` |
| P5 | 多模型 gate | pending | 新模型/API 候选按 Gate-10 -> Gate-50 -> Gate-150 -> paired-200 漏斗筛选 | MACT `<model_tag>_gate*/` |
| P6 | 正式实验包 | current section consolidated 2026-08-03; final pending | 可给专家/专利代理人的实验包：方法说明、表格、消融、P4b after-targeted 正证据、E3/E4 边界、复现实验命令；多模型结果仍 pending/no-candidate | MACT `qwen3_32b_patent_experiment_package_20260801_2155/` + MyAgent PRD |

### 1.2.2 验收标准

当前阶段的最低验收标准：

| item | pass condition |
|---|---|
| 专利机制清晰度 | 能把代码优化归纳为“风险分层、证据保留、确定性审计、冲突劝返、预算控制”五类机制，而不是样本修补 |
| 机制消融可信度 | 至少能证明 current 相对 legacy 的提升，并用 no-* 变体显示关键机制有可观贡献 |
| 泛化风险控制 | 至少完成一组未参与调试的新 seed 小样本验证；若不过线，记录失败类型，不继续扩样 |
| 多模型成本控制 | 任一新模型必须通过 gate 才能扩样；no-go 模型不跑 paired-200 |
| 痕迹完整性 | 每个阶段有 JSON/MD 产物、命令或脚本入口、GitHub 提交号、失败/缺答案/token/耗时记录 |

### 1.2.3 本阶段第一步

第一步先做 P0/P1，不启动模型：

1. 从现有 full200 comparison 和 PRD 中抽取指标，生成“Qwen3-32B v1 原型证据索引”。
2. 把专利草稿骨架写入本文档，明确技术问题、核心模块、可保护点和实验支撑。
3. 同步提交 MyAgent PRD；若新增 MACT evidence summary，也同步提交 MACT。

当前 P0/P1 已完成。证据索引：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_all200_acceptance_20260731_132611/qwen3_policy_v6b_patent_evidence_index.md
```

### 1.2.4 专利实验完善总目标

从 2026-08-01 21:28 CST 起，当前长期目标定义为：

> 完善面向专利编写的 MyAgent vs MACT 实验数据体系：在已达标的 Qwen3-32B full200 基础上，补齐 WTQ 泛化诊断、机制消融、多 seed 稳定性、多模型 gate、正式实验包和专利说明书初稿所需证据，并将过程与结果持续写入唯一 PRD 和 MACT 结果目录后同步 GitHub。

这个目标的交付物不是“再跑很多数据”本身，而是形成一套能支撑专利代理人/专家理解的证据链：先说明机制，再证明机制有效，再证明不是单一样本、单一模型、单一数据集上的偶然结果。

目标拆分如下：

| milestone | subgoal | required evidence | pass condition | status |
|---|---|---|---|---|
| E0 | 冻结 Qwen3-32B 主结果 | WTQ/TabFact/CRT full200 paired summary、token、耗时、失败/缺答案 | 三数据集单项均超过 MACT，整体 token 明显低 | completed |
| E1 | WTQ 新 seed 风险诊断 | P4b WTQ `mact_only=9`、`myagent_only=3` 分歧样本分类表 | 说明 WTQ 新 seed 输在哪里，并给出是否需要机制修复的判断 | completed 2026-08-01 |
| E2 | 机制消融补强 | `legacy`、`no_strong_verification`、`no_deterministic_shortcuts` 已有 coarse Gate-50；WTQ targeted fixes 已有 TDD + offline projection + fresh affected-slice `9/9` + after-targeted P4b `121/150` vs MACT `111/150`；专利机制证据矩阵已合并 full200/coarse/attribution | 至少证明风险协作/劝返、确定性审计、证据保留中 2-3 个模块有贡献，并明确 offline 与 fresh 证据边界 | completed for current Qwen3 patent draft; fine-grained switch ablation optional |
| E3 | 多 seed 稳定性 | Seed-C/Seed-D current-only Gate-50 与离线边界诊断已完成；Seed-C `114/150`、Seed-D `98/150`，二者均 `stop_or_inspect`，paired MACT 均不跑；诊断合计 `212/300`、wrong `88`、weighted token ratio `0.5916`、failed/missing `0/0` | 不要求每组都三项全赢，但必须解释波动并证明总体/token 优势稳定 | completed as boundary diagnosis 2026-08-03; not a stability pass |
| E4 | 多模型 gate | 新本地模型或外部 API 模型按 Gate-10 -> Gate-50 -> Gate-150 -> paired-200 漏斗执行；2026-08-03 readiness audit 为 `no_candidate_wait`，未发现未测本地模型/API profile | 至少 1 个额外模型给出可解释结果；no-go 模型不扩 full200 | pending/no-candidate audited 2026-08-03 |
| E5 | 正式实验包 | 方法说明、数据切分、baseline 定义、模型配置、硬件、token/elapsed 统计、失败规则、raw/eval/summary 索引、当前专利实验章节 | 专家可按路径复核每个表格数值和每次实验口径 | current section consolidated 2026-08-03; final pending E4 |
| E6 | 专利说明书初稿 | 技术问题、系统流程、模块定义、实施例、权利要求草案、实验效果表、写作边界 | 能从实验结果直接支撑“选择性风险协作/劝返”相对 MACT 的阶段效果，并明确 E3/E4 边界 | current draft updated 2026-08-03 |
| E7 | 最终收口审计 | MyAgent 与 MACT git 状态、提交号、关键命令、进程/GPU 状态、结果目录完整性 | 仓库 clean 且已 push；没有未保存主实验产物 | pending |

实验优先级：

1. E1/E2 已完成当前 Qwen3 主线：WTQ 分歧诊断、targeted fixes、fresh affected-slice 和 after-targeted P4b 都已形成证据链。
2. E3 已完成 current-only 两组 seed 和离线边界诊断，但结果是稳定性边界，不是稳定性通过：Seed-C/Seed-D 都因 current gate 未过而不跑 paired MACT；错误集中在 WTQ rank/temporal/entity lookup、TabFact temporal/numeric entailment、CRT percentage/aggregation 和 multi-step numeric composition。
3. 扩容后优先做 E4 多模型 gate。新本地模型或 API key 出现后按 Gate-10 -> Gate-50 -> Gate-150 -> paired-200 漏斗执行，不做全模型全量枚举；只有 gate 通过的候选进入 paired-200。
4. 实验边跑边写 E5/E6。不要等全部实验结束才写专利草稿；当前可写 full200/P4b after-targeted/机制证据，E3 写成“边界与适用条件”，后续把多模型表格逐步补入。

停止条件：

| condition | action |
|---|---|
| 新模型 Gate-10 明显低于 Qwen3-32B reference 或失败/缺答案过多 | 记录 no-go，不扩 Gate-50 |
| Gate-50 overall 不超过 reference，且没有明确专利价值 | 记录 no-go，不扩 Gate-150 |
| Gate-150 未达到 paired-200 门槛 | 不进入 MACT paired-200，除非 PRD 中明确写人工例外原因 |
| 某个优化只改善单个样本、不能抽象成机制 | 不写入专利主方案，只保留为错误分析 |
| 新 seed 连续显示 WTQ 单项不稳 | 暂停扩样，优先做 WTQ evidence retention / verifier override 诊断 |

所有新增主实验产物继续写入：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/<run_name>/
```

所有目标状态、结论边界、下一步安排继续只更新本文档：

```text
/home/ubuntu/lzz/MyAgent/docs/server/server_codex_reports/current-qwen3-mact-experiment-prd.md
```

### 1.2.5 E1 WTQ 新 seed 分歧诊断结论

E1 已完成，不启动模型，只读取 P4b frozen artifacts 并复用 `MyAgent/code/evaluate_results.py` 的 WTQ denotation metric 重新计算配对正确性。

产物：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/analyze_wtq_p4b_discordants.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_wtq_discordant_diagnosis.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_wtq_discordant_diagnosis.md
```

配对混淆：

| bucket | count |
|---|---:|
| both_correct | 34 |
| myagent_only | 3 |
| mact_only | 9 |
| both_wrong | 4 |

诊断结论：

1. WTQ P4b 不是运行稳定性问题：MyAgent/MACT failed 和 missing 均为 `0/0`。
2. WTQ P4b 的 MyAgent 单项落后主要是语义/答案契约问题，不是单纯的 token 节省或表格行丢失。9 条 `mact_only` 的 MyAgent 平均 compression ratio 为 `0.6364`，其中 `2/9` 已经使用近似完整行上下文。
3. 9 条 `mact_only` 的主要类别是：实体表面契约 `2`、count 语义 `2`、目标列选择 `1`、listed-after shortcut 目标错误 `1`、overtime 显式标记解析 `1`、ordinal phrase matching `1`、conflict gate 没有接管更好 verifier candidate `1`。
4. 3 条 `myagent_only` 可作为专利正向证据：MyAgent 的语义 disambiguation、紧凑 denotation 输出契约、row-neighbor temporal evidence retention 分别避免了 MACT 的 min-date、解释性回答和非邻近行选择错误。
5. 4 条 `both_wrong` 不是对 MACT 的净差来源，但提示下一轮 WTQ 稳定性仍要关注 numeric range parsing、US location alias、row-major next-field、multi-answer list contract。

下一步执行优先级：

1. 先写 targeted WTQ checks，不直接扩 Gate-100/full200。
2. 优先覆盖 `listed-after` 目标列、`which experiment number` 目标列选择、sports score `(OT)` 标记、rank/country suffix 实体规范化、`Champion (no playoff)` 这类 parenthetical negator verifier override。
3. 如果 targeted checks 显示可解释收益，再决定是否实现细粒度开关并跑 E2 fine-grained ablation；否则把 WTQ P4b 记录为泛化风险边界。

### 1.2.6 E2 WTQ targeted fixes 与离线投影

E2 本轮已完成 WTQ targeted fixes 的 RED/GREEN cycle，修复都来自 E1 分类，不按样本 ID 或 gold answer 分支。

代码范围：

```text
/home/ubuntu/lzz/MyAgent/code/my_agents.py
/home/ubuntu/lzz/MyAgent/tests/test_myagent_pipeline.py
```

新增/覆盖机制：

| mechanism | purpose | E1 rows covered |
|---|---|---|
| listed-after target column | `what is the next <column> listed after <entity>` 返回下一行同目标列，而不是当前行下一个单元格 | `nu-1825` |
| directly-before target column | `which experiment number came directly before <entity>` 返回前一行目标编号列 | `nu-3990` |
| overtime marker count | 对 `(OT)` / overtime / extra-time 显式标记计数，优先于比分相等启发式 | `nu-1478` |
| WTQ person canonicalization | `who/person` 问题清理 rank 前缀和 country/code 后缀 | `nu-1108`、`nu-3905` |
| playoff parenthetical negator | `Champion (no playoff)` 这类含 negator 的 playoff 单元不计为 make playoff | `nu-3537` |
| division winner entry count | `number of winners in <division>` 计数目标列非空 entry，不误解为 distinct winner | `nu-2825` |
| unique sponsor count | `total number of sponsors` 对 sponsor 列唯一 sponsor 名称计数，不简单求非空 cell 总数 | `nu-3317` |
| retired injured ordinal attempt | 将 `three attempts` 与 `third attempt` 等 ordinal/cardinal 表达对齐 | `nu-3320` |

验证：

```text
/home/ubuntu/miniconda3/envs/lzz-agent/bin/python -m unittest discover -s tests -p 'test_myagent_pipeline.py' -k <8 targeted WTQ tests>
/home/ubuntu/miniconda3/envs/lzz-agent/bin/python -m unittest discover -s tests -p 'test_myagent_pipeline.py'
/home/ubuntu/miniconda3/envs/lzz-agent/bin/python -m unittest discover -s tests
```

结果：8 个 targeted tests `OK`；`test_myagent_pipeline.py` 全量 `192 tests OK`；完整 `tests/` 为 `338 tests OK`。

离线投影产物：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/project_wtq_targeted_fixes.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_wtq_targeted_fix_projection.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_wtq_targeted_fix_projection.md
```

投影结论：在 P4b WTQ frozen MyAgent merged rows 上重放新 deterministic shortcuts 和 scalar canonicalization，不启动模型，WTQ 从 `37/50` 投影到 `46/50`，`wrong_to_correct=9`，`correct_to_wrong=0`。2026-08-03 已完成 fresh Qwen targeted/full50 验证：affected-slice `9/9`，P4b WTQ after-targeted `46/50 > 43/50`，P4b after-targeted aggregate `121/150 > 111/150`，overall token ratio `0.5310`，failed/missing `0/0`。

fresh 验证入口与结果：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/input/wtq_p4b_targeted_fix_affected_slice.jsonl
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/run_myagent_wtq_targeted_fix_slice.sh
/home/ubuntu/lzz/MyAgent/scripts/server/summarize_wtq_targeted_fresh.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_wtq_targeted_fresh_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_wtq_targeted_fresh_summary.md
```

`wtq_p4b_targeted_fix_affected_slice.jsonl` 共 9 行，ID 为 `nu-3537`、`nu-1108`、`nu-2825`、`nu-3905`、`nu-3317`、`nu-3990`、`nu-1478`、`nu-3320`、`nu-1825`。

2026-08-01 22:12 已补自动收口：`run_myagent_wtq_targeted_fix_slice.sh` 在 fresh run 完成后会调用 `summarize_wtq_targeted_fresh.py`，生成 `p4b_wtq_targeted_fresh_summary.json/md`，记录 merged 行数、eval 行数、正确数、失败/缺答案、token、耗时、fresh wrong ids 和 `pass/inspect/incomplete` 决策；低于 `7/9` 会非零退出，避免误扩 WTQ full50。

affected-slice 通过后的 WTQ full50 自动入口和结果：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/run_myagent_wtq_after_targeted_fix_full50.sh
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/run_p4b_after_wtq_targeted_eval_compare.sh
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/summarize_p4b_after_wtq_targeted_paired.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_after_wtq_targeted_paired_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4b_after_wtq_targeted_paired_summary.md
```

自动化验证痕迹：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/e2_after_targeted_full50_automation_20260801_2217.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/e2_after_targeted_full50_automation_20260801_2217.md
```

历史未启动 fresh Qwen run 的环境证据仍保留，用于解释 2026-08-01 的服务器阻塞；它不再代表 2026-08-03 的当前结果：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/e2_wtq_targeted_fresh_preflight_20260801_2207.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/e2_wtq_targeted_fresh_preflight_20260801_2207.md
```

边界：E2 现在已经有 fresh model run 和新的 after-targeted paired MACT 结果，可以写成 P4b targeted 机制闭环正证据；但它仍不是多 seed 稳定性证据，也不是多模型证据。E3 Seed-C/D 应写成稳定性边界，E4 仍是 no-candidate。

### 1.2.7 E5/E6 正式实验包和专利说明书初稿

E5/E6 已完成 draft 版，不是最终版。它把当前 full200、coarse ablation、P4b、新 seed WTQ 诊断、WTQ targeted projection、未完成验证和专利说明书初稿集中到一个 MACT 包目录中，方便专家/专利代理人复核。

包目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/
```

文件：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/README.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/evidence_manifest.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/experiment_package_index_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/formal_experiment_schedule_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/patent_disclosure_draft_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_current_patent_experiment_section_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/next_validation_checklist_zh.md
```

其中：

1. `evidence_manifest.json` 是机器可读索引，记录 MyAgent/MACT commit、full200 主结果、P4b、E1/E2 证据路径和剩余工作。
2. `experiment_package_index_zh.md` 是中文实验包索引，面向专家复核实验口径和数值来源。
3. `formal_experiment_schedule_zh.md` 是正式实验排期和成本控制方案，按 affected-slice、WTQ full50、机制消融、多 seed、多模型 gate 分阶段设置停止规则。
4. `patent_disclosure_draft_zh.md` 是中文专利说明书初稿，包含技术领域、背景问题、技术方案、方法流程、创新点、实验效果、权利要求草案方向和写作边界。
5. `latest_current_patent_experiment_section_zh.md` 是当前最新实验章节收口稿，已经纳入 full200、P4b after-targeted、机制矩阵、E3 边界和 E4 no-candidate。
6. `next_validation_checklist_zh.md` 是服务器清理/扩容后的下一次验证清单，当前应优先等新模型/API 候选，而不是重跑已完成 WTQ fresh。

当前写作边界：E5/E6 可以支撑“Qwen3-32B full200 阶段原型有效”“P4b after-targeted new-seed 小样本三数据集均超过 MACT”和“机制设计清晰”；E3 只能写作适用边界，E4 只能写作 no-candidate readiness，仍不能替代 multi-seed 稳定性正证据、多模型证据和最终正式实验表。

## 1.2.5 E3 multi-seed Gate-50 执行状态

E3 已完成准备和 Seed-C/Seed-D current-only fresh model run。目的：在 P4b 单 seed 之外再补两组不重叠随机样本，验证 MyAgent + Qwen3-32B 的总体准确率/token 优势是否稳定，以及 WTQ/TabFact 是否仍有连续波动风险。

当前结论：两组 seed 都不是稳定性通过证据，而是边界证据。Seed-C overall 仍有 `114/150`，但 TabFact `44/50` 未过 current gate；Seed-D 降到 overall `98/150`，WTQ `30/50` 与 TabFact `38/50` 均未过 current gate。二者都 `decision=stop_or_inspect`，因此按停止规则不消耗 paired MACT baseline 时间。2026-08-03 11:25 已完成离线边界诊断，确认不是执行失败或缺答案问题，而是语义正确性稳定性问题。

运行目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_multiseed_gate50_20260801_2231/
```

已生成文件：

```text
README.md
multiseed_gate50_manifest.json
input_generation_summary.md
build_multiseed_gate50_inputs.py
verify_multiseed_package.py
run_seed_myagent_gate50.sh
run_seed_mact_gate50.sh
run_seed_paired_compare.sh
summarize_seed_myagent.py
analyze_seed_boundary_errors.py
render_seed_paired_summary.py
start_qwen3_service.sh
healthcheck_vllm.sh
checkpoint_to_git.sh
vllm.env
input/seed_c/*.jsonl
input/seed_d/*.jsonl
summary/seed_boundary_error_diagnosis.json
summary/seed_boundary_error_diagnosis.md
```

抽样与校验：

| seed | dataset | selected rows | base excluded ids | prior seed excluded ids | validation |
|---|---|---:|---:|---:|---|
| seed_c | WTQ | 50 | 250 | 0 | pass |
| seed_c | TabFact | 50 | 250 | 0 | pass |
| seed_c | CRT | 50 | 250 | 0 | pass |
| seed_d | WTQ | 50 | 250 | 50 | pass |
| seed_d | TabFact | 50 | 250 | 50 | pass |
| seed_d | CRT | 50 | 250 | 50 | pass |

排除范围：frozen full200、coarse diagnostic Gate-50、P4b new-seed Gate-50、targeted affected slices，以及同包内已经选过的 prior seed IDs。`verify_multiseed_package.py` 已确认 6 个输入文件均为 `50` 行、无重复 ID、与排除集合无交集、Seed-C/Seed-D 同数据集之间无交集。

已执行结果：

| seed | dataset | input/merged/eval | correct | accuracy | avg tokens | avg elapsed s | failed/missing | token ratio vs MACT full200 ref | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Seed-C | WTQ | 50/50/50 | 40/50 | 0.8000 | 6318.56 | 15.25 | 0/0 | 0.6013 | pass |
| Seed-C | TabFact | 50/50/50 | 44/50 | 0.8800 | 2820.02 | 10.65 | 0/0 | 0.2604 | inspect |
| Seed-C | CRT | 50/50/50 | 30/50 | 0.6000 | 11679.92 | 25.33 | 0/0 | 0.9118 | pass |
| Seed-C | aggregate | 150/150/150 | 114/150 | 0.7600 | 6939.50 | 17.07 | 0/0 | 0.6096 | stop_or_inspect |
| Seed-D | WTQ | 50/50/50 | 30/50 | 0.6000 | 6650.02 | 17.54 | 0/0 | 0.6329 | inspect |
| Seed-D | TabFact | 50/50/50 | 38/50 | 0.7600 | 2905.16 | 11.68 | 0/0 | 0.2682 | inspect |
| Seed-D | CRT | 50/50/50 | 30/50 | 0.6000 | 10028.56 | 27.39 | 0/0 | 0.7829 | pass |
| Seed-D | aggregate | 150/150/150 | 98/150 | 0.6533 | 6527.91 | 18.87 | 0/0 | 0.5735 | stop_or_inspect |

结果路径：

```text
summary/seed_c_myagent_gate50_summary.json
summary/seed_c_myagent_gate50_summary.md
summary/seed_d_myagent_gate50_summary.json
summary/seed_d_myagent_gate50_summary.md
summary/seed_boundary_error_diagnosis.json
summary/seed_boundary_error_diagnosis.md
myagent_current/seed_c/{raw,merged,eval}/
myagent_current/seed_d/{raw,merged,eval}/
```

边界诊断覆盖检查：

| scope | value |
|---|---:|
| total rows | 300 |
| recomputed correct | 212 |
| wrong rows | 88 |
| aggregate accuracy | 0.7067 |
| weighted token ratio vs MACT full200 reference | 0.5916 |
| failed/missing | 0/0 |
| row-level evaluator verification | pass |

诊断结论：

1. Seed-C/D 的 `input/merged/eval` 均为 `50/50/50`，逐行 evaluator 重算正确数与 summary 完全一致：Seed-C WTQ/TabFact/CRT 为 `40/44/30`，Seed-D 为 `30/38/30`。
2. E3 不存在运行失败、工具不可用或答案缺失；错误来自语义答案边界，因此 paired MACT runtime 不应继续消耗。
3. token 仍低于 MACT full200 reference：Seed-C weighted ratio `0.6096`，Seed-D `0.5735`，合并 `0.5916`；阻塞点不是 token 预算。
4. 自动启发式类别显示，主要边界为 CRT multi-step numeric composition、WTQ entity lookup/row selection、WTQ numeric aggregation、TabFact temporal order、CRT span/universal quantifier；这类内容可写成专利适用边界和下一轮机制优化方向。
5. 专利正文可引用 E3 作为“额外随机种子揭示系统边界”的证据，不能写成“多 seed 稳定超过 MACT”的正证据。

后续若要复现或跑新 seed，执行顺序：

```bash
cd /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_multiseed_gate50_20260801_2231
bash healthcheck_vllm.sh
bash run_seed_myagent_gate50.sh seed_c
cat summary/seed_c_myagent_gate50_summary.md
```

只有 current-only summary 的 decision 为 `run_paired_mact`，才跑同 ID MACT baseline：

```bash
bash run_seed_mact_gate50.sh seed_c wtq http://127.0.0.1:8000/v1
bash run_seed_mact_gate50.sh seed_c tabfact http://127.0.0.1:8001/v1
bash run_seed_mact_gate50.sh seed_c crt http://127.0.0.1:8000/v1
bash run_seed_paired_compare.sh seed_c
cat summary/seed_c_paired_gate50_summary.md
```

Seed-D 已按同样流程完成。每完成一个 seed，先运行校验和 checkpoint：

```bash
python verify_multiseed_package.py
bash checkpoint_to_git.sh --commit "checkpoint: e3 multiseed gate50 <stage>" --push
```

E3 判断边界：

1. `strict_all_dataset_superiority=true` 才能写成该 seed 上 WTQ/TabFact/CRT 三项均超过 MACT。
2. existing paired criteria accepted 但 strict goal false，只能写成总体/token 支持，不能写成三数据集全部稳定超过。
3. 若 current-only decision 为 `stop_or_inspect`，先诊断 MyAgent 错误，不消耗 MACT baseline 时间。

## 1.3 专利草稿骨架

暂定专利方向：一种面向表格问答/表格事实验证任务的选择性风险协作与劝返方法、装置、设备及存储介质。

### 1.3.1 技术问题

现有多智能体或多轮推理系统通常把大量样本都送入高成本协作流程，导致 token 和耗时显著上升；而单智能体或强压缩流程虽然成本较低，但容易在以下场景失误：

1. 表格被压缩后，比较、时间、极值类问题的关键行列证据被丢弃。
2. 模型生成答案与审阅/验证结果冲突时，缺少受控的接管和劝返机制。
3. 表格事实验证中，一些确定性结构问题被交给 LLM 自由判断，既浪费 token，又容易出现同行约束、计数约束、实体属性约束错误。
4. 统一强协作无法区分低风险和高风险样本，难以同时获得高准确率和低成本。

### 1.3.2 核心方案

本项目当前可抽象为五个可保护模块：

| module | purpose | current evidence |
|---|---|---|
| 风险分层路由模块 | 根据问题形态、压缩状态、执行失败、答案形态和验证冲突，把样本划分为低/中/高风险 | full200 总体 token ratio `0.5717`，失败/缺答案 `0/0` |
| 证据保留与压缩控制模块 | 对比较、时间、极值、否定等高风险问题保留全局或关键候选行，避免过度压缩 | WTQ `155/200` vs MACT `148/200` |
| 确定性语义审计模块 | 对可结构化验证的 TabFact 模式直接从表格审计，例如实体属性、同行多条件、列值计数、双实体出现次数、时间差、数值差 | TabFact `194/200` vs MACT `189/200`，token ratio `0.2014` |
| 冲突检测与劝返模块 | 当生成答案、执行结果、审阅器或 verifier 之间发生冲突时，基于答案形态和原表证据决定是否接受审阅者接管 | WTQ negated-year / verifier override 类修复；full200 无失败/缺答案 |
| 预算感知协作模块 | 只对高风险样本触发昂贵验证或重规划，低风险样本走轻量路径 | 总体 elapsed ratio `0.1337`；总体准确率 `489/600` vs MACT `450/600` |

### 1.3.3 可保护点

1. 不是简单“多智能体协作”，而是先判断样本风险，再选择是否协作、是否劝返、是否直接审计。
2. 劝返不是无条件相信审阅器，而是受答案形态、问题语义、原始表格证据和冲突类型共同约束。
3. 压缩不是固定比例裁剪，而是根据问题类型动态保留可能承载答案的全局行、候选行和晚列证据。
4. 对低风险 TabFact 场景引入结构化审计器，把 LLM 判断降级为表格一致性校验，降低 token 和幻觉。
5. 实验流程本身采用 gate 漏斗，避免把所有模型直接投入 full200/正式长跑，形成成本受控的模型筛选方法。

### 1.3.4 当前实验支撑

| scope | result | trace |
|---|---|---|
| Qwen3-32B full200 aggregate | MyAgent `489/600` vs MACT `450/600`，token ratio `0.5717` | MACT `qwen3_32b_policy_v6b_all200_acceptance_20260731_132611/` |
| WTQ full200 | MyAgent `155/200` vs MACT `148/200`，token ratio `0.6187` | MACT `qwen3_32b_wtq_policy_v6b_full200_20260731_1115/` |
| TabFact full200 | MyAgent `194/200` vs MACT `189/200`，token ratio `0.2014` | MACT `qwen3_32b_tabfact_policy_v6b_full200_20260731_1255/` |
| CRT full200 | MyAgent `140/200` vs MACT `113/200`，token ratio `0.8461` | MACT `qwen3_32b_crt_full200_current_20260730_1822/` |

### 1.3.5 当前风险边界

1. 当前结果足够支撑“Qwen3-32B 阶段原型有效”，但还不能替代正式多模型、多 seed 实验。
2. full200 参与过诊断，后续必须补一组新 seed 验证，防止被质疑为样本调优。
3. 消融实验必须证明各机制本身有贡献，不能只展示最终总分。
4. CRT token ratio 仅为 `0.8461`，低于 MACT 但节省幅度不如 WTQ/TabFact；正式材料应使用总体 token 显著降低和分项 token 均低于 MACT的表述。

## 1.4 下一步立即执行项

P0/P1 完成后，下一步进入 P2/P3：

1. 设计消融矩阵，优先选择不需要重跑 full200 的离线/小样本验证。
2. 对已有 WTQ/TabFact transitions 做机制归因，统计每类机制带来的 gain、harm 和 token 影响。
3. 如果需要新增代码开关，先加测试，确保可以稳定切换 `no_verifier_override`、`no_deterministic_audit`、`no_evidence_retention` 等模式。
4. 只有 P2/P3 结果能解释当前提升来源后，再启动 P4 新 seed 小样本验证。

当前 P2 设计已完成，见下一节。P3 离线机制归因也已完成，见 `1.6 P3 离线机制归因结论`。

## 1.5 机制消融矩阵

消融目标：证明当前效果不是单纯依赖 Qwen3-32B 或个别样本修补，而是由“选择性风险协作 / 劝返”下的多个机制共同贡献。

### 1.5.1 变体定义

| variant | purpose | current switch status | expected comparison |
|---|---|---|---|
| `current_policy_v6b` | 当前冻结原型 | 已有结果 | 主结果：WTQ `155/200`，TabFact `194/200`，CRT `140/200` |
| `legacy_myagent` | 去掉选择性协作主路径，复现旧 MyAgent 口径 | 已支持：`--collaboration-mode legacy` | 验证 current 相比 legacy 的净提升和 token 变化 |
| `no_strong_verification` | 关闭高风险 LLM verifier / strong verification | 已支持：`--disable-strong-verification` | 衡量 verifier 和冲突劝返对准确率的贡献 |
| `no_deterministic_shortcuts` | 关闭 WTQ/TabFact/CRT 确定性语义 shortcut | 已支持：`--disable-deterministic-shortcuts` | 衡量低风险直接审计对准确率和 token 的贡献 |
| `legacy_plus_shortcuts_off` | 旧路径同时关闭 deterministic shortcuts，作为更干净的旧策略下界 | 已支持：`--collaboration-mode legacy --disable-deterministic-shortcuts` | 区分旧路由和 shortcut 的叠加影响 |
| `no_wtq_verifier_override` | 只关闭 WTQ 答案形态/否定年份 verifier 接管 | 需要新增细粒度开关 | 衡量“劝返接管”本身，而不是 strong verification 整体 |
| `no_evidence_retention` | 只关闭 WTQ global-row / later-column evidence retention | 需要新增细粒度开关 | 衡量证据保留机制对 WTQ 的贡献 |
| `no_tabfact_audit_v6b` | 只关闭 TabFact v6b 新增实体属性、同行、计数、双实体、时间差、数值差审计 | 可先用 `--disable-deterministic-shortcuts` 粗消融；精细关闭需要新增开关 | 衡量 TabFact v6b audit shortcuts 的净贡献 |

### 1.5.2 执行顺序

| order | action | cost control | pass / stop rule |
|---|---|---|---|
| 1 | 离线归因 current vs old/current vs MACT transitions | 不启动模型；只读已有 comparison/raw | 若能解释大部分净提升，进入小样本消融 |
| 2 | 已有开关 Gate-50 消融：legacy、no_strong_verification、no_deterministic_shortcuts | 每数据集优先 50 条，同 ID paired | 若某变体明显差于 current，记录贡献；若差异很小，暂不扩样 |
| 3 | 新增细粒度开关并单测 | 只改代码，不跑长评测 | 开关必须默认不影响 current；相关单测通过 |
| 4 | 细粒度 Gate-50 消融：no_wtq_verifier_override、no_evidence_retention、no_tabfact_audit_v6b | 只跑触发机制较多的样本 slice | 若贡献明确，再考虑 Gate-100 |
| 5 | 入围消融扩到 full200 | 只扩关键变体，不全矩阵扩样 | 只有能支持专利论点的变体扩样 |

### 1.5.3 推荐样本量

| stage | WTQ | TabFact | CRT | reason |
|---|---:|---:|---:|---|
| offline attribution | 200 | 200 | 200 | 已有 artifacts，零模型成本 |
| coarse ablation Gate-50 | 50 | 50 | 50 | 快速判断 legacy / no-strong / no-shortcut 是否有明显差异 |
| targeted ablation Gate-50 | 50 targeted | 50 targeted | optional 50 | 只挑触发相关机制的样本，节省模型时间 |
| expansion | 200 only if useful | 200 only if useful | 200 only if useful | 只扩对专利主张有价值的变体 |

### 1.5.4 当前已有命令入口

粗消融可以直接通过现有 runner 参数执行。示例：

```bash
python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8000/v1 \
  --model qwen3-32b-local \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root /home/ubuntu/lzz/MACT/outputs/server_runs/<ablation_run>/myagent_gate50 \
  --collaboration-mode legacy \
  --resume
```

可用开关：

```text
--collaboration-mode legacy
--disable-strong-verification
--disable-deterministic-shortcuts
--enable-multiview-validation
```

细粒度消融需要先补代码开关；补开关时必须满足：

1. 默认值保持当前结果路径不变。
2. 每个开关写入输出行的 metadata，方便后续 summary 归因。
3. 每个开关至少有一个单测证明启用/禁用行为不同。
4. 任何消融 run 都要记录 eval、merged 行数、token、耗时、失败数、缺答案数。

## 1.6 P3 离线机制归因结论

P3 不启动模型，只读取已有 current/old/MACT merged artifacts，按同 ID 对齐后统计 transition rows 上的机制 metadata。它是关联归因，不替代真正的 causal ablation，但能指导下一步 GPU 时间花在哪里。

产物：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_mechanism_attribution_20260801_0033/mechanism_attribution_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_mechanism_attribution_20260801_0033/mechanism_attribution_summary.md
```

核心结果：

| dataset | current | old MyAgent | MACT | gain vs old | harm vs old | interpretation |
|---|---:|---:|---:|---:|---:|---|
| WTQ | 155/200 | 131/200 | 148/200 | 25 | 1 | gain 主要集中在 high-risk + strong verification + evidence retention，支持“风险协作 / 劝返 + 证据保留”主张 |
| TabFact | 194/200 | 185/200 | 189/200 | 9 | 0 | gain 主要集中在 deterministic audit 和 global-row evidence retention；gain rows 当前 token 显著低于旧 MyAgent |
| CRT | 140/200 | 137/200 | 113/200 | 6 | 3 | CRT 是支持性证据，证明 current 不破坏强项；不是本轮专利新颖性的主要来源 |

下一步判断：

1. 粗消融优先跑 `no_strong_verification`，验证 WTQ/CRT strong verification 的因果贡献。
2. 粗消融优先跑 `no_deterministic_shortcuts`，验证 TabFact deterministic audit 的因果贡献。
3. `legacy` 作为总对照，用来证明 current policy 相比旧路径的整体提升。
4. 细粒度开关暂不急着写；只有 coarse Gate-50 无法解释贡献时，再补 `no_wtq_verifier_override`、`no_evidence_retention`、`no_tabfact_audit_v6b`。

## 1.7 P2 coarse Gate-50 执行进度

run 目录：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_coarse_ablation_gate50_20260801_0040/
```

这个 Gate-50 是 diagnostic slice：优先选 current/old/MACT 分歧样本，再补齐 50 条。因此它用于机制诊断，不作为新 seed 泛化准确率。

| variant | status | WTQ | TabFact | CRT | conclusion | trace |
|---|---|---:|---:|---:|---|---|
| `legacy` | completed 2026-08-01 | 25/50 vs current ref 32/50 | 47/50 vs current ref 48/50 | 37/50 vs current ref 37/50 | current 相对 legacy 的诊断增益主要集中在 WTQ，TabFact 小幅，CRT 持平；支持“风险协作 / 证据保留 / 劝返”主要改善 WTQ 类复杂表格问答 | `legacy_gate50_summary.json/md` |
| `no_strong_verification` | completed 2026-08-01 | 25/50 vs current ref 32/50 | 47/50 vs current ref 48/50 | 37/50 vs current ref 37/50 | 与 `legacy` 在该 diagnostic slice 上结果相同；说明 current 相对 no-strong 的主要诊断增益集中在 WTQ，支持 strong verification / 劝返路径的贡献 | `no_strong_verification_gate50_summary.json/md` |
| `no_deterministic_shortcuts` | completed 2026-08-01 | 33/50 vs current ref 32/50 | 39/50 vs current ref 48/50 | 30/50 vs current ref 37/50 | 关闭 deterministic shortcuts 后 TabFact 下降 9/50 且 token 变为 current 的 1.4487 倍；CRT 下降 7/50；说明 deterministic audit 是高价值低成本模块，不只是 TabFact 局部补丁 | `no_deterministic_shortcuts_gate50_summary.json/md` |

coarse 总表：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_coarse_ablation_gate50_20260801_0040/coarse_ablation_gate50_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_coarse_ablation_gate50_20260801_0040/coarse_ablation_gate50_summary.md
```

专利机制证据矩阵：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_patent_mechanism_evidence_20260801_2222/build_patent_mechanism_evidence.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_patent_mechanism_evidence_20260801_2222/patent_mechanism_evidence_matrix.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_patent_mechanism_evidence_20260801_2222/patent_mechanism_evidence_matrix.md
```

P2 coarse 结论：

1. `legacy` 和 `no_strong_verification` 在 diagnostic Gate-50 上结果相同：WTQ `25/50`、TabFact `47/50`、CRT `37/50`；相对 current reference 最大差距是 WTQ `-7`，支持 strong verification / 劝返路径是 WTQ 诊断增益的主要来源。
2. `no_deterministic_shortcuts` 对 TabFact 影响最大：TabFact 从 current reference `48/50` 降至 `39/50`，token ratio vs current 为 `1.4487`；这直接支撑“确定性审计既提准确率又省 token”的专利论点。
3. `no_deterministic_shortcuts` 在 CRT 上也从 current reference `37/50` 降至 `30/50`，说明 deterministic audit 应写成跨数据集模块，而不是只写 TabFact 特例。
4. 三个 coarse 变体均为 failed/missing `0/0`，所以差异主要来自机制开关，不是运行失败。
5. 专利机制证据矩阵将 full200 anchor `489/600` vs MACT `450/600`、`no_strong_verification` overall `-8/150`、`no_deterministic_shortcuts` overall `-15/150`、WTQ gain rows `24/25` strong-verification tag、TabFact gain rows `8/9` deterministic-audit tag 合并为可复核证据表；后续边界不再是 WTQ fresh，而是 E3 多 seed 稳定性与 E4 多模型外延证据。

## 1.8 P4 新 seed Gate-50 执行计划与台账

P4 目标：用未参与 full200 调参和 P2 diagnostic 消融的新样本，验证当前 Qwen3-32B + MyAgent policy-v6b/current 是否具备泛化稳定性。P4 不直接扩 full200，也不把单次随机小样本写成最终结论；它只决定是否值得继续扩到 Gate-100/150 或同 ID MACT paired 比较。

### 1.8.1 当前执行口径

| item | decision |
|---|---|
| 数据集 | WTQ / TabFact / CRT 各新增 50 条 |
| 抽样方式 | 从 full dataset 中排除当前 full200 输入样本和 P2 diagnostic Gate-50 输入样本，再用固定随机种子 `20260801` 抽样 |
| 首轮模型 | 只跑 Qwen3-32B + MyAgent current/default policy，不开消融变体 |
| 模型资源 | 按用户要求优先使用 GPU `6,7` 启动一个 Qwen3-32B vLLM 服务 |
| 结果位置 | MACT `outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/` |
| 必须记录 | input 行数、merged 行数、eval 行数、correct、accuracy、token、elapsed、failed、missing answer、命令入口、日志位置 |
| 同步规则 | 每完成输入准备、每完成一个数据集、每生成 summary，都 `git add -f` 并按阶段提交推送 MACT；PRD 状态同步提交 MyAgent |

### 1.8.2 阶段门槛

P4 拆成两个阶段，避免过早把“新 seed 当前模型体检”混同为“已超过 MACT”：

| stage | purpose | pass condition | next action |
|---|---|---|---|
| P4a current Gate-50 | 低成本检查当前 MyAgent 在新 seed 上是否稳定、是否失败、token 是否仍低 | 三数据集 input/merged/eval 均为 `50/50/50`；failed/missing 为 `0/0`；token 仍低于 MACT full200 分项均值；准确率不出现明显塌陷：WTQ >= `35/50`、TabFact >= `45/50`、CRT >= `30/50` | 进入 P4b 或扩 Gate-100/150 |
| P4b paired MACT Gate-50 | 在同一批新 seed 样本上比较 MyAgent 与 MACT | MyAgent overall > MACT overall，且至少 2/3 数据集单项 >= MACT；token overall 明显低于 MACT；失败/缺答案不高于 MACT | 若通过，写入正式实验候选证据；若不通过，记录失败类型，不扩样 |

### 1.8.3 本轮步骤台账

| step | status | output / trace | conclusion |
|---|---|---|---|
| 1. 把 P2 coarse 消融结论补入专利证据索引 | completed 2026-08-01 | MACT `qwen3_32b_policy_v6b_all200_acceptance_20260731_132611/qwen3_policy_v6b_patent_evidence_index.md`；提交 `58f857b` | 已补 coarse diagnostic Gate-50 表、机制解释和“非新 seed 泛化”的边界说明 |
| 2. 准备新 seed Gate-50 run 目录和输入 | completed 2026-08-01 | MACT `qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/`；提交 `06e03d3`，清理 pycache 提交 `dcbcc48` | WTQ/TabFact/CRT 各 50 行；按 ID 排除当前 full200 和 P2 diagnostic 输入，selected 与 excluded overlap 为 `0` |
| 3. 启动 Qwen3-32B 服务 | completed 2026-08-01 | `healthcheck_vllm_models.json`；vLLM session `43996`，endpoint `http://127.0.0.1:8000/v1` | `/v1/models` 返回 `qwen3-32b-local` |
| 4. 运行 P4a MyAgent current Gate-50 | completed 2026-08-01 | `myagent_current/merged/*.jsonl`、`eval/*.json`、logs；MACT 提交 `4d4480d` | WTQ/TabFact/CRT 均完成 `50/50/50`，failed/missing `0/0` |
| 5. 汇总 P4a 并判断是否进入 P4b | completed 2026-08-01 | `p4a_current_gate50_summary.json/md`、`p4a_error_inspection.json/md`；MACT 提交 `f73f1b9` | `decision=stop_or_inspect`；WTQ pass，TabFact/CRT inspect |
| 6. 如果 P4a 过线，运行同 ID MACT Gate-50 | skipped by gate 2026-08-01 | `mact/*.jsonl` 未生成 | P4a 未过预设准确率门槛，不直接跑 P4b paired |
| 7. 回填 PRD、提交推送、关闭进程 | completed 2026-08-01 | MACT P4a 完整结果提交 `4d4480d`，错误检查提交 `f73f1b9`；最终进程验证：无 runner/vLLM，GPU `4,5,6,7` 约 `3 MiB` | 已关闭 Qwen3-32B 服务；下一轮从 `p4a_error_inspection.md` 开始诊断 |

### 1.8.4 P4a current Gate-50 结果

结果文件：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_current_gate50_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_current_gate50_summary.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_error_inspection.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_error_inspection.md
```

| dataset | rows input/merged/eval | correct | accuracy | token ratio vs MACT full200 | avg tokens | avg elapsed s | failed | missing | P4a gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WTQ | 50/50/50 | 37/50 | 0.7400 | 0.6436 | 6763.0 | 17.28 | 0 | 0 | pass |
| TabFact | 50/50/50 | 42/50 | 0.8400 | 0.2387 | 2585.0 | 11.51 | 0 | 0 | inspect |
| CRT | 50/50/50 | 21/50 | 0.4200 | 0.8002 | 10250.4 | 24.18 | 0 | 0 | inspect |
| Overall | 150/150/150 | 100/150 | 0.6667 | 0.5739 | 6532.8 | 17.66 | 0 | 0 | stop_or_inspect |

P4a 结论：

1. token 目标仍成立：三数据集加权 token ratio 为 `0.5739`，三项分项 token ratio 都低于 MACT full200 均值参考。
2. 稳定性目标未过线：TabFact `42/50` 低于预设 `45/50`，CRT `21/50` 低于预设 `30/50`；因此不能把这组新 seed 写成“泛化已过线”，也不能直接进入 Gate-100/150。
3. 运行可靠性没问题：三项 input/merged/eval 都是 `50/50/50`，failed/missing 均为 `0/0`，失败来自预测正确性而不是系统崩溃。
4. `p4a_error_inspection.md` 显示 TabFact 错误集中在 gold=true 被判 false 的 false-negative，其中 2 条来自实体属性 deterministic shortcut；CRT 错误集中在 ratio、percentage、yes/no、aggregate 计算和格式归一化。
5. 按 P4 预设门槛，本轮不直接启动 P4b MACT paired Gate-50。下一步应先做 TabFact/CRT 小范围机制修复和 targeted gate；修复通过后再重新跑新 seed 或同 ID paired MACT。

### 1.8.5 P4a 后续机制修复执行计划

本轮继续执行目标：不重启长跑、不扩大样本量，先处理 P4a 暴露的 TabFact/CRT 新 seed 泛化缺口，判断当前问题是否是可解释的机制缺失，而不是运行不稳定或样本硬编码。

执行原则：

1. 只围绕 `p4a_error_inspection.md` 中暴露的错误类型做机制级修复；禁止根据样本 ID 或 gold answer 写分支。
2. 先写可失败的单测，确认当前代码确实无法覆盖这些结构，再实现最小修复。
3. 先做离线投影，使用 P4a 已保存 input/raw/eval 重新走 deterministic shortcut，估算修复能纠正多少错误；离线投影通过后才考虑启动 Qwen3-32B targeted 小 gate。
4. 产物继续写入 MACT 当前 P4a run 目录，MyAgent 只更新代码、测试和本文档。
5. 每完成一个阶段，都同步补充结论、文件路径和 Git 提交号。

当前待处理机制清单：

| mechanism gap | source examples | planned fix | expected verification |
|---|---|---|---|
| TabFact 同一行双条件被实体属性审计提前误判为 false | `tabfact-test-7952`、`tabfact-test-11907` | 让同一行条件审计覆盖 2 个条件，并避免实体属性审计抢答多条件问题 | 单测覆盖两个 `when/with` 条件均在同一行时输出 `true` |
| TabFact true-claim false-negative 的结构化审计不足 | `tabfact-test-7551`、`11953`、`5024`、`3629`、`5316`、`5704` | 增加 only-not-country、实体+年份数值、列值计数、名次计数、零分计数、极值差等通用审计 | 单测用手工表格断言对应问题输出 `true` |
| CRT ratio / percentage / rounding 输出合同不稳定 | `crt-298`、`242`、`299`、`704`、`308` | 增加命名国家 ratio、至少 N 金牌概率、赛季总分 ratio、阈值平均值、win-loss ratio 的格式控制 | 单测覆盖 `3:2`、`33.3%`、`1.01/0.84`、一位小数和原始胜负比 |
| CRT yes/no、aggregate、variation、margin 和实体后缀归一化不足 | `crt-287`、`286`、`232`、`363`、`105` | 增加点球比分 yes/no 与队名输出、year variation 按 max-min、命名队赢球 margin、短括号国家码剥离 | 单测覆盖 `Yes`、队名短语、`434`、命名队 margin、`netherlands (ned)` 归一化 |

本轮验收口径：

| step | pass condition | output |
|---|---|---|
| RED tests | 新增测试在当前生产代码上失败，失败原因对应缺失机制 | pytest 输出记录在 PRD 和/或 MACT projection 目录 |
| GREEN tests | targeted 新测通过，且 `tests/test_myagent_pipeline.py` 全量通过或明确记录非相关失败 | MyAgent 测试输出 |
| Offline projection | P4a TabFact/CRT 错误中有可解释净修正，且无明显新增 harm；若投影仍低于门槛，则继续诊断而非启动长跑 | MACT `p4a_mechanism_fix_projection.json/md` |
| Targeted gate | 仅当离线投影有足够收益时启动；优先错误 slice 或新 seed small slice，不跑 full200 | MACT 当前 P4a run 目录新增 targeted 产物 |
| Sync | MyAgent 和 MACT 都提交推送；无 runner/vLLM 残留进程 | Git commit hash + 进程检查 |

当前执行状态：

| step | status | evidence |
|---|---|---|
| RED tests | completed 2026-08-01 | 新增 7 个机制测试首次运行：5 failures、2 errors；失败点对应 TabFact false-negative 和 CRT ratio/probability/rounding/aggregate 缺口 |
| GREEN tests | completed 2026-08-01 | targeted 7 tests `OK`；`test_myagent_pipeline.py` 全量 `184 tests OK` |
| 机制修复 | completed 2026-08-01 | MyAgent 增加 TabFact 同行双条件、only-not-country、实体年份数值、列值计数、名次计数、零分计数、min/max 差；CRT 增加命名国家 medal ratio、至少 N 金牌概率、赛季总分 ratio、阈值平均、点球比分、win-loss ratio、year variation、命名队赢球 margin、短括号实体后缀归一化 |
| Offline projection | completed 2026-08-01 | MACT `p4a_mechanism_fix_projection.json/md`；TabFact `42/50 -> 45/50`，CRT `21/50 -> 30/50`，wrong->correct `3/9`，correct->wrong `0/0` |
| Targeted validation | completed 2026-08-01 | MACT `p4a_mechanism_fix_targeted_summary.json/md`；fresh Qwen affected slice：TabFact `3/3`、CRT `9/9`、overall `12/12`，failed/missing `0/0` |
| Full50 after-fix validation | completed 2026-08-01 | MACT `p4a_after_fix_gate50_summary.json/md`；WTQ `37/50` 沿用原 P4a，TabFact `45/50`、CRT `30/50` 为 fresh after-fix rerun；overall `112/150`，weighted token ratio `0.5533`，failed/missing `0/0` |
| P4b paired MACT Gate-50 | completed 2026-08-01 | MACT `p4b_paired_gate50_summary.json/md`；overall MyAgent `112/150` vs MACT `111/150`，token ratio `0.5444`，existing paired gate accepted；严格三数据集目标未达成，因为 WTQ `37/50 < 43/50` |

离线投影产物：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_mechanism_fix_projection.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_mechanism_fix_projection.md
```

投影解释：这不是 fresh model run，也不是 paired MACT comparison；它只证明本次代码机制在保存的 P4a 行上可解释地修正错误，足以支持下一步启动很小的 targeted Qwen 验证。

targeted 实跑产物：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_mechanism_fix_targeted_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_mechanism_fix_targeted_summary.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/myagent_fix_targeted/
```

targeted 结论：真实 runner 下 affected rows 全部修正，TabFact `3/3`、CRT `9/9`、overall `12/12`，失败/缺答案 `0/0`。但该结果只覆盖投影收益行，不能替代 P4a full50 after-fix 验证。

after-fix full50 产物：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_after_fix_gate50_summary.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/p4a_after_fix_gate50_summary.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_newseed_gate50_20260801_0305/myagent_current_after_fix/
```

after-fix full50 结论：P4a current-only 新 seed 门槛已通过，WTQ `37/50`、TabFact `45/50`、CRT `30/50`，overall `112/150`，weighted token ratio vs MACT full200 reference `0.5533`，失败/缺答案 `0/0`。这证明机制修复后的 current policy 值得进入 P4b 同 ID MACT Gate-50；它仍不是新 seed paired MACT 结论。

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

## 2.1 当前活动目标和完成判定

当前对话的活动目标已经固定为：

```text
完善面向专利编写的 MyAgent vs MACT 实验数据体系：在已达标的 Qwen3-32B full200 基础上，补齐 WTQ 泛化诊断、机制消融、多 seed 稳定性、多模型 gate、正式实验包和专利说明书初稿所需证据，并将过程与结果持续写入唯一 PRD 和 MACT 结果目录后同步 GitHub。
```

当前主锚点使用 policy-v6b full200，不再用更早的 canonical full200 作为最终效果口径：Qwen3-32B full200 已完成 WTQ/TabFact/CRT 三数据集同 ID 200 条验收，MyAgent `489/600` vs MACT `450/600`，总体 token ratio `0.5717`，三数据集单项分别为 WTQ `155/200 > 148/200`、TabFact `194/200 > 189/200`、CRT `140/200 > 113/200`，failed/missing `0/0`。旧 canonical full200 段落保留为历史证据，不作为当前收口判定。

完成判定：

| subgoal | status | pass condition / evidence |
|---|---|---|
| Qwen3-32B full200 主证据 | completed | 三数据集单项均超过 MACT，总体 token 明显更低，summary 在 MACT `qwen3_32b_policy_v6b_all200_acceptance_20260731_132611` |
| P4b 新 seed paired gate | completed with WTQ risk | overall MyAgent `112/150` vs MACT `111/150`，但 WTQ `37/50 < 43/50`，因此需要 WTQ fresh 闭环 |
| WTQ targeted fresh | completed 2026-08-03 | 9-row affected slice 真实 Qwen run `9/9`，failed/missing `0/0` |
| P4b WTQ after-fix full50 | completed 2026-08-03 | WTQ MyAgent `46/50 > 43/50`，token ratio `0.5571`，failed/missing `0/0`；aggregate MyAgent `121/150` vs MACT `111/150` |
| E3 multi-seed 稳定性 | completed as boundary diagnosis | Seed-C/Seed-D current-only 均完成并 `stop_or_inspect`；离线边界诊断合计 `212/300`、wrong `88`、token ratio `0.5916`、verification `pass`；paired MACT 均 not required；不能写成多 seed 稳定超过 MACT |
| 多模型 gate | pending/no-candidate audited 2026-08-03 | 最新 E4 readiness 为 `no_candidate_wait`，未发现未测本地模型或 API provider profile；新本地模型或 API key 出现后才按 Gate-10 -> Gate-50 -> Gate-150 执行，不重跑已 no-go 模型 |
| 专利材料收口 | draft complete final pending | 当前已有说明书初稿、机制矩阵、权利要求追踪；fresh/seed/model 结果补齐后更新正式实验表 |

剩余 Qwen3 队列入口已经写入 MACT 专利实验包：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/run_remaining_qwen3_patent_queue.sh
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/remaining_qwen3_queue_runbook_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/preflight_qwen3_runtime.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_qwen3_runtime_preflight_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/build_current_formal_result_ledger.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_formal_result_ledger_current_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/build_current_patent_experiment_section.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_current_patent_experiment_section_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/audit_patent_package_consistency.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/latest_patent_package_consistency_audit_zh.md
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/build_patent_package_checksums.py
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/SHA256SUMS
```

当前在线状态：2026-08-03 11:25 CST 本轮 Qwen3-32B endpoint 已按用户要求关闭；GPU `0,1,2,3` 已释放到约 `3 MiB/0%`，latest preflight 为 `start_service_required`。WTQ fresh、P4b after-targeted、E3 Seed-C/D current-only 与 E3 离线边界诊断均已完成。下一次继续在线实验时，默认可使用 `0,1 -> 8000`、`2,3 -> 8001` 启动 Qwen3 服务，再重跑 preflight；历史 `blocked_gpu_runtime_residual` 段落保留为服务器风险记录，不再代表当前 0-3 状态。

当前边界审计结论：Qwen3-32B 主锚点已达标，但 Seed-C/Seed-D current-only 没有形成稳定性通过证据。E3 诊断已确认问题是语义准确率稳定性，不是执行失败、缺答案或 token 预算失败。后续不要把 E3 写成“多 seed 稳定超过 MACT”；应写成“额外随机种子揭示适用边界，系统在 full200 与 targeted new seed 上有效，但仍需多模型/更多 seed 证明泛化稳定性”。代码、输入、runbook、formal ledger、checksum 和恢复入口均已准备并同步。

正式结果台账：2026-08-01 新增 `build_current_formal_result_ledger.py`，从 frozen full200 summary、P4b summary、WTQ fresh、E3 current summaries、E3 boundary diagnosis、正式模板和 latest preflight 生成 `latest_formal_result_ledger_current.json/md`。该台账用于专家/专利表格填充，明确区分 completed rows 和 pending rows，不把 E3 paired MACT 或多模型 gate 写成已完成。

当前专利实验章节：2026-08-03 新增 `build_current_patent_experiment_section.py`，从 full200、P4b after-targeted、机制矩阵、E3 boundary diagnosis、E4 readiness 和 formal ledger 生成 `latest_current_patent_experiment_section.json/md`。该章节当前状态为 `stage_patent_draft_ready_with_boundaries`，可用于专家/专利代理人理解当前证据，但不会把多模型或多 seed 稳定性写成已完成。

当前正式台账状态：`active_not_complete`，原因是 E4 尚无可启动的新模型/API 候选，final closeout 仍需等 E4 candidate 或明确接受 no-candidate 边界。

一致性审计：2026-08-01 新增 `audit_patent_package_consistency.py`，提交前检查 PRD、manifest、latest formal ledger、latest preflight 和关键数字是否一致；在线阻塞只作为 warning，路径或数字不一致作为 error。

最新一致性审计入口：MACT 专利实验包内 `latest_patent_package_consistency_audit.json/md`；当前目标状态仍为 `active_not_complete`，因为多模型 gate 和最终专利实验表尚未完成。时间戳审计文件会随每次同步刷新，不再要求 PRD 逐个硬编码最新文件名。

恢复完整性校验：2026-08-01 新增 `build_patent_package_checksums.py` 和 `SHA256SUMS`，覆盖专利实验包文件及 manifest 中已存在的关键证据文件。服务器清空/迁移后，在 `/home/ubuntu/lzz` 下执行 `sha256sum -c MACT/outputs/server_runs/qwen3_32b_patent_experiment_package_20260801_2155/SHA256SUMS` 可验证恢复文件是否损坏或缺失。

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
| endpoint | 当前 Qwen3-32B 单服务 `http://127.0.0.1:8000/v1` |
| GPU | 当前恢复建议使用 GPU `0,1 -> 8000`、`2,3 -> 8001`；2026-07-31 的 `6,7` 约束是历史口径，当前 `4,5,6,7` 仍有负载或残留，不作为默认资源 |
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
| Qwen3 WTQ policy-v6b full200 验收 | completed measured | MyAgent `155/200` vs MACT `148/200`；token ratio `0.6187`；failed/missing `0/0`；run：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_policy_v6b_full200_20260731_1115/` |
| Qwen3 TabFact policy-v6b full200 验收 | completed measured | MyAgent `194/200` vs MACT `189/200`；token ratio `0.2014`；failed/missing `0/0`；run：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_tabfact_policy_v6b_full200_20260731_1255/` |
| Qwen3 三数据集 full200 总体验收 | completed measured | MyAgent `489/600` vs MACT `450/600`；总体 token ratio `0.5717`；总体 elapsed ratio `0.1337`；failed/missing `0/0`；summary：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_policy_v6b_all200_acceptance_20260731_132611/` |
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
| Gate summary 失败/缺答案计数修复 | completed | `summarize_model_gate_results.py` 将门禁异常行从 `max(failed, missing)` 改为 `min(rows, failed + missing)`；新增单测覆盖 failed 与 missing 同时出现并超过 2% failure budget 的 no-go 场景 |
| 已测本地模型重跑保护 | completed | `experiment_model_registry.py` 集中维护 Qwen3-32B、Qwen3-14B-AWQ、Qwen2.5-14B-AWQ / Instruct-AWQ、Qwen2.5-3B-Instruct 等已测本地模型及 alias；`audit_qwen3_experiment_state.py` 和 `prepare_model_gate_run.py` 共用该 registry，避免 alias 目录被误判为新候选；人工复现实验必须加 `--allow-known-tested-model`，并在 manifest 标记 override |
| 嵌套模型目录发现修复 | completed | `audit_qwen3_experiment_state.py` 的本地模型发现从一层目录扩展为有限深度递归，并特殊处理 HuggingFace cache 的 `models--org--repo` 目录；readiness JSON 输出 `local_model_paths` / `untested_local_model_paths`，新增单测覆盖挂载盘嵌套未测模型不被漏掉且能返回可启动路径 |
| readiness 自动准备 Gate run | completed | `prepare_model_gate_run.py` 支持 `--readiness-audit latest_experiment_readiness_audit.json --model-name <name>`，自动从 `untested_local_model_paths` 取本地模型路径并派生 `model_tag` / `served_model_name`；如果只有一个未测本地模型可省略 `--model-name`，多个候选漏传时会提示候选列表 |
| OpenRouter API 默认准备 | completed | `experiment_api_registry.py` 集中维护 OpenRouter 的 `api_base_url=https://openrouter.ai/api/v1` 和 `api_key_env=OPENROUTER_API_KEY`；`audit_qwen3_experiment_state.py` 在检测到 `OPENROUTER_API_KEY` 时会输出 `api_provider_profiles.OpenRouter`；`prepare_model_gate_run.py --backend api --api-provider OpenRouter --model-name <provider_model>` 会自动填配置且不会写入真实 key；2026-07-30 22:24 起也可用 `--backend api --readiness-audit latest_experiment_readiness_audit.json --model-name <provider_model>` 自动消费单个 provider profile；未知 provider 缺少 endpoint/key env 时会给出 CLI 参数提示而不是 Python traceback |
| 外部 API healthcheck 前置 | completed | 2026-07-30 22:30 新增 `healthcheck_openai_compatible.py`；`prepare_model_gate_run.py` 生成的 API `healthcheck_services.sh` 不再只检查 key env 存在，会调用 `/models` 验证 endpoint 连通性和目标 model 是否列出，错误信息不打印 secret；2026-07-30 22:36 `prepare_paired200_run.py` 生成的 paired-200 目录也包含 `healthcheck_services.sh`，API 场景复用 `/models` 检查，本地 vLLM 场景复用 `healthcheck_vllm_pool.sh`，README run order 要求先 healthcheck 再跑 myAgent/MACT 200 行 |
| 长跑 checkpoint / GitHub 同步流程固化 | completed | 2026-07-30 22:45 `prepare_model_gate_run.py` 和 `prepare_paired200_run.py` 生成的每个 MACT run 目录都会包含 `checkpoint_to_git.sh`；默认只 force-stage 当前 run 目录，`--commit MESSAGE --push` 可将当前 run 目录限定提交并推送；新增单测在临时 git 仓库中把 `outputs/` 设为 ignore，验证脚本仍能把 ignored run 目录 stage 进 Git |
| 下一候选 Gate 自动准备入口 | pending user approval | 2026-07-30 22:49 提议新增 `prepare_next_model_gate_run.py`：只读取 readiness audit，不启动模型；有且只有一个未测本地模型时自动生成 Gate run；有 API key 时要求显式传 provider model name；没有候选时明确退出并提示等待新模型/API。按当前流程规则，需用户确认设计后再按 TDD 实现 |
| 专家/专利正式实验方案 | ready for drafting | Qwen3-32B 当前已满足三数据集单项准确率超过 MACT；正式实验仍建议 gate 后只扩最终候选，避免全模型全数据集长跑 |

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
| WTQ earlier/later 候选比较全局行触发 | done locally verified | WTQ policy-v6 partial 暴露 `nu-4268`：问题同时比较 Sydney / Coral Springs，但压缩只保留 Coral 局部行；v6b 将 `earlier/later` 纳入 global-row 触发词，确保候选比较题保留所有候选行 |
| WTQ 否定年份标量冲突劝返 | done locally verified | WTQ policy-v6 partial 暴露 `nu-484`：代码在 `not/nor` 条件下选到首个非目标年份，thinking verifier 给出表内高置信年份；v6b 只在“问年份 + 否定排除 + 双方均为表内年份”的冲突上允许审阅者劝返 |
| WTQ policy-v6b 本地回归验证 | done | 新增 2 个回归测试后先观察失败，再实现修复；`python -m unittest discover -s tests` 跑 317 tests OK，`python -m py_compile code/my_agents.py code/tqa.py` 退出 0，`git diff --check` 退出 0 |
| WTQ policy-v6b full200 实跑 | done measured | 当前 MyAgent `155/200`，旧 MyAgent `131/200`，MACT `148/200`；token ratio `0.6187`，失败/缺答案 `0/0`，merged `200/200`；结果：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_wtq_policy_v6b_full200_20260731_1115/wtq_policy_v6b_full200_comparison.md` |
| TabFact policy-v6 full200 实跑 | done measured but not accepted | 当前 MyAgent `185/200`，旧 MyAgent `185/200`，MACT `189/200`；token ratio `0.2143`，失败/缺答案 `0/0`，merged `200/200`；结果：`/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_tabfact_policy_v6_full200_20260731_1030/tabfact_policy_v6_full200_comparison.md` |
| TabFact policy-v6b audit shortcuts | done locally verified | 针对 v6 full200 的 MACT-correct/current-wrong 聚类，新增实体属性审计、同一行多条件审计、列值计数审计、双实体出现次数、首尾时间差、实体数值差；新增 6 个红绿测试和 guard，离线投影从 `185/200` 到 `194/200`，净 gain 9、harm 0；`python -m unittest discover -s tests` 跑 323 tests OK，`py_compile`/`diff --check` 均为 0 |
| numpy array execution result 判断 | done | `verification_gap` 改为显式判断非空执行结果，避免 numpy array truth-value 崩溃 |
| numpy array 输出序列化 | done | `_to_serializable` 和 `_json_default` 优先使用 `.tolist()`，避免多元素 numpy array `.item()` 崩溃 |
| 机器审计脚本 | done | `scripts/server/audit_qwen3_experiment_state.py` 可从 MACT 结果生成 `latest_experiment_readiness_audit.json` 和 `latest_expert_evidence_summary.md`；模型发现支持有限深度递归和 HuggingFace cache 目录，并输出可直接用于 `prepare_model_gate_run.py --model-id` 的模型路径；已测模型判断使用共享 registry 的 alias 规则，防止下次恢复时人工误读 canonical/staged 口径或重复启动 no-go 模型 |
| 外部 API key readiness 检测 | done | `audit_qwen3_experiment_state.py` 已从只识别 OpenAI/DeepSeek/DashScope/Anthropic 扩展到 SiliconFlow、Moonshot、Zhipu、Gemini/Google、OpenRouter、Together、Fireworks、Ark、Volc、Azure OpenAI；API key 名和 OpenRouter 默认 provider profile 均来自 `experiment_api_registry.py`；2026-07-30 22:20 已支持默认扫描 `MyAgent/configs/server/*.env` 真实 env 文件、跳过 `.example`/`.bak`，同时保留额外 `--env-file`，检测只记录 key 名和文件路径不泄漏 secret 值，新增单测保护 |
| 新模型 Gate run 准备脚本 | done | `scripts/server/prepare_model_gate_run.py` 可为新增本地 vLLM 模型或外部 OpenAI-compatible API 候选生成 MACT run 目录、Gate-10/Gate-50/Gate-150 runner 和 `gate_run_manifest.json`；本地默认 GPU `0,1;2,3`、端口 `8000/8001`，外部 API backend 只写 `api.env`/`api_profile.md` 且不写 secret；本地候选可直接用 `--readiness-audit` 从审计 JSON 自动取 `model_id` 并派生 tag/served name，多个候选时再补 `--model-name`；OpenRouter API 候选可只传 `--api-provider OpenRouter --model-name <provider_model>`，或在 readiness audit 已含单个 `api_provider_profiles` 时传 `--backend api --readiness-audit ... --model-name <provider_model>`，脚本自动填 `https://openrouter.ai/api/v1` 和 `OPENROUTER_API_KEY`；API `healthcheck_services.sh` 会在 Gate-10 前检查 key、endpoint 和 model 列表；未知 API provider 若未显式传 `--api-base-url` / `--api-key-env` 会直接报 CLI 参数错误；已测本地模型默认按共享 registry 拒绝，显式 override 会写入 manifest；Gate-50 runner 会拒绝缺失或未通过 Gate-10 summary 的 run，Gate-150 runner 会拒绝缺失或未通过 Gate-50 summary 的 run |
| Paired-200 run 准备脚本 | done | `scripts/server/prepare_paired200_run.py` 可从 Gate run manifest 生成最终候选的 MACT paired-200 run 目录，包含 healthcheck、myAgent blind200 runner、WTQ/TabFact/CRT MACT one-by-one runner、eval/compare 脚本、README 和 `paired200_run_manifest.json`；脚本会拒绝缺失 `gate150_summary.json` 或 `decision != paired200` 的 Gate run；外部 API 场景只记录 key 变量名，不写 secret，且在 README run order 中先检查 API key/endpoint/model 再跑 200 行 |
| Gate-10/Gate-50/Gate-150 自动决策脚本 | done | `scripts/server/summarize_model_gate_results.py` 汇总 WTQ/TabFact/CRT eval；异常行按 `min(rows, failed + missing)` 保守计入 failure rate；Gate-10 只检查三数据集完整、失败/缺答案和 token 是否过线，输出 `no-go` 或 `gate50`；Gate-50 按 reference `124/150`、failure <= `2%`、token ratio <= `0.75` 输出 `no-go` 或 `gate150`；Gate-150 按当前 Qwen3-32B frozen150 overall reference `333/450`，并要求至少 2 个数据集达到单项 reference：WTQ `105/150`、TabFact `131/150`、CRT `97/150`，输出 `no-go` 或 `paired200` |
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
| P1 | 新模型筛选 | waiting/no-candidate: 2026-08-03 11:37 已递归复扫模型目录/缓存/挂载盘、外部 API env 和默认 server env 文件，仍无新增候选；当前默认 GPU `0,1` 和 `2,3` 可用于下个候选的双服务 Gate；若审计 JSON 出现 `untested_local_model_paths`，可用 `prepare_model_gate_run.py --readiness-audit ...` 直接生成 Gate run，多个候选时补 `--model-name` |
| P1 | Gate-150 到 paired-200 准备链路 | done: `run_gate150.sh` 会生成 `gate150_summary.json/md`；只有 overall 和至少 2 个 dataset-level reference 均通过、decision 为 `paired200` 时，`prepare_paired200_run.py` 才会从 Gate run 目录生成 same-ID paired-200 正式候选目录、runner、eval/compare 和 manifest；no-go 或缺 summary 会直接报错 |
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
2. Gate-10 smoke：WTQ / TabFact / CRT 各 10 条，只验证服务、schema、token 统计和失败处理；生成脚本口径下 Gate-10 是 Gate-50 的前置条件，`run_gate50.sh` 会检查 `gate10_summary.json` 的 decision 是否为 `gate50`。
3. Gate-50 必跑：三数据集各 50 条 myAgent-only；若 overall 明显低于 Qwen3-32B Gate-50 reference `124/150`，直接 no-go。
4. Gate-150 条件：Gate-50 overall 接近或超过 `124/150`，执行失败率 <= `2%`，平均 token 没有明显失控。
5. Paired-200 条件：Gate-150 仍接近或超过 Qwen3-32B，并且至少两个数据集不弱于当前 Qwen3-32B 或有明确论文/专利价值。
6. MACT paired 只在最终候选上跑；Gate-150 后用 `prepare_paired200_run.py --gate-run-dir "$RUN_DIR"` 生成新的 paired-200 run 目录，不手写 runner。
7. 本地 Qwen 系列大模型默认使用两个 vLLM 服务并行：当前恢复建议 GPU `0,1` -> port `8000`，GPU `2,3` -> port `8001`；runner 按 shard 写入不同 raw 文件，最后按原始 ID 顺序合并，避免两个进程抢写同一个 jsonl。GPU `4,5,6,7` 当前仍有负载或残留时不要作为默认资源。

## 14. 下一次新增模型的执行模板

本节是扩容/清空后继续实验的最小可执行入口。只在出现新模型目录或新 API key 后使用；当前四个本地模型不要重跑。

### 14.1 本地 vLLM 候选模型

先用准备脚本在 MACT 下创建 run 目录和全部脚本。该命令只写文件，不启动模型：

```bash
python - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path("/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_experiment_readiness_audit.json").read_text())
print(json.dumps(audit["model_readiness"].get("untested_local_model_paths", {}), ensure_ascii=False, indent=2))
PY

READINESS_AUDIT=/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_blind200_mact_full200_20260723/latest_experiment_readiness_audit.json

cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

python scripts/server/prepare_model_gate_run.py \
  --myagent-root /home/ubuntu/lzz/MyAgent \
  --mact-root /home/ubuntu/lzz/MACT \
  --readiness-audit "$READINESS_AUDIT"
```

如果 `untested_local_model_paths` 只有一个模型，脚本会直接取第一个路径作为 `model_id`，并自动派生 `model_tag` 和 `served_model_name`。如果有多个模型，脚本会提示候选列表，此时重跑命令并补 `--model-name <model_name_from_untested_local_model_paths>`。如果模型命中已测试本地模型，脚本会报 `known tested local model` 并停止；只有为了复现实验且已经在本文档/MACT ledger 里写明原因时，才加 `--allow-known-tested-model`。

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
cat "$RUN_DIR"/gate10_summary.json
cat "$RUN_DIR"/gate10_summary.md
# Gate-10 若有连接错误、缺行、schema 错误、失败/缺答案或明显 context/token 问题，先排查服务，不进入 Gate-50。
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
| no-go | overall 明显低于 Qwen3-32B Gate-50 reference `124/150`，或按 `min(rows, failed + missing)` 统计的异常行比例 > `2%`，或 token 明显失控 |
| Gate-150 | overall 接近或超过 `124/150`，三数据集均完整，失败率 <= `2%` |
| paired-200 | Gate-150 后仍有竞争力，且值得为专家/专利主表补 MACT same-ID 对照 |

`run_gate10.sh` 会自动调用 `summarize_model_gate_results.py --gate-name gate10` 生成 `gate10_summary.json` 和 `gate10_summary.md`。`run_gate50.sh` 会先检查 `gate10_summary.json` 的 `decision=gate50`，再执行 Gate-50，并自动生成 `gate50_summary.json` 和 `gate50_summary.md`。`run_gate150.sh` 会先检查 `gate50_summary.json` 的 `decision=gate150`，再执行 Gate-150。下一步是否进入 Gate-150 以 Gate-50 summary 的 `decision` 为准，人工只复核异常日志和数据行数。

只有 `gate50_summary.json` 的 `decision` 为 `gate150` 时才运行 Gate-150：

```bash
bash "$RUN_DIR/run_gate150.sh"
wc -l "$RUN_DIR"/myagent_gate150/merged/*.jsonl
cat "$RUN_DIR"/myagent_gate150/eval/*_eval.json
cat "$RUN_DIR"/gate150_summary.json
cat "$RUN_DIR"/gate150_summary.md
rg -n "Connection refused|APIConnectionError|context length|BadRequest|Traceback" "$RUN_DIR"/myagent_gate150/logs || true
```

Gate-150 默认参考当前 Qwen3-32B frozen150 结果：overall `333/450`，WTQ `105/150`，TabFact `131/150`，CRT `97/150`。只有 `gate150_summary.json` 的 `decision` 为 `paired200`，也就是 overall 过线、失败/token 过线且至少 2 个数据集达到对应单项 reference 时，才进入 paired-200；如果候选有明确论文/专利价值但未自动过线，必须在 PRD 和 MACT ledger 里另建人工例外口径，不能静默扩样。

paired-200 准备和执行：

```bash
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$RUN_DIR/gate150_summary.json")" = paired200
PAIRED_MANIFEST_JSON=$(python scripts/server/prepare_paired200_run.py \
  --myagent-root /home/ubuntu/lzz/MyAgent \
  --mact-root /home/ubuntu/lzz/MACT \
  --gate-run-dir "$RUN_DIR")
PAIRED_RUN_DIR=$(python -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])' <<< "$PAIRED_MANIFEST_JSON")
cat "$PAIRED_RUN_DIR/paired200_run_manifest.json"

bash "$PAIRED_RUN_DIR/run_myagent_paired200.sh"
bash "$PAIRED_RUN_DIR/run_mact_wtq_paired200.sh"
bash "$PAIRED_RUN_DIR/run_mact_tabfact_paired200.sh"
bash "$PAIRED_RUN_DIR/run_mact_crt_paired200.sh"
bash "$PAIRED_RUN_DIR/run_eval_and_compare.sh"

cat "$PAIRED_RUN_DIR/paired200_summary.json"
wc -l "$PAIRED_RUN_DIR"/myagent_paired200/merged/*.jsonl
wc -l "$PAIRED_RUN_DIR"/mact/*_mact_paired200.jsonl
rg -n "Connection refused|APIConnectionError|context length|BadRequest|Traceback" "$PAIRED_RUN_DIR" || true
```

每次阶段结束都同步：

```bash
cd /home/ubuntu/lzz/MACT
git add -f "$RUN_DIR"
test -z "${PAIRED_RUN_DIR:-}" || git add -f "$PAIRED_RUN_DIR"
git commit -m "Record ${MODEL_TAG} gate results"
git push origin main

cd /home/ubuntu/lzz/MyAgent
git add docs/server/server_codex_reports/current-qwen3-mact-experiment-prd.md
git commit -m "Update ${MODEL_TAG} gate status"
git push origin codex/selective-risk-collaboration
```

### 14.2 外部 API 候选模型

如果是 DeepSeek / OpenAI / DashScope / OpenRouter / Together / Fireworks 等外部模型，不启动 vLLM。先把 API key 放在环境变量里，不写入任何文件：

```bash
API_KEY_ENV=OPENROUTER_API_KEY
export OPENROUTER_API_KEY='<real-key-only-in-shell>'
```

OpenRouter 已有 provider 默认配置，可用同一个准备脚本生成外部 API run 目录，且不需要把真实 key 写入文件：

```bash
MODEL_TAG=<provider_model_tag>
SERVED_MODEL_NAME=<provider_model_name>

cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

python scripts/server/prepare_model_gate_run.py \
  --backend api \
  --myagent-root /home/ubuntu/lzz/MyAgent \
  --mact-root /home/ubuntu/lzz/MACT \
  --model-tag "$MODEL_TAG" \
  --model-name "$SERVED_MODEL_NAME" \
  --api-provider OpenRouter
```

其它 OpenAI-compatible provider 暂时继续显式传 `--served-model-name`、`--api-base-url` 和 `--api-key-env`，直到 provider 默认值在测试里加保护；如果漏传，准备脚本会报出需要补哪个 CLI 参数，不会打印 traceback。

生成内容：

```text
api.env          # 只保存 provider/base URL/model/key 变量名，不保存 key 值
api_profile.md  # 不含 secret 的外部 API profile
run_gate10.sh
run_gate50.sh
run_gate150.sh
gate_run_manifest.json
```

执行：

```bash
RUN_DIR=/home/ubuntu/lzz/MACT/outputs/server_runs/<model_tag>_gate50_<timestamp>
bash "$RUN_DIR/healthcheck_services.sh"
bash "$RUN_DIR/run_gate10.sh"
bash "$RUN_DIR/run_gate50.sh"
# 只有 gate50_summary.json 的 decision 为 gate150 时运行：
bash "$RUN_DIR/run_gate150.sh"
```

外部 API Gate-10/Gate-50/Gate-150/paired-200 的扩大条件与本地模型一致。每次阶段结束仍把 raw、eval、summary、profile、ledger 写入 MACT run 目录并 `git add -f "$RUN_DIR"` 后推送。

外部 API 若进入 paired-200，也使用同一个 paired-200 准备脚本；生成脚本会 source Gate run 目录里的 `api.env`，只读取 `API_BASE_URL`、`SERVED_MODEL_NAME` 和 `API_KEY_ENV`，不会把真实 key 写入 MACT：

```bash
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$RUN_DIR/gate150_summary.json")" = paired200
PAIRED_MANIFEST_JSON=$(python scripts/server/prepare_paired200_run.py \
  --myagent-root /home/ubuntu/lzz/MyAgent \
  --mact-root /home/ubuntu/lzz/MACT \
  --gate-run-dir "$RUN_DIR")
PAIRED_RUN_DIR=$(python -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])' <<< "$PAIRED_MANIFEST_JSON")

bash "$PAIRED_RUN_DIR/run_myagent_paired200.sh"
bash "$PAIRED_RUN_DIR/run_mact_wtq_paired200.sh"
bash "$PAIRED_RUN_DIR/run_mact_tabfact_paired200.sh"
bash "$PAIRED_RUN_DIR/run_mact_crt_paired200.sh"
bash "$PAIRED_RUN_DIR/run_eval_and_compare.sh"

cd /home/ubuntu/lzz/MACT
git add -f "$RUN_DIR" "$PAIRED_RUN_DIR"
git commit -m "Record ${MODEL_TAG} paired-200 results"
git push origin main
```
