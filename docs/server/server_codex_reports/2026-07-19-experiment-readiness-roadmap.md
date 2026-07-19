# 2026-07-19 Experiment Readiness and Roadmap

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
报告起点提交：`411e01e2b61af88d873245505745c28999d86dfb` (`Report Qwen3 policy v5 200 validation`)
关键代码基线：`33663ab42366000fd94bc884f0f3d200d71bf756` (`Improve TabFact deterministic verification`)

## 1. 我对目标的理解

这个项目当前不是继续追求极限省 token，而是要证明 myAgent 的风险自适应协作策略在同模型、同数据、同 evaluator 口径下，可以达到或超过 MACT 的总体准确率，同时平均 API token 明显低于 MACT。用户给出的可接受 token 尺度是总体约为 MACT 的 70% 左右。

专利/论文中应强调的是：

- 输入适配、answer contract 和 evaluator 统一，避免评测口径偏差。
- 风险评分、证据包、压缩、deterministic semantic operators、选择性强校验和 replan 共同决定 token 投入。
- 高风险题增加协作和验证，低风险题使用确定性 shortcut 或轻量路径。
- 结论必须来自同模型、同 frozen split、同 evaluator 的 myAgent vs MACT 配对实验。

不应写成“所有任务所有设置全面稳定超过 MACT”，除非后续配对实验真正支持。

## 2. 当前状态

已完成当前 myAgent policy v5 在 Qwen3-32B 上 WTQ、TabFact、CRT 各 200 条验证。报告见：

```text
docs/server/server_codex_reports/2026-07-19-qwen3-policy-v5-200-all.md
```

核心结果：

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ | 134/200 | 0.670 | 11,317.45 | 20.860 | 0 | 0 |
| TabFact | 166/200 | 0.830 | 2,680.65 | 12.833 | 0 | 0 |
| CRT | 129/200 | 0.645 | 9,739.92 | 22.179 | 0 | 0 |
| Overall | 429/600 | 0.715 | 7,912.67 | 18.624 | 0 | 0 |

当前可用 Qwen3 MACT 参考只覆盖 50 条/数据集：

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ | 23/50 | 0.460 | 10,946.00 | 97.567 | 1 | 3 |
| TabFact | 44/50 | 0.880 | 11,051.98 | 118.543 | 0 | 0 |
| CRT | 33/50 | 0.660 | 12,384.26 | 151.058 | 0 | 0 |
| Overall | 100/150 | 0.667 | 11,460.75 | 122.389 | 1 | 3 |

基于这个参考，myAgent 当前总体 accuracy 是 71.50%，高于 MACT 50 参考的 66.67%；总体 avg tokens 是 MACT 参考的 69.04%。但这还不是严格的 200 条同 split 配对结论。

## 3. 之前问题是否还存在

结论：主要工程问题当前没有复现；另发现并修复了一个服务器路径兼容问题。

| 问题 | 当前状态 | 证据 |
|---|---|---|
| vLLM 未就绪导致 `Connection refused` | 当前未复现 | 2026-07-19 重新运行 `healthcheck_vllm_pool.sh`，`qwen3-32b-local` 返回 `ok` |
| `tqa.py` 吞异常导致外层只看到缺输出文件 | 已修复 | `b994f5a Propagate TQA shard failures`，`tests/test_tqa_failure_exit.py` 通过 |
| TabFact strong verification 过度消耗 token | 当前 policy v5 已避开 | TabFact 200 中 strong applied 为 0/200，avg tokens 2,680.65 |
| TabFact accuracy 偏弱 | 有改善但不能说完全解决 | 当前 TabFact 200 为 0.830，低于现有 MACT 50 参考 0.880 |
| dataset adapter 和 MACT smoke 测试使用旧路径 | 已修复 | 新增对 `dataset/WikiTableQuestions`、`dataset/Table-Fact-Checking`、`MACT/code` 的 fallback |
| 输出缺行或 shard 静默失败 | 当前未复现 | WTQ/TabFact/CRT raw、merged、eval 都是 200 行/样本，失败数 0 |

已通过的验证：

```text
python tests/test_tqa_failure_exit.py
python tests/test_evaluate_results.py
python tests/test_compare_blind_results.py
python tests/test_calibrate_risk_policy.py
python tests/test_dataset_profiles.py
python tests/test_dataset_adapters.py
python tests/test_task_modes.py
python tests/test_model_backends.py
python tests/test_myagent_pipeline.py
python tests/test_answer_contracts.py
python tests/test_blind_holdout.py
python tests/test_evidence_builder.py
python tests/test_risk_control.py
python tests/test_sample_benchmark.py
python tests/test_selective_collaboration.py
python tests/test_mact_smoke.py
python -m py_compile code/tqa.py code/my_agents.py code/dataset_adapters.py scripts/server/run_sharded_tqa.py code/evaluate_results.py code/compare_blind_results.py
```

## 4. 当前是否符合要求

工程可运行性：基本符合。当前 runner 能跑通三数据集 200 条，输出 raw、merged、eval 完整，失败数为 0，核心单测通过，vLLM 服务可健康响应。

阶段性实验判断：可以进入下一阶段。当前 Qwen3 myAgent 的总体准确率高于已有 Qwen3 MACT 50 参考，token 也在约 70% MACT 的目标线内。

正式结论判断：还不能最终确认。原因是缺少同一 frozen split 上的 MACT 100/150/200 配对结果。尤其是 TabFact 和 CRT 在现有 50 条 MACT 参考上分别低 5.0 pp 和 1.5 pp，因此不能只用总体数值写成“各数据集都超过 MACT”。

## 5. 为什么不建议全量跑

全量数据量：

```text
WTQ       4,344
TabFact  12,779
CRT         728
Total    17,851
```

按当前 Qwen3 wall time 粗估：

| run scope | estimated wall time |
|---|---:|
| myAgent full WTQ | 25.17 h |
| myAgent full TabFact | 45.55 h |
| myAgent full CRT | 4.49 h |
| myAgent full all | 75.21 h |
| MACT full WTQ | 117.73 h |
| MACT full TabFact | 420.80 h |
| MACT full CRT | 30.55 h |
| MACT full all | 569.07 h |

全量 myAgent 约 3.1 天，全量 MACT 约 23.7 天。这个成本不适合作为模型筛选或常规迭代，也不适合作为每改一次代码就重跑的验证方式。

## 6. 推荐实验路线

### Gate 0: 固定代码和服务

冻结当前代码提交，不再针对 TabFact 继续局部调参。每次实验前记录：

```text
git rev-parse HEAD
vLLM model id
env file
dataset JSONL sha256
run command
start/end time
raw rows
merged rows
eval num_samples
avg prompt/completion/total tokens
failed/missing
```

### Gate 1: 新模型 smoke

每个候选模型先跑 myAgent 20 或 50 条/数据集，只判断服务、输出完整性、失败数、token 是否异常。这个阶段不写正式结论。

myAgent 示例：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/<model>_policy_v5_smoke50_all \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

### Gate 2: 同 split 配对筛选

对有希望的模型跑 myAgent 和 MACT 同一批 50 条/数据集。Qwen3 风格速度下预计 myAgent 加 MACT 共约 5.9 小时。

筛选标准：

- Overall accuracy 不低于 MACT。
- Overall avg total tokens 不高于 MACT 的 75%，目标约 70%。
- failed rate 不高于 2%。
- 任一数据集 accuracy 不比 MACT 低超过 5 pp。
- 至少两个数据集 accuracy 不低于 MACT。

### Gate 3: 正式前确认

用 frozen 100 或 150 条/数据集做配对确认。Qwen3 风格速度下预计：

| sample size per dataset | myAgent + MACT estimated wall time |
|---:|---:|
| 100 | 11.75 h |
| 150 | 17.63 h |
| 200 | 23.50 h |

服务器时间紧张时，建议先跑 100 条/数据集配对。如果过线，再扩到 150 或 200。已经完成的 myAgent 200 结果可以作为阶段参考，但正式配对最好用同一个 frozen sample 目录。

### Gate 4: 正式报告

最终建议用 150 或 200 条/数据集作为正式主结果；每个候选模型不要都跑正式规模，只给最有希望的 1 到 2 个模型跑。正式报告至少包含：

- myAgent vs MACT per dataset 和 overall accuracy。
- avg prompt/completion/total tokens、llm calls、elapsed seconds。
- raw rows、merged rows、eval num_samples、failed、missing。
- risk strata、route distribution、strong applied、shortcut 数。
- McNemar 或 paired exact test，Wilson interval，token ratio。
- 失败样本分类：adapter/gold/evaluator/routing/compression/reasoning/code execution。

### Gate 5: 消融实验

消融不要全量跑。建议每个设置 50 或 100 条/数据集，且只在最终选定模型上做：

- legacy collaboration。
- no-strong verification。
- no deterministic shortcuts。
- max-replan 0/1/2。
- optional multiview validation。

如果主结果只有 150 或 200 条/数据集，消融用 50 或 100 条/数据集即可支撑机制分析，不需要把所有消融也跑到全量。

## 7. Frozen split 建议

不要用 first-N 作为正式实验集。当前 runner 的 `--limit-per-task` 适合 smoke 和快速 gate，但 TabFact/CRT 可能有同表聚集，正式结论应使用已有脚本冻结 table-diverse holdout。

推荐命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

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

然后让 myAgent 和 MACT 都使用这个目录中的：

```text
datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl
datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl
datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl
datasets_ready/frozen_qwen3_eval_150_2026-07-19/manifest.json
```

注意：如果使用 `--history_root outputs` 排除历史样本，要把正式 frozen split 自己的目录加入 `--ignore_root` 或先输出到新目录，脚本当前会自动 ignore output_dir。

## 8. 配对运行建议

myAgent runner 已支持 `--wtq-dataset/--tabfact-dataset/--crt-dataset` 覆盖默认 full 数据路径。正式 frozen split 可以继续走同一个 shard、merge、eval 和行数检查流程。

myAgent frozen split 示例：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_policy_v5_frozen150 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

MACT Qwen3 单任务示例：

```bash
cd /home/ubuntu/lzz/MACT
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source /home/ubuntu/lzz/MyAgent/configs/server/qwen3_32b_2gpu_local.env

python code/tqa.py \
  --task wtq \
  --dataset_path /home/ubuntu/lzz/MyAgent/datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --output_path outputs/server_runs/qwen3_32b_frozen150/wtq_mact.jsonl \
  --plan_model_name "$SERVED_MODEL_NAME" \
  --code_model_name "$SERVED_MODEL_NAME" \
  --model_provider openai_compatible \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key_env LOCAL_VLLM_API_KEY \
  --thinking disabled \
  --temperature 0 \
  --max_tokens 2048 \
  --plan_sample 1 \
  --code_sample 1 \
  --max_step 3 \
  --max_actual_step 3
```

TabFact 在 MACT 中使用 `--task scitab`，CRT 使用 `--task crt`。MACT 目前仍需要跑后检查输出行数，因为它的 `code/tqa.py` 对样本异常的处理不如 myAgent runner 严格。

## 9. 可写入专利/论文的当前表述

当前可以写：

> 在 Qwen3-32B 本地 OpenAI-compatible 服务上，myAgent policy v5 已能稳定完成 WTQ、TabFact、CRT 三数据集各 200 条验证，失败数为 0。合计 600 条上 primary accuracy 为 71.50%，平均 total tokens 为 7,912.67。相对当前可用 Qwen3 MACT 50 条/数据集参考，myAgent 的总体准确率更高，token 约为 MACT 的 69.04%，说明风险自适应 token 分配和选择性协作机制值得进入配对扩大实验。

当前不应写：

> myAgent 已经在 Qwen3-32B 上正式证明全面超过 MACT。

更严格的正式结论需要下一步同 frozen split 的 MACT 100/150/200 配对结果。

## 10. 下一步推荐

最实际的下一步：

1. 先冻结一个 `150/数据集` 的 table-diverse split。
2. 跑 myAgent 和 MACT 的 Qwen3 配对 150，如果耗时压力大则先做 100。
3. 若 overall accuracy 过线且 token ratio 仍低于 75%，把这个模型作为当前主模型。
4. 其他候选模型只先跑 50/数据集配对 gate，不要每个模型都跑 150 或 200。
5. 选出 1 到 2 个模型后，再跑正式 150/200 和少量消融。
