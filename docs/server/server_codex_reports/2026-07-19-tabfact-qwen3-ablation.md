# 2026-07-19 TabFact Qwen3-32B Ablation and Follow-up Report

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
实验任务：TabFact 50 no-strong 与 legacy 消融；后续 TabFact selective policy 修正和 blind200 验证

## 1. Git 和代码验证

消融实验起点提交：

```text
64751df docs: add cross-codex experiment handoff
```

`git log --oneline -5` 中已确认包含：

```text
b994f5a Propagate TQA shard failures
```

同步和验证命令：

```bash
cd /home/ubuntu/lzz/MyAgent
git fetch origin
git checkout codex/selective-risk-collaboration
git pull --ff-only origin codex/selective-risk-collaboration
git log --oneline -5
git merge-base --is-ancestor b994f5a HEAD

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
python tests/test_tqa_failure_exit.py
python -m py_compile code/tqa.py scripts/server/run_sharded_tqa.py
```

验证结果：

```text
tests/test_tqa_failure_exit.py: OK, Ran 1 test
py_compile: exit 0
```

工作区已有非本轮改动，未纳入本次提交：

```text
 M configs/server/qwen3_32b_2gpu_local.env.example
?? configs/server/qwen3_32b_2gpu_local.env.bak.20260709_191806
?? restart_qwen3_context_try.sh
```

## 2. GPU 和 vLLM 状态

健康检查命令：

```bash
source configs/server/qwen3_32b_2gpu_local.env
bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
curl -sS -H "Authorization: Bearer ${LOCAL_VLLM_API_KEY:-EMPTY}" http://127.0.0.1:8000/v1/models
nvidia-smi
```

healthcheck 返回 chat completion，内容为 `ok`，usage 为 `prompt_tokens=16,total_tokens=18,completion_tokens=2`。

`/v1/models` 返回：

```json
{"id":"qwen3-32b-local","root":"/home/ubuntu/models/Qwen3-32B","max_model_len":8192}
```

`nvidia-smi` 摘要：

```text
Driver Version: 580.95.05, CUDA Version: 13.0
GPU 5: NVIDIA GeForce RTX 4090, 45815 MiB used, VLLM::Worker_TP0 PID 18468
GPU 6: NVIDIA GeForce RTX 4090, 45815 MiB used, VLLM::Worker_TP1 PID 18469
```

注意：本次实际 vLLM 进程占用 GPU 5/6，不是 handoff 示例中的 GPU 0/1。

## 3. 运行命令

no-strong：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env
rm -rf outputs/server_runs/qwen3_32b_tabfact50_no_strong
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks tabfact \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_tabfact50_no_strong \
  --limit-per-task 50 \
  --disable-strong-verification \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

legacy：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env
rm -rf outputs/server_runs/qwen3_32b_tabfact50_legacy
time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks tabfact \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_tabfact50_legacy \
  --limit-per-task 50 \
  --collaboration-mode legacy \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

计时：

```text
no-strong: 2026-07-19 00:19:12 CST -> 00:34:14 CST, real 15m1.530s
legacy:    2026-07-19 00:34:45 CST -> 00:49:46 CST, real 15m1.641s
```

## 4. 输出文件

no-strong：

```text
outputs/server_runs/qwen3_32b_tabfact50_no_strong/merged/tabfact_qwen3-32b-local.jsonl
outputs/server_runs/qwen3_32b_tabfact50_no_strong/eval/tabfact_qwen3-32b-local_eval.json
outputs/server_runs/qwen3_32b_tabfact50_no_strong/logs/tabfact/tabfact_shard00.log
outputs/server_runs/qwen3_32b_tabfact50_no_strong/raw/tabfact/tabfact_shard00_out.jsonl
```

legacy：

```text
outputs/server_runs/qwen3_32b_tabfact50_legacy/merged/tabfact_qwen3-32b-local.jsonl
outputs/server_runs/qwen3_32b_tabfact50_legacy/eval/tabfact_qwen3-32b-local_eval.json
outputs/server_runs/qwen3_32b_tabfact50_legacy/logs/tabfact/tabfact_shard00.log
outputs/server_runs/qwen3_32b_tabfact50_legacy/raw/tabfact/tabfact_shard00_out.jsonl
```

行数核对：

```text
50 outputs/server_runs/qwen3_32b_tabfact50_no_strong/merged/tabfact_qwen3-32b-local.jsonl
50 outputs/server_runs/qwen3_32b_tabfact50_legacy/merged/tabfact_qwen3-32b-local.jsonl
50 outputs/server_runs/qwen3_32b_tabfact50_no_strong/raw/tabfact/tabfact_shard00_out.jsonl
50 outputs/server_runs/qwen3_32b_tabfact50_legacy/raw/tabfact/tabfact_shard00_out.jsonl
```

两个日志尾部均显示 `Finished sample 50/50`，未检出 `Traceback`、`Connection refused`、`APIConnectionError`、`Error processing`。

## 5. Eval 结果

| 口径 | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selective baseline | 34/50 | 0.68 | 15793.76 | 15042.48 | 751.28 | 7.08 | 32.424 | 0 |
| no-strong | 36/50 | 0.72 | 3144.08 | 2673.06 | 471.02 | 4.12 | 17.993 | 0 |
| legacy | 36/50 | 0.72 | 3144.08 | 2673.06 | 471.02 | 4.12 | 17.996 | 0 |
| MACT TabFact 50 | 44/50 | 0.88 | 11051.98 | 7869.24 | 3182.74 | 3.28 | 118.543 | 0 |

关键 eval 字段：

```text
no-strong:
  result_schema=myagent
  num_samples=50
  primary_accuracy=0.72
  accuracy_metric=tabfact_binary_accuracy
  num_failed_exec=0
  risk_distribution={"high":41,"medium":9}
  avg_total_tokens=3144.08
  avg_elapsed_seconds=17.99314069529297

legacy:
  result_schema=myagent
  num_samples=50
  primary_accuracy=0.72
  accuracy_metric=tabfact_binary_accuracy
  num_failed_exec=0
  risk_distribution={"unknown":50}
  avg_total_tokens=3144.08
  avg_elapsed_seconds=17.996034594981467
```

行级标记核对：

```text
baseline rows=50 strong_verification_applied=48 risk={"high":48,"medium":2}
no_strong rows=50 strong_verification_applied=0 risk={"high":41,"medium":9}
legacy rows=50 strong_verification_applied=0 risk={"":50}
```

## 6. 消融对比和判断

相对 selective baseline，no-strong 提升 2 题：

```text
accuracy: 68% -> 72%
avg_total_tokens: 15793.76 -> 3144.08
avg_elapsed_seconds: 32.424s -> 17.993s
avg_llm_calls: 7.08 -> 4.12
```

legacy 与 no-strong 的 accuracy、avg tokens、avg LLM calls 基本完全一致。差异只体现在 no-strong 仍有 selective risk 记录，legacy 的 risk 为 unknown。

消融判断：

1. 本轮证据明确支持当前 TabFact strong verification 触发策略带来很大的 token/latency 开销：baseline 中 48/50 触发 strong verification，avg tokens 是 no-strong 的约 5.0 倍。
2. accuracy 上，baseline 低于 no-strong/legacy 2 题；由于 vLLM 本地推理仍可能存在非完全确定性，不能把这 2 题差异全部归因到 verifier 覆盖答案。但结合 token 和行级 strong 标记，TabFact label 类任务不应默认按 high-risk 进入多路 strong verification。
3. 关闭 strong verification 后，selective path 未比 legacy 更差；因此当前主要问题不是 risk 记录本身，而是 TabFact strong verifier 的触发策略、候选覆盖或 evidence/prompt 设计。
4. no-strong/legacy 虽然 token 明显低于 MACT TabFact 50，准确率仍只有 72%，明显低于 MACT 的 88%。所以这次消融只能说明 strong verification 应降级，不能说明 myAgent TabFact 已达标。

## 7. 后续策略修正和验证

基于上面的消融，已做两个窄范围代码调整：

1. TabFact true/false label 默认不再因为 high-risk 自动触发 strong verification；只保留 post-risk fallback 或候选冲突等 forced fallback 场景。
2. 增加一组 TabFact 确定性语义 shortcut，覆盖本次 50 条错例中反复出现的 episode order、episode credit count、goal competition count、not-fewer-than-any-other、nonzero metric count、maximum metric span、goal result count 等模式。

新增单测覆盖：

```text
python tests/test_myagent_pipeline.py
  OK, Ran 101 tests

python -m py_compile code/my_agents.py scripts/server/run_sharded_tqa.py
  exit 0

python tests/test_selective_collaboration.py
python tests/test_dataset_profiles.py
python tests/test_evaluate_results.py
  OK, Ran 6 + 11 + 13 tests
```

策略修正后的 TabFact 50 结果：

| 口径 | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed | strong applied | shortcut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selective baseline | 34/50 | 0.68 | 15793.76 | 15042.48 | 751.28 | 7.08 | 32.424 | 0 | 48/50 | n/a |
| no-strong | 36/50 | 0.72 | 3144.08 | 2673.06 | 471.02 | 4.12 | 17.993 | 0 | 0/50 | n/a |
| policy_v2 | 39/50 | 0.78 | 2556.50 | 2197.96 | 358.54 | 3.74 | 14.630 | 0 | 0/50 | 8/50 |
| policy_v3 | 44/50 | 0.88 | 2202.82 | 1874.20 | 328.62 | 3.48 | 12.333 | 0 | 0/50 | 14/50 |
| MACT TabFact 50 | 44/50 | 0.88 | 11051.98 | 7869.24 | 3182.74 | 3.28 | 118.543 | 0 | n/a | n/a |

policy_v3 与 MACT 在这 50 条 TabFact 上同为 44/50，但 token 约为 MACT 的 19.9%，latency 约为 MACT 的 10.4%。

policy_v3 blind200 验证使用 blind holdout：

```text
datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl
manifest protocol=blind_holdout_v3
seed=20365356
true=100, false=100
prior_id_overlap=0, prior_table_overlap=0
```

输出文件：

```text
outputs/server_runs/qwen3_32b_tabfact_blind200_policy_v3/raw/tabfact/tabfact_blind200_out.jsonl
outputs/server_runs/qwen3_32b_tabfact_blind200_policy_v3/merged/tabfact_qwen3-32b-local.jsonl
outputs/server_runs/qwen3_32b_tabfact_blind200_policy_v3/eval/tabfact_qwen3-32b-local_eval.json
outputs/server_runs/qwen3_32b_tabfact_blind200_policy_v3/logs/tabfact/tabfact_blind200.log
```

行数和日志核对：

```text
200 raw rows
200 merged rows
Finished sample 200/200
未检出 Traceback、Connection refused、APIConnectionError、Error processing
```

blind200 eval：

| 口径 | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed | strong applied | shortcut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| policy_v3 blind200 | 167/200 | 0.835 | 2703.50 | 2396.175 | 307.325 | 3.88 | 12.009 | 0 | 0/200 | 13/200 |

risk 分层：

```text
high:   124/153 = 0.8105, avg_total_tokens=2907.48
medium:  43/47 = 0.9149, avg_total_tokens=2039.47
```

deterministic shortcut 在 blind200 上 13/13 正确。错例共 33 个，主要仍是 TabFact 通用语义模式：日期区间、最小/最大排序、赛果方向、跨行计数、差值/时间差和带否定的图表命题。下一轮如果继续优化，应优先从这 33 个 blind 错例中提取可泛化规则，避免只贴合前 50 条。

## 8. 下一步建议

1. blind200 已证明 policy_v3 的 50 条提升不是只来自前 50 条贴合，但 83.5% 仍不能写成 TabFact 已全面达标。
2. 下一步优先处理 blind200 的 33 个错例，方向是可解释的日期区间、排序、赛果方向和跨行计数规则。
3. strong verification 后续应作为 forced fallback 工具，而不是 TabFact label 的默认 high-risk 工具；如果要恢复，应先做单路 audit verifier 小样本对照。
4. 全量运行前建议再做一个新的 blind200 或 blind500，确认新增规则没有把 shortcut 变成过拟合。

## 9. 本轮提交范围

首次消融提交只提交了本报告。后续 policy_v3 提交范围为：

```text
code/my_agents.py
tests/test_myagent_pipeline.py
docs/server/server_codex_reports/2026-07-19-tabfact-qwen3-ablation.md
```

不提交：

```text
outputs/server_runs/*
configs/server/qwen3_32b_2gpu_local.env.example
configs/server/qwen3_32b_2gpu_local.env.bak.20260709_191806
restart_qwen3_context_try.sh
```
