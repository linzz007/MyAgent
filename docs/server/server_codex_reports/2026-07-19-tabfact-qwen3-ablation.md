# 2026-07-19 TabFact Qwen3-32B Ablation Report

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
实验任务：TabFact 50 no-strong 与 legacy 消融

## 1. Git 和代码验证

当前提交：

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

## 6. 对比和判断

相对 selective baseline，no-strong 提升 2 题：

```text
accuracy: 68% -> 72%
avg_total_tokens: 15793.76 -> 3144.08
avg_elapsed_seconds: 32.424s -> 17.993s
avg_llm_calls: 7.08 -> 4.12
```

legacy 与 no-strong 的 accuracy、avg tokens、avg LLM calls 基本完全一致。差异只体现在 no-strong 仍有 selective risk 记录，legacy 的 risk 为 unknown。

初步判断：

1. 本轮证据支持 strong verification 对 TabFact 是负贡献：baseline 中 48/50 触发 strong verification，准确率更低且 token 高约 5.0 倍。
2. 关闭 strong verification 后，selective path 未比 legacy 更差；因此当前主要问题不是 risk 记录本身，而是 TabFact strong verifier 的触发策略、候选覆盖或 evidence/prompt 设计。
3. no-strong/legacy 虽然 token 明显低于 MACT TabFact 50，准确率仍只有 72%，明显低于 MACT 的 88%。所以这次消融只能说明 strong verification 应降级，不能说明 myAgent TabFact 已达标。

## 7. 下一步建议

1. 代码方向：TabFact 默认不要触发当前 direct/audit/program 多路 strong verification。可改为只在低置信度、候选冲突、或确定性 shortcut 不可用时触发单路 audit verifier。
2. Evidence 方向：检查 strong verifier 是否重复携带 `original_table` 和 `compressed_table`，避免长表二分类中把模型带偏。
3. Prompt 方向：单独优化 TabFact true/false 输出契约，减少 verifier 覆盖原本正确答案的机会。
4. 实验方向：改完后先跑 TabFact 50，再跑 TabFact 200 no-strong/新策略，不建议直接跑全量。
5. 结论表述：当前只能写“Qwen3-32B TabFact 上 strong verification 在 50 条消融中表现为负贡献”，不能写“myAgent 已全面超过 MACT”。

## 8. 本轮提交范围

本轮没有改代码。按 handoff 要求，只提交本报告：

```text
docs/server/server_codex_reports/2026-07-19-tabfact-qwen3-ablation.md
```

不提交：

```text
outputs/server_runs/*
configs/server/qwen3_32b_2gpu_local.env.example
configs/server/qwen3_32b_2gpu_local.env.bak.20260709_191806
restart_qwen3_context_try.sh
```
