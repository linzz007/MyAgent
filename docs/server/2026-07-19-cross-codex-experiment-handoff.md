# Cross-Codex Experiment Handoff

日期：2026-07-19  
面向对象：服务器 VSCode Codex，以及本地 Windows Codex  
项目：`MyAgent` 专利实验，对标 `MACT`

## 0. 先读结论

当前目标不是继续聊天式排查，而是让服务器侧 Codex 接管真实实验执行与结果记录。本文件是两边 Codex 的共享事实表。

服务器侧 Codex 的第一任务：

1. 拉取 GitHub 最新代码，确认包含 `b994f5a Propagate TQA shard failures`。
2. 确认 Qwen3-32B vLLM 服务已经健康，而不是刚启动但还没监听端口。
3. 重跑 `TabFact 50 no-strong` 消融。
4. 再跑 `TabFact 50 legacy` 消融。
5. 生成服务器侧结果报告，并提交/推送回 GitHub，方便本地 Codex 继续分析。

不要一上来跑全量。当前最关键的诊断是 TabFact 为什么弱，以及 strong verification 是否导致过度验证、token 变高、准确率下降。

## 1. 研究目标

用户的最终目标：

- myAgent 的准确率要比肩或略高于 MACT。
- token 消耗可以增加，不再追求极限省 token。
- 如果平均 token 能控制在 MACT 的 70% 左右以内，就可以接受。
- 更重要的是：准确率不能差，尤其不能为了省 token 牺牲性能。

专利/论文上希望强调的技术点：

- 问题类型识别。
- 风险评分机制。
- 前后两阶段评分/校验机制。
- 表格信息压缩。
- 高风险题增强协作或二次校验。
- 预算不是固定少 token，而是风险自适应投入 token。

当前要避免的说法：

- 不要写“已经在所有设置下稳定全面超过 MACT”。
- 只能基于同模型、同数据、同 evaluator 的结果下结论。
- 服务器实验前，历史 DeepSeek 结果只能作为阶段性参考，不是 Qwen3 本地实验结论。

## 2. 当前仓库和分支

本地 GitHub 仓库：

```text
git@github.com:linzz007/MyAgent.git
```

当前工作分支：

```text
codex/selective-risk-collaboration
```

服务器侧应拉取这个分支：

```bash
cd /home/ubuntu/lzz/MyAgent

git fetch origin
git checkout codex/selective-risk-collaboration
git pull --ff-only origin codex/selective-risk-collaboration

git log --oneline -5
```

必须确认最新提交至少包含：

```text
b994f5a Propagate TQA shard failures
```

这个提交做了什么：

- `code/tqa.py`：样本处理异常后不再 `break` 静默退出，而是 `raise`，让外层 runner 得到非 0 退出码。
- `tests/test_tqa_failure_exit.py`：新增回归测试，证明异常不会再被吞掉。

为什么重要：

之前 `TabFact no-strong` 第一次失败时，真实错误是 vLLM `Connection refused`，但因为 `tqa.py` 吞掉异常，外层只看到误导性的 `FileNotFoundError: shard output jsonl not found`。

## 3. 服务器路径约定

服务器项目路径：

```text
/home/ubuntu/lzz/MyAgent
/home/ubuntu/lzz/MACT
```

模型路径：

```text
/home/ubuntu/models/Qwen3-32B
```

myAgent 配置文件：

```text
/home/ubuntu/lzz/MyAgent/configs/server/qwen3_32b_2gpu_local.env
```

典型配置：

```bash
export MODEL_ID=/home/ubuntu/models/Qwen3-32B
export SERVED_MODEL_NAME=qwen3-32b-local
export GPU_GROUPS="0,1"
export BASE_PORT=8000
export VLLM_MAX_MODEL_LEN=8192
export VLLM_GPU_MEMORY_UTILIZATION=0.88
export VLLM_EXTRA_ARGS="--trust-remote-code"
export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"
```

注意：

- 2 张 4090 跑 Qwen3-32B 时，`VLLM_MAX_MODEL_LEN=16384` 已经尝试过，会 CUDA OOM。
- 目前可行上限按 `8192` 处理。
- 如果服务器实际有更多卡，也先不要随便改上下文，除非先记录显存、并发和启动日志。

## 4. 数据集路径

不要混淆 raw source 和 runnable JSONL。

本地原始数据源：

```text
D:\AAAcode\code-code\agent+\dataset
```

服务器实验实际使用的是仓库内已经适配好的 JSONL：

```text
/home/ubuntu/lzz/MyAgent/datasets_ready/full/wtq_unseen.jsonl
/home/ubuntu/lzz/MyAgent/datasets_ready/full/tabfact_test.jsonl
/home/ubuntu/lzz/MyAgent/datasets_ready/full/crt.jsonl
```

已知全量条数：

```text
WTQ      4344
TabFact 12779
CRT       728
```

检查命令：

```bash
cd /home/ubuntu/lzz/MyAgent
wc -l datasets_ready/full/wtq_unseen.jsonl \
      datasets_ready/full/tabfact_test.jsonl \
      datasets_ready/full/crt.jsonl
```

runner 默认任务到数据文件的映射在：

```text
scripts/server/run_sharded_tqa.py
```

字段契约：

- MACT/myAgent 共享 MACT-style JSONL。
- 每条样本至少应有 `statement` 或 `question`、`table_text`、`answer`。
- TabFact 使用 `statement` 和 true/false answer。

## 5. 当前模型服务启动方式

每次跑实验前，必须先确认 vLLM 健康，不要只看启动进程存在。

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
```

如果 healthcheck 失败，启动并等待健康：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

bash scripts/server/start_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env

until bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env; do
  echo "waiting for vLLM..."
  sleep 10
done
```

看日志：

```bash
tail -f logs/server/vllm_8000.log
```

真实可服务的日志标志类似：

```text
Starting vLLM API server 0 on http://0.0.0.0:8000
```

已知失败：

```text
openai.APIConnectionError: Connection error.
httpcore.ConnectError: [Errno 111] Connection refused
```

这通常表示实验启动早于 vLLM 端口就绪。

## 6. 当前 Qwen3-32B 50 条结果

这批结果来自服务器用户回传，需要服务器 Codex 尽量用文件复核。

Run tag：

```text
qwen3_32b_50
```

myAgent 50/数据集：

| 数据集 | 准确率 | 平均耗时 | 平均 token | 失败 |
|---|---:|---:|---:|---:|
| WTQ | 0.62 | 22.94s/题 | 11656.56 | 0 |
| TabFact | 0.68 | 32.39s/题 | 15793.76 | 0 |
| CRT | 0.76 | 27.95s/题 | 12299.60 | 0 |
| 总体 | 103/150 = 68.67% | 27.76s/题 | 13249.97 | 0 |

myAgent vs MACT 50/数据集配对比较：

| 数据集 | myAgent | MACT | 主要观察 |
|---|---:|---:|---|
| WTQ | 31/50 = 62% | 23/50 = 46% | myAgent 明显更好；MACT WTQ 有上下文/失败处理痕迹，需谨慎看 |
| TabFact | 34/50 = 68% | 44/50 = 88% | myAgent 明显弱，是当前优先诊断目标 |
| CRT | 38/50 = 76% | 33/50 = 66% | myAgent 更好 |
| 总体 | 103/150 = 68.67% | 100/150 = 66.67% | myAgent 略高，但 token 更高 |

token 观察：

- myAgent avg tokens：约 `13249.86`
- MACT avg tokens：约 `11460.75`
- myAgent / MACT token ratio：约 `1.156`

这不符合“token 低于 MACT 70%”目标。

主要原因的当前判断：

1. `scripts/server/run_sharded_tqa.py` 的 `--mact-avg-tokens` 默认仍是历史 DeepSeek MACT 基线 `47439.2633`，不是当前 Qwen3 MACT 基线 `11460`。
2. `BudgetController` 当前更多是记录和报告，不是强制门控。
3. TabFact 几乎全部被判为 high risk/complex。
4. TabFact 在 strong verification 下可能额外跑 `direct/audit/program` 多路 verifier。
5. verifier evidence 可能重复携带原始表和压缩表，导致 token 高。

## 7. 当前核心假设

当前不是继续省 token，而是找 TabFact 为什么低。

核心假设：

```text
TabFact 的 strong verification 触发过多，导致 token 上升，同时 Qwen3-32B 在长表二分类验证中可能被二次 verifier 带偏，覆盖了原本正确的答案。
```

所以第一优先级不是全量实验，而是消融：

1. `TabFact 50 no-strong`：关闭 strong verification。
2. `TabFact 50 legacy`：完全关闭 selective collaboration，只走 legacy。
3. 对比三者：
   - 当前 selective baseline：`qwen3_32b_50` 中 TabFact 68%。
   - no-strong 是否提升准确率或显著降 token。
   - legacy 是否更稳。

判定逻辑：

- 如果 no-strong 准确率高于 68% 且 token 降低，说明 strong verification 当前负贡献。
- 如果 legacy 高于 selective，说明 TabFact 的 selective path/risk policy 需要数据集特化。
- 如果 no-strong 和 legacy 都低，说明问题在主流程、表格压缩、TabFact prompt 或 true/false 答案规范。

## 8. 立即执行：TabFact no-strong

先拉最新代码：

```bash
cd /home/ubuntu/lzz/MyAgent

git fetch origin
git checkout codex/selective-risk-collaboration
git pull --ff-only origin codex/selective-risk-collaboration

git log --oneline -1
```

验证最新代码：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

python tests/test_tqa_failure_exit.py
python -m py_compile code/tqa.py scripts/server/run_sharded_tqa.py
```

确保模型健康：

```bash
source configs/server/qwen3_32b_2gpu_local.env

if ! bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env; then
  bash scripts/server/start_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
  until bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env; do
    echo "waiting for vLLM..."
    sleep 10
  done
fi
```

运行 no-strong：

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

查看结果：

```bash
cat outputs/server_runs/qwen3_32b_tabfact50_no_strong/eval/tabfact_${SERVED_MODEL_NAME}_eval.json
wc -l outputs/server_runs/qwen3_32b_tabfact50_no_strong/merged/*.jsonl
tail -100 outputs/server_runs/qwen3_32b_tabfact50_no_strong/logs/tabfact/tabfact_shard00.log
```

## 9. 立即执行：TabFact legacy

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

查看结果：

```bash
cat outputs/server_runs/qwen3_32b_tabfact50_legacy/eval/tabfact_${SERVED_MODEL_NAME}_eval.json
wc -l outputs/server_runs/qwen3_32b_tabfact50_legacy/merged/*.jsonl
tail -100 outputs/server_runs/qwen3_32b_tabfact50_legacy/logs/tabfact/tabfact_shard00.log
```

## 10. 进度监控

推荐用 `tmux`，不要只依赖 `nohup`。

```bash
tmux new -s myagent-exp
```

另一窗口查看：

```bash
ps -ef --forest | grep -E "run_sharded_tqa|code/tqa.py|vllm" | grep -v grep
watch -n 20 'find /home/ubuntu/lzz/MyAgent/outputs/server_runs -path "*raw*" -name "*.jsonl" -exec wc -l {} \; | sort'
tail -f /home/ubuntu/lzz/MyAgent/outputs/server_runs/qwen3_32b_tabfact50_no_strong/logs/tabfact/tabfact_shard00.log
```

如果进程结束但没有 eval：

```bash
find outputs/server_runs/qwen3_32b_tabfact50_no_strong -maxdepth 5 -type f -printf '%p\t%s bytes\n' | sort
tail -300 outputs/server_runs/qwen3_32b_tabfact50_no_strong/logs/tabfact/tabfact_shard00.log
```

## 11. 服务器 Codex 应生成的回传报告

服务器侧每完成一次实验或代码修改，都应该新建一份报告：

```text
docs/server/server_codex_reports/YYYY-MM-DD-<short-topic>.md
```

建议本次报告路径：

```text
docs/server/server_codex_reports/2026-07-19-tabfact-qwen3-ablation.md
```

报告必须包含：

1. 当前 git commit：
   ```bash
   git log --oneline -1
   ```
2. GPU 和 vLLM 状态：
   ```bash
   nvidia-smi
   curl -sS -H "Authorization: Bearer ${LOCAL_VLLM_API_KEY:-EMPTY}" http://127.0.0.1:8000/v1/models
   ```
3. 运行命令原文。
4. 输出文件路径：
   - `outputs/server_runs/<tag>/merged/*.jsonl`
   - `outputs/server_runs/<tag>/eval/*.json`
   - `outputs/server_runs/<tag>/logs/**/*.log`
5. 准确率、平均 token、平均耗时、失败数。
6. 与当前 baseline 的比较：
   - TabFact selective baseline：68%，avg token 15793.76。
   - MACT TabFact 50：88%，avg token 11051.98。
7. 初步判断：
   - no-strong 是否改善准确率。
   - legacy 是否改善准确率。
   - 是否说明 strong verification 负贡献。
8. 下一步建议：
   - 是否改代码。
   - 改哪些文件。
   - 是否需要再跑 200 条。

## 12. 两边 Codex 同步协议

核心原则：

```text
GitHub 分支 + Markdown 报告 = 两边 Codex 的共享记忆。
```

服务器 Codex 做完事情后：

```bash
cd /home/ubuntu/lzz/MyAgent

git status -sb
git add docs/server/server_codex_reports/<report>.md

# 如果改了代码，只 add 本次明确相关的代码和测试，不要 add outputs 大文件。
git add code/<changed>.py tests/<changed>.py

git commit -m "Record server TabFact ablation"
git push origin codex/selective-risk-collaboration
```

本地 Codex 继续前：

```powershell
cd D:\AAAcode\code-code\agent+\myAgent-main
git fetch origin
git pull --ff-only origin codex/selective-risk-collaboration
```

本地 Codex 也应该写回报告，例如：

```text
docs/server/local_codex_reports/YYYY-MM-DD-<short-topic>.md
```

避免事项：

- 不要把 `outputs/server_runs` 的大 JSONL 全量提交进 Git，除非用户明确要求。
- 不要提交 API key、`.env` 私密配置、模型文件。
- 不要混合提交无关报告、Word 文档、临时文件。
- 不要在没有记录命令和 commit 的情况下直接改实验代码。

## 13. 如果服务器 Codex 要修改代码

优先判断是否真的需要改代码。

当前可能需要改的方向：

### 13.1 TabFact 强验证降级

如果 no-strong 或 legacy 明显好于 selective baseline：

- TabFact 默认不要跑 `direct/audit/program` 三路 strong verifier。
- 改成只在低置信度或候选冲突时触发。
- 或 TabFact 只跑单路 `audit` verifier。
- verifier evidence 不要同时塞完整 `original_table` 和 `compressed_table`。

相关文件：

```text
code/my_agents.py
code/selective_collaboration.py
code/dataset_profiles.py
tests/test_myagent_pipeline.py
tests/test_selective_collaboration.py
tests/test_dataset_profiles.py
```

### 13.2 预算控制从记录改为门控

当前 `BudgetController` 主要用于记录预算状态。若要体现专利中的预算公式，应考虑把它接入执行门控：

- 超过预算时跳过额外 verifier。
- 对 TabFact 使用更低 verification budget。
- 对 WTQ/CRT 高风险题允许更高预算。

相关文件：

```text
code/risk_control.py
code/my_agents.py
scripts/server/run_sharded_tqa.py
```

### 13.3 Qwen3 MACT token baseline

当前 server runner 默认 `--mact-avg-tokens` 仍是历史 DeepSeek 结果 `47439.2633`。

对 Qwen3 50 条比较，传参应使用：

```text
--mact-avg-tokens 11460
```

长期可以把默认值改成更安全的配置策略，但要避免把某一次实验的 MACT token 写死成通用常数。专利里应写成 `B_MACT` 或基准系统平均 token。

## 14. MACT 现状

MACT 是对标项目，当前不建议轻易全量重跑，除非需要正式论文全量基线。

服务器路径：

```text
/home/ubuntu/lzz/MACT
```

注意事项：

- 本地/服务器 MACT 已做过运行兼容性改造，以支持 OpenAI-compatible API、输出路径和 token 统计。
- 这些改造不应加入 myAgent 的专利机制。
- MACT WTQ 在 8k context 下可能因为 scratchpad 增长导致上下文超限。
- 降低 `--max_tokens` 只能减少输出预算，不能解决所有 prompt/scratchpad 增长问题。

当前阶段：

- 先不要重跑 MACT。
- 先用已有 MACT 50 对比结果定位 myAgent TabFact 问题。
- 如果 myAgent 消融证明可改善，再决定是否跑 200 或全量。

## 15. 已知历史报告

这些文件可作为背景，不要混成当前 Qwen3 结论：

```text
docs/server/2026-07-07-vscode-codex-handoff.md
docs/server/2026-07-06-current-server-qwen3-32b-runbook.md
docs/patent/2026-06-27-final-system-summary.md
docs/patent/2026-06-28-v12-final-optimization-report.md
docs/patent/2026-06-29-v13-mact-paper-aligned-update.md
docs/patent/2026-07-08-mact-baseline-audit.md
```

需要注意：

- DeepSeek v4flash/v13 的历史结果曾显示 myAgent 在 blind200 上可略超 MACT、token 更低。
- Qwen3-32B 本地实验是新的模型口径，不能直接继承 DeepSeek 结论。
- 当前 Qwen3 50 条结果显示总体 myAgent 略高于 MACT，但 TabFact 明显弱，且 token 高于 MACT。

## 16. 服务器 Codex 的最低完成标准

本轮服务器 Codex 至少要交付：

1. 证明最新代码已拉取：
   - `git log --oneline -1`
2. 证明模型健康：
   - healthcheck 输出
   - `/v1/models` 输出
3. 完成两个 TabFact 50 消融：
   - `qwen3_32b_tabfact50_no_strong`
   - `qwen3_32b_tabfact50_legacy`
4. 提供 eval JSON 内容和 merged 行数。
5. 写入 `docs/server/server_codex_reports/2026-07-19-tabfact-qwen3-ablation.md`。
6. 如果改代码，必须：
   - 写测试或最小验证。
   - 提交相关代码和测试。
   - push 到 `codex/selective-risk-collaboration`。

## 17. 给服务器 Codex 的直接提示词

可以把下面这段直接发给服务器 VSCode Codex：

```text
请先阅读 docs/server/2026-07-19-cross-codex-experiment-handoff.md。

你现在在服务器上，拥有 /home/ubuntu/lzz/MyAgent、/home/ubuntu/lzz/MACT、vLLM、GPU 和实验输出文件的直接权限。请不要重新猜测历史背景，先按文档完成：

1. 拉取 origin/codex/selective-risk-collaboration 最新代码，确认 commit b994f5a 或更新。
2. 检查 Qwen3-32B vLLM 是否健康；如果不健康，启动并等待 healthcheck 通过。
3. 重跑 TabFact 50 no-strong 消融。
4. 跑 TabFact 50 legacy 消融。
5. 读取 eval JSON、merged 行数和日志，判断 strong verification 是否造成 TabFact 准确率下降或 token 过高。
6. 把结果写到 docs/server/server_codex_reports/2026-07-19-tabfact-qwen3-ablation.md。
7. 如需改代码，先定位根因，再改最小范围，并提交测试、代码和报告，push 回 GitHub。

不要提交 outputs/server_runs 大文件，不要提交 API key 或模型文件。所有结论必须附命令、路径和结果。
```

