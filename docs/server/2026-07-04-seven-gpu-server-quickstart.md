# 七卡服务器从 0 到 1 实验执行文档

日期：2026-07-04  
项目：`myAgent-main`  
默认硬件：7 x RTX 4090 24GB  
默认模型：`Qwen/Qwen2.5-14B-Instruct-AWQ`  
默认推理框架：vLLM OpenAI-compatible server  

这份文档的目标是：服务器拿到项目后，按命令一行一行执行，完成模型下载、vLLM 服务启动、myAgent 全量评测、消融实验、结果合并、评估和 SQLite 索引。

## 0. 实验设计结论

默认方案使用 7 张卡分别启动 7 个 14B-AWQ 服务：

```text
GPU0 -> http://127.0.0.1:8000/v1
GPU1 -> http://127.0.0.1:8001/v1
...
GPU6 -> http://127.0.0.1:8006/v1
```

实验脚本会把每个数据集切成 7 份，并发请求 7 个服务。这比用 7 张卡 tensor parallel 跑一个 14B 服务更适合批量评测，吞吐更高，也更不容易 OOM。

当前 full 数据集：

| 数据集 | 文件 | 数量 |
|---|---|---:|
| WTQ | `datasets_ready/full/wtq_unseen.jsonl` | 4344 |
| TabFact | `datasets_ready/full/tabfact_test.jsonl` | 12779 |
| CRT | `datasets_ready/full/crt.jsonl` | 728 |
| 总计 |  | 17851 |

运行时不需要 MySQL/PostgreSQL。项目输入是 JSONL，输出也是 JSONL。为了后续筛错题，本项目提供了可选 SQLite 索引脚本。

## 1. 上传项目

如果服务器能访问你的 Git 仓库：

```bash
cd /data
git clone <你的仓库地址> myAgent-main
cd /data/myAgent-main
```

如果没有 Git 仓库，就在本地压缩后上传：

```bash
cd /data
tar -xzf myAgent-main.tar.gz
cd /data/myAgent-main
```

确认关键文件存在：

```bash
ls code/tqa.py
ls datasets_ready/full
ls configs/server/qwen14b_7gpu.env.example
```

## 2. 安装依赖

```bash
cd /data/myAgent-main
bash scripts/server/install_server_deps.sh
source .venv-server/bin/activate
```

如果服务器国内网络下载 HuggingFace 慢，可以临时设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果 vLLM 安装失败，优先确认 CUDA、NVIDIA driver 和 Python 版本。建议 Python 3.10 或 3.11。

## 3. 配置模型和端口

复制配置文件：

```bash
cp configs/server/qwen14b_7gpu.env.example configs/server/qwen14b_7gpu.env
nano configs/server/qwen14b_7gpu.env
```

默认配置如下：

```bash
export MODEL_ID=Qwen/Qwen2.5-14B-Instruct-AWQ
export SERVED_MODEL_NAME=qwen25-14b-awq
export GPU_GROUPS="0;1;2;3;4;5;6"
export BASE_PORT=8000
export VLLM_API_KEY=local-vllm-key-change-me
export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"
```

只需要重点检查：

| 变量 | 说明 |
|---|---|
| `MODEL_ID` | HuggingFace 模型名 |
| `SERVED_MODEL_NAME` | myAgent 请求时使用的模型名 |
| `GPU_GROUPS` | GPU 分组，`;` 分隔服务，`,` 分隔 tensor parallel 卡 |
| `BASE_PORT` | 第一个 vLLM 端口 |
| `VLLM_API_KEY` | 本地服务鉴权 key，随便设但要前后一致 |

如果你的服务器只开放一个外部端口，不影响实验。myAgent 和 vLLM 都在服务器内部通信，使用 `127.0.0.1:8000-8006` 即可，不需要把 7 个端口暴露给外网。

## 4. 下载模型

```bash
source .venv-server/bin/activate
bash scripts/server/download_models.sh configs/server/qwen14b_7gpu.env
```

也可以直接让 vLLM 启动时自动下载，但提前下载更稳。

## 5. 启动七个 vLLM 服务

```bash
source .venv-server/bin/activate
bash scripts/server/start_vllm_pool.sh configs/server/qwen14b_7gpu.env
```

观察日志：

```bash
tail -f logs/server/vllm_8000.log
```

观察显卡：

```bash
watch -n 2 nvidia-smi
```

健康检查：

```bash
bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen14b_7gpu.env
```

如果 7 个端口都返回内容，说明模型服务已经接通。

停止服务：

```bash
bash scripts/server/stop_vllm_pool.sh
```

## 6. Smoke Test

先每个数据集跑 21 条，确认端到端无误：

```bash
source .venv-server/bin/activate
source configs/server/qwen14b_7gpu.env

ENDPOINTS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1"

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_smoke \
  --limit-per-task 21
```

检查输出：

```bash
ls outputs/server_runs/qwen14b_smoke/merged
cat outputs/server_runs/qwen14b_smoke/eval/wtq_qwen25-14b-awq_eval.json
cat outputs/server_runs/qwen14b_smoke/eval/tabfact_qwen25-14b-awq_eval.json
cat outputs/server_runs/qwen14b_smoke/eval/crt_qwen25-14b-awq_eval.json
```

如果 smoke test 有失败，先不要跑全量，把下面文件发回来：

```text
outputs/server_runs/qwen14b_smoke/logs/
logs/server/vllm_8000.log
```

## 7. 全量主实验：myAgent-full

确认 smoke 通过后，跑全量：

```bash
source .venv-server/bin/activate
source configs/server/qwen14b_7gpu.env

ENDPOINTS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1"

nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_full \
  > logs/server/qwen14b_full_runner.log 2>&1 &
```

查看进度：

```bash
tail -f logs/server/qwen14b_full_runner.log
find outputs/server_runs/qwen14b_full/raw -name "*.jsonl" -exec wc -l {} \;
```

如果中断后重跑，用 `--resume`：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_full \
  --resume \
  > logs/server/qwen14b_full_runner_resume.log 2>&1 &
```

全量完成后应出现：

```text
outputs/server_runs/qwen14b_full/merged/wtq_qwen25-14b-awq.jsonl
outputs/server_runs/qwen14b_full/merged/tabfact_qwen25-14b-awq.jsonl
outputs/server_runs/qwen14b_full/merged/crt_qwen25-14b-awq.jsonl
outputs/server_runs/qwen14b_full/eval/*_eval.json
```

## 8. 关键消融实验

优先跑这三个，足够支撑论文和专利的主要机制。

### 8.1 w/o deterministic verifier

关闭公式化 deterministic shortcut，用来证明 verifier 的贡献：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_no_deterministic_verifier \
  --disable-deterministic-shortcuts \
  > logs/server/qwen14b_no_deterministic_verifier.log 2>&1 &
```

### 8.2 w/o strong verification

关闭高风险二次校验，用来证明风险题增强协作的贡献：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_no_strong_verification \
  --disable-strong-verification \
  > logs/server/qwen14b_no_strong_verification.log 2>&1 &
```

### 8.3 legacy collaboration

关闭选择性协作，回到 legacy 路径，用来证明风险分层和协作机制的贡献：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_legacy \
  --collaboration-mode legacy \
  > logs/server/qwen14b_legacy.log 2>&1 &
```

建议顺序：

```text
qwen14b_full
qwen14b_no_deterministic_verifier
qwen14b_no_strong_verification
qwen14b_legacy
```

## 9. SQLite 结果索引

运行时不需要数据库。跑完后如果要快速筛错题，可以把 merged JSONL 写入 SQLite：

```bash
python scripts/server/index_results_sqlite.py \
  --db outputs/server_runs/results.sqlite \
  --run-name qwen14b_full \
  --task wtq \
  --jsonl outputs/server_runs/qwen14b_full/merged/wtq_qwen25-14b-awq.jsonl

python scripts/server/index_results_sqlite.py \
  --db outputs/server_runs/results.sqlite \
  --run-name qwen14b_full \
  --task tabfact \
  --jsonl outputs/server_runs/qwen14b_full/merged/tabfact_qwen25-14b-awq.jsonl

python scripts/server/index_results_sqlite.py \
  --db outputs/server_runs/results.sqlite \
  --run-name qwen14b_full \
  --task crt \
  --jsonl outputs/server_runs/qwen14b_full/merged/crt_qwen25-14b-awq.jsonl
```

查询每个数据集准确率：

```bash
sqlite3 outputs/server_runs/results.sqlite \
  "select run_name, task, count(*) n, sum(correct) correct, round(avg(correct),4) acc, round(avg(total_tokens),1) avg_tokens from samples group by run_name, task;"
```

筛错题：

```bash
sqlite3 outputs/server_runs/results.sqlite \
  "select task, sample_id, prediction, gold, risk_level from samples where run_name='qwen14b_full' and correct=0 limit 20;"
```

## 10. 打包结果发回

主实验跑完后，打包这些结果：

```bash
tar -czf outputs/server_runs/qwen14b_full_results.tgz \
  outputs/server_runs/qwen14b_full/merged \
  outputs/server_runs/qwen14b_full/eval \
  outputs/server_runs/results.sqlite \
  logs/server/qwen14b_full_runner.log
```

如果跑了消融，也一起打包：

```bash
tar -czf outputs/server_runs/qwen14b_all_results.tgz \
  outputs/server_runs/qwen14b_full \
  outputs/server_runs/qwen14b_no_deterministic_verifier \
  outputs/server_runs/qwen14b_no_strong_verification \
  outputs/server_runs/qwen14b_legacy \
  outputs/server_runs/results.sqlite \
  logs/server
```

把 `.tgz` 发回来即可。

## 11. 七卡的其他模型方案

### 11.1 默认推荐：7 x 14B-AWQ

```bash
export MODEL_ID=Qwen/Qwen2.5-14B-Instruct-AWQ
export SERVED_MODEL_NAME=qwen25-14b-awq
export GPU_GROUPS="0;1;2;3;4;5;6"
export BASE_PORT=8000
```

优点：吞吐高、稳定、最适合全量评测。

### 11.2 强模型补充：3 x 32B-AWQ 服务

如果要跑强模型小样本或补充实验：

```bash
export MODEL_ID=Qwen/Qwen2.5-32B-Instruct-AWQ
export SERVED_MODEL_NAME=qwen25-32b-awq
export GPU_GROUPS="0,1;2,3;4,5"
export BASE_PORT=8100
```

这会启动 3 个服务，每个服务用 2 张卡 tensor parallel。GPU6 空出来用于监控或另起一个 14B 服务。

对应 endpoints：

```bash
ENDPOINTS="http://127.0.0.1:8100/v1,http://127.0.0.1:8101/v1,http://127.0.0.1:8102/v1"
```

建议 32B 先跑 smoke 或每数据集 200-500 条，不建议一开始直接全量消融。

## 12. 常见问题

### 12.1 vLLM OOM

先调低：

```bash
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.82
```

然后重启 vLLM。

### 12.2 端口不通

确认服务在本机监听：

```bash
ss -lntp | grep 800
```

确认日志没有报错：

```bash
tail -n 100 logs/server/vllm_8000.log
```

### 12.3 HuggingFace 下载失败

可以设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/server/download_models.sh configs/server/qwen14b_7gpu.env
```

### 12.4 只开放一个服务器端口怎么办

不影响实验。实验命令在服务器本机运行，使用 `127.0.0.1` 访问多个本地端口。外部只需要 SSH 端口即可。

### 12.5 怎么确认跑完

每个 merged 文件行数应等于 full 数据集数量：

```bash
wc -l outputs/server_runs/qwen14b_full/merged/*.jsonl
```

期望：

```text
4344  wtq_qwen25-14b-awq.jsonl
12779 tabfact_qwen25-14b-awq.jsonl
728   crt_qwen25-14b-awq.jsonl
```

## 13. 参考来源

- vLLM OpenAI-compatible server 文档：`https://docs.vllm.ai/en/v0.7.1/serving/openai_compatible_server.html`
- Qwen2.5-14B-Instruct-AWQ 模型卡：`https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ`
- Qwen2.5-Coder-14B-Instruct-AWQ 模型卡：`https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`
- Qwen2.5-32B-Instruct-AWQ 模型卡：`https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ`
- HuggingFace CLI 下载文档：`https://huggingface.co/docs/huggingface_hub/en/guides/cli`
