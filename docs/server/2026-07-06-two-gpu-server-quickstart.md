# 两张 4090 服务器从 0 到 1执行文档

日期：2026-07-06  
默认硬件：2 x RTX 4090 24GB  
默认模型：`Qwen/Qwen2.5-14B-Instruct-AWQ`  
默认推理框架：vLLM OpenAI-compatible server  

## 0. 你要理解的启动顺序

顺序是：

```text
下载模型
-> 启动 vLLM 本地模型服务
-> myAgent 通过 http://127.0.0.1:8000/v1 和 8001/v1 调模型
-> 跑 smoke test
-> 跑 full 实验
-> 合并/评估/打包结果
```

模型不需要写进项目代码。项目只需要知道：

| 配置 | 含义 |
|---|---|
| `SERVED_MODEL_NAME` | vLLM 对外暴露的模型名 |
| `api_base` | vLLM OpenAI-compatible 地址 |
| `LOCAL_VLLM_API_KEY` | 本地服务鉴权 key |

## 1. 进入项目并安装依赖

假设项目放在 `/data/myAgent-main`：

```bash
cd /data/myAgent-main
bash scripts/server/install_server_deps.sh
source .venv-server/bin/activate
```

如果 HuggingFace 下载慢，先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 2. 创建两卡配置

```bash
cd /data/myAgent-main
cp configs/server/qwen14b_2gpu.env.example configs/server/qwen14b_2gpu.env
nano configs/server/qwen14b_2gpu.env
```

默认不用改也能跑：

```bash
export MODEL_ID=Qwen/Qwen2.5-14B-Instruct-AWQ
export SERVED_MODEL_NAME=qwen25-14b-awq
export GPU_GROUPS="0;1"
export BASE_PORT=8000
export VLLM_API_KEY=local-vllm-key-change-me
export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"
```

这表示：

```text
GPU0 -> http://127.0.0.1:8000/v1
GPU1 -> http://127.0.0.1:8001/v1
```

如果 `14B-AWQ` 启动 OOM，把 `VLLM_MAX_MODEL_LEN=8192` 改成：

```bash
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.82
```

## 3. 下载模型

```bash
source .venv-server/bin/activate
bash scripts/server/download_models.sh configs/server/qwen14b_2gpu.env
```

下载完成后模型会在 `HF_HOME`，默认：

```text
/data/hf
```

## 4. 启动两个 vLLM 模型服务

```bash
source .venv-server/bin/activate
bash scripts/server/start_vllm_pool.sh configs/server/qwen14b_2gpu.env
```

查看日志：

```bash
tail -f logs/server/vllm_8000.log
tail -f logs/server/vllm_8001.log
```

查看显卡：

```bash
watch -n 2 nvidia-smi
```

健康检查：

```bash
bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen14b_2gpu.env
```

如果两个端口都返回回答，模型服务就启动好了。

停止模型服务：

```bash
bash scripts/server/stop_vllm_pool.sh
```

## 5. 跑 myAgent smoke test

先不要直接全量。先每个数据集跑 10 条，确认接口、输出、评估都正常：

```bash
source .venv-server/bin/activate
source configs/server/qwen14b_2gpu.env

ENDPOINTS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_2gpu_smoke \
  --limit-per-task 10
```

看结果：

```bash
ls outputs/server_runs/qwen14b_2gpu_smoke/merged
cat outputs/server_runs/qwen14b_2gpu_smoke/eval/wtq_qwen25-14b-awq_eval.json
cat outputs/server_runs/qwen14b_2gpu_smoke/eval/tabfact_qwen25-14b-awq_eval.json
cat outputs/server_runs/qwen14b_2gpu_smoke/eval/crt_qwen25-14b-awq_eval.json
```

如果这里失败，把这些文件发回来：

```text
logs/server/vllm_8000.log
logs/server/vllm_8001.log
outputs/server_runs/qwen14b_2gpu_smoke/logs/
```

## 6. 跑 full 主实验

smoke test 通过后跑全量：

```bash
source .venv-server/bin/activate
source configs/server/qwen14b_2gpu.env

ENDPOINTS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"

nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_2gpu_full \
  > logs/server/qwen14b_2gpu_full_runner.log 2>&1 &
```

查看进度：

```bash
tail -f logs/server/qwen14b_2gpu_full_runner.log
find outputs/server_runs/qwen14b_2gpu_full/raw -name "*.jsonl" -exec wc -l {} \;
```

如果中断，重新执行并加 `--resume`：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_2gpu_full \
  --resume \
  > logs/server/qwen14b_2gpu_full_resume.log 2>&1 &
```

完成后检查行数：

```bash
wc -l outputs/server_runs/qwen14b_2gpu_full/merged/*.jsonl
```

期望：

```text
4344  wtq_qwen25-14b-awq.jsonl
12779 tabfact_qwen25-14b-awq.jsonl
728   crt_qwen25-14b-awq.jsonl
```

## 7. 跑两个最关键消融

主实验完成后，优先跑这两个消融。

### 7.1 关闭 deterministic verifier

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_2gpu_no_deterministic_verifier \
  --disable-deterministic-shortcuts \
  > logs/server/qwen14b_2gpu_no_deterministic_verifier.log 2>&1 &
```

这个实验用于证明专利里的“问题类型触发公式化 verifier”有贡献。

### 7.2 关闭 strong verification

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_2gpu_no_strong_verification \
  --disable-strong-verification \
  > logs/server/qwen14b_2gpu_no_strong_verification.log 2>&1 &
```

这个实验用于证明“高风险题增强协作/二次校验”有贡献。

如果时间还够，再跑 legacy：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen14b_2gpu_legacy \
  --collaboration-mode legacy \
  > logs/server/qwen14b_2gpu_legacy.log 2>&1 &
```

## 8. SQLite 结果索引

运行不依赖数据库。跑完后为了方便查错题，可以导入 SQLite：

```bash
python scripts/server/index_results_sqlite.py \
  --db outputs/server_runs/results_2gpu.sqlite \
  --run-name qwen14b_2gpu_full \
  --task wtq \
  --jsonl outputs/server_runs/qwen14b_2gpu_full/merged/wtq_qwen25-14b-awq.jsonl

python scripts/server/index_results_sqlite.py \
  --db outputs/server_runs/results_2gpu.sqlite \
  --run-name qwen14b_2gpu_full \
  --task tabfact \
  --jsonl outputs/server_runs/qwen14b_2gpu_full/merged/tabfact_qwen25-14b-awq.jsonl

python scripts/server/index_results_sqlite.py \
  --db outputs/server_runs/results_2gpu.sqlite \
  --run-name qwen14b_2gpu_full \
  --task crt \
  --jsonl outputs/server_runs/qwen14b_2gpu_full/merged/crt_qwen25-14b-awq.jsonl
```

查询准确率：

```bash
sqlite3 outputs/server_runs/results_2gpu.sqlite \
  "select run_name, task, count(*) n, sum(correct) correct, round(avg(correct),4) acc, round(avg(total_tokens),1) avg_tokens from samples group by run_name, task;"
```

## 9. 打包结果发回

主实验：

```bash
tar -czf outputs/server_runs/qwen14b_2gpu_full_results.tgz \
  outputs/server_runs/qwen14b_2gpu_full/merged \
  outputs/server_runs/qwen14b_2gpu_full/eval \
  outputs/server_runs/results_2gpu.sqlite \
  logs/server/qwen14b_2gpu_full_runner.log
```

如果消融也跑了：

```bash
tar -czf outputs/server_runs/qwen14b_2gpu_all_results.tgz \
  outputs/server_runs/qwen14b_2gpu_full \
  outputs/server_runs/qwen14b_2gpu_no_deterministic_verifier \
  outputs/server_runs/qwen14b_2gpu_no_strong_verification \
  outputs/server_runs/qwen14b_2gpu_legacy \
  outputs/server_runs/results_2gpu.sqlite \
  logs/server
```

把 `.tgz` 发回来即可。

## 10. 如果 14B 太慢或 OOM

### 10.1 改用 7B

编辑 `configs/server/qwen14b_2gpu.env`：

```bash
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export SERVED_MODEL_NAME=qwen25-7b
export VLLM_MAX_MODEL_LEN=8192
```

然后重新下载和启动：

```bash
bash scripts/server/download_models.sh configs/server/qwen14b_2gpu.env
bash scripts/server/stop_vllm_pool.sh
bash scripts/server/start_vllm_pool.sh configs/server/qwen14b_2gpu.env
```

### 10.2 两张卡合起来跑一个 32B

如果你后续想跑 32B 小样本，不要先全量。配置：

```bash
export MODEL_ID=Qwen/Qwen2.5-32B-Instruct-AWQ
export SERVED_MODEL_NAME=qwen25-32b-awq
export GPU_GROUPS="0,1"
export BASE_PORT=8100
```

对应 endpoint 只有一个：

```bash
ENDPOINTS="http://127.0.0.1:8100/v1"
```

先跑：

```bash
python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen32b_2gpu_smoke \
  --limit-per-task 20
```
