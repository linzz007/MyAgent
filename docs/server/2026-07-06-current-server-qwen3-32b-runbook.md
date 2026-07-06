# 当前服务器 Qwen3-32B 两卡运行 myAgent 指南

日期：2026-07-06

这份文档按你当前服务器目录写：

```text
/home/ubuntu/models/Qwen3-32B
/home/ubuntu/models/Qwen2.5-3B-Instruct
/home/ubuntu/lzz/MyAgent
```

结论先说清楚：现在不需要再下载主模型。先用已有的 `Qwen3-32B` 做主实验，用已有的 `Qwen2.5-3B-Instruct` 做快速连通性测试即可。后续只有在 32B 跑不动、速度过慢，或者论文需要“不同规模模型”对比时，再补下载 7B/14B。

## 1. 是否还要下载模型

推荐优先级如下：

| 用途 | 模型 | 是否现在下载 |
|---|---|---|
| 主实验 | `/home/ubuntu/models/Qwen3-32B` | 不需要，已经有 |
| 快速 smoke test | `/home/ubuntu/models/Qwen2.5-3B-Instruct` | 不需要，已经有 |
| 中等规模对比/备用 | `Qwen/Qwen2.5-14B-Instruct-AWQ` | 暂时不必 |
| 更小备用 | `Qwen/Qwen2.5-7B-Instruct` | 暂时不必 |

我的建议是先不要继续堆模型。你的核心问题是验证专利流程能否在完整数据集上超过 MACT，而不是把服务器硬盘塞满。先把 `Qwen3-32B` 的全量结果跑出来；如果速度或显存不理想，再补 14B-AWQ。

## 2. 推荐启动顺序

```text
从 GitHub 拉最新 myAgent
-> 安装 myAgent 服务器依赖
-> 使用已有 Qwen3-32B 启动 vLLM
-> myAgent 通过 http://127.0.0.1:8000/v1 调用模型
-> 先跑 smoke test
-> 再跑 full run
-> 打包 outputs/server_runs 结果发回
```

注意：模型服务和项目不是一个东西。你先启动 vLLM 模型服务，然后 myAgent 只是像调用 OpenAI API 一样调用本地地址。

## 3. 把最新 myAgent 放到服务器

如果本地代码已经推到 GitHub，服务器上执行：

```bash
cd /home/ubuntu/lzz
git clone git@github.com:linzz007/MyAgent.git MyAgent
cd /home/ubuntu/lzz/MyAgent
```

如果服务器已经有旧版：

```bash
cd /home/ubuntu/lzz/MyAgent
git pull
```

如果服务器没有 GitHub SSH key，也可以先用 HTTPS：

```bash
cd /home/ubuntu/lzz
git clone https://github.com/linzz007/MyAgent.git MyAgent
cd /home/ubuntu/lzz/MyAgent
```

## 4. 安装项目依赖

```bash
cd /home/ubuntu/lzz/MyAgent
bash scripts/server/install_server_deps.sh
source .venv-server/bin/activate
```

如果你更想复用已有 conda 环境：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
pip install -r requirements-server.txt
```

二选一即可，不要两个环境混着跑。

## 5. 创建 Qwen3-32B 本地配置

```bash
cd /home/ubuntu/lzz/MyAgent
cp configs/server/qwen3_32b_2gpu_local.env.example configs/server/qwen3_32b_2gpu_local.env
nano configs/server/qwen3_32b_2gpu_local.env
```

确认里面是：

```bash
export MODEL_ID=/home/ubuntu/models/Qwen3-32B
export SERVED_MODEL_NAME=qwen3-32b-local
export GPU_GROUPS="0,1"
export BASE_PORT=8000
export VLLM_API_KEY=local-vllm-key-change-me
export LOCAL_VLLM_API_KEY="${VLLM_API_KEY}"
```

这里的 `GPU_GROUPS="0,1"` 表示一个 vLLM 服务同时使用 GPU0 和 GPU1。32B 模型在两张 4090 上更适合这样跑。

## 6. 下载步骤

因为模型已经在 `/home/ubuntu/models/Qwen3-32B`，这一步会自动跳过下载：

```bash
source .venv-server/bin/activate
bash scripts/server/download_models.sh configs/server/qwen3_32b_2gpu_local.env
```

看到类似输出就正常：

```text
local model directory exists, skip HuggingFace download
```

## 7. 启动 vLLM

```bash
source .venv-server/bin/activate
bash scripts/server/start_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
```

看日志：

```bash
tail -f logs/server/vllm_8000.log
```

看显卡：

```bash
watch -n 2 nvidia-smi
```

健康检查：

```bash
bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
```

如果返回模型回答，说明本地模型服务已经能被 myAgent 调用。

如果你想继续使用服务器上已有的 `/home/ubuntu/lzz/qwen_manager.sh`，也可以：

```bash
cd /home/ubuntu/lzz
./qwen_manager.sh start
./qwen_manager.sh status
```

这时不要再执行 `scripts/server/start_vllm_pool.sh`，否则两个 vLLM 服务会抢同一个端口。已有管理脚本通常会把模型名暴露成模型路径，所以后续运行 myAgent 时把 `--model` 改成：

```bash
--model "/home/ubuntu/models/Qwen3-32B"
```

而不是：

```bash
--model "${SERVED_MODEL_NAME}"
```

## 8. 跑 myAgent smoke test

```bash
source .venv-server/bin/activate
source configs/server/qwen3_32b_2gpu_local.env

ENDPOINTS="http://127.0.0.1:8000/v1"

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen3_32b_smoke \
  --limit-per-task 10
```

看结果：

```bash
cat outputs/server_runs/qwen3_32b_smoke/eval/wtq_qwen3-32b-local_eval.json
cat outputs/server_runs/qwen3_32b_smoke/eval/tabfact_qwen3-32b-local_eval.json
cat outputs/server_runs/qwen3_32b_smoke/eval/crt_qwen3-32b-local_eval.json
```

## 9. 跑完整数据集

smoke test 通过后再跑全量：

```bash
source .venv-server/bin/activate
source configs/server/qwen3_32b_2gpu_local.env

ENDPOINTS="http://127.0.0.1:8000/v1"

nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen3_32b_full \
  > logs/server/qwen3_32b_full_runner.log 2>&1 &
```

查看进度：

```bash
tail -f logs/server/qwen3_32b_full_runner.log
find outputs/server_runs/qwen3_32b_full/raw -name "*.jsonl" -exec wc -l {} \;
```

中断后续跑：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --output-root outputs/server_runs/qwen3_32b_full \
  --resume \
  > logs/server/qwen3_32b_full_resume.log 2>&1 &
```

## 10. 打包结果发回

```bash
cd /home/ubuntu/lzz/MyAgent
tar -czf qwen3_32b_full_results.tar.gz \
  outputs/server_runs/qwen3_32b_full \
  logs/server/qwen3_32b_full_runner.log
```

你把这个文件和下面三份 eval JSON 发回来即可：

```text
outputs/server_runs/qwen3_32b_full/eval/wtq_qwen3-32b-local_eval.json
outputs/server_runs/qwen3_32b_full/eval/tabfact_qwen3-32b-local_eval.json
outputs/server_runs/qwen3_32b_full/eval/crt_qwen3-32b-local_eval.json
```

## 11. 如果 Qwen3-32B 启动失败

先把上下文长度降到 4096：

```bash
nano configs/server/qwen3_32b_2gpu_local.env
```

改成：

```bash
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.82
```

然后重启：

```bash
bash scripts/server/stop_vllm_pool.sh
bash scripts/server/start_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
```

如果仍然 OOM，再考虑下载 14B-AWQ 作为主实验备用模型。
