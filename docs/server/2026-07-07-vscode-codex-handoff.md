# VSCode Codex 接手文档：myAgent 服务器实验

日期：2026-07-07  
目标读者：在 VSCode 中接手本项目的 Codex  
本地仓库路径：`D:\AAAcode\code-code\agent+\myAgent-main`  
服务器项目路径：`/home/ubuntu/lzz/MyAgent`

## 1. 当前总目标

用户要完成的是 myAgent 专利项目在 WTQ、TabFact、CRT 三个表格问答/事实验证数据集上的服务器实验。

最终目标不是单纯跑通，而是：

```text
myAgent 准确率 >= MACT
token 消耗可以增加，但希望仍低于 MACT，约 70% 以内也可以接受
实验流程、模块设计、消融结果要能写进硕士论文和专利报告
```

此前已有 200 条盲测估计结果：

```text
myAgent v13: 487/600 = 81.17%, avg token 15116.43
MACT:        479/600 = 79.83%, avg token 47439.26
token ratio: 31.86%
```

但现在用户希望用服务器本地模型继续验证，不再依赖 DeepSeek API。

## 2. 数据集位置和规模

项目中已经带了适配后的 JSONL：

```text
datasets_ready/full/wtq_unseen.jsonl      4344 条
datasets_ready/full/tabfact_test.jsonl   12779 条
datasets_ready/full/crt.jsonl              728 条
```

总计约 17851 条。全量可能需要较长时间，所以推荐实验顺序：

```text
10 条/数据集 smoke
50 条/数据集测速和粗看准确率
200 条/数据集盲测估计
全量 full run
```

## 3. GitHub 分支和关键提交

远端仓库：

```text
git@github.com:linzz007/MyAgent.git
```

当前应使用分支：

```text
codex/selective-risk-collaboration
```

接手时先在服务器执行：

```bash
cd /home/ubuntu/lzz/MyAgent
git fetch origin
git checkout codex/selective-risk-collaboration
git pull
git log --oneline -5
```

必须看到这些修复中的最新提交：

```text
0138797 fix: disable qwen3 thinking for vllm runs
bdc5bce fix: create server run output directories
dee79d9 feat: package myagent server experiments
```

如果没有看到 `0138797`，说明服务器代码不是最新，先不要继续跑实验。

## 4. 服务器现状

用户服务器上已有目录：

```text
/home/ubuntu/models/Qwen3-32B
/home/ubuntu/models/Qwen2.5-3B-Instruct
/home/ubuntu/lzz/MyAgent
/home/ubuntu/lzz/qwen_manager.sh
```

已有 conda 环境：

```text
base
lzz-agent
hello-agent
openclaw
```

实验推荐使用：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
```

当前 vLLM 服务使用项目脚本启动，曾经日志显示：

```text
model: /home/ubuntu/models/Qwen3-32B
served model name: qwen3-32b-local
port: 8000
GPU: 5,6
tensor parallel size: 2
```

注意：不要同时使用 `/home/ubuntu/lzz/qwen_manager.sh start` 和项目里的 `scripts/server/start_vllm_pool.sh`，否则会抢 8000 端口。

## 5. 当前最关键问题：Qwen3 thinking 导致极慢

用户发现 30 条 smoke 快跑了一个小时，这是不正常的。

根因判断：

```text
Qwen3 默认输出 <think>
myAgent 每条题可能有多次 LLM 调用
thinking 被 planner/verifier/replan 放大，导致速度严重下降
```

已经在代码中修复：

```text
code/model_backends.py
```

对于 `model_provider=openai_compatible` 且模型名包含 `qwen3` 的请求，现在默认传：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

同时 healthcheck 也加了这个参数：

```text
scripts/server/healthcheck_vllm_pool.sh
```

接手后要先确认服务器已经 `git pull` 到 `0138797`，然后重新跑 smoke。

## 6. 下一步应该执行的命令

### 6.1 停掉旧 smoke 进程

如果用户当前前台命令还卡着，先让用户按：

```text
Ctrl+C
```

然后检查残留：

```bash
ps aux | grep code/tqa.py | grep -v grep
ps aux | grep run_sharded_tqa.py | grep -v grep
```

如果还有残留：

```bash
pkill -f "code/tqa.py"
pkill -f "run_sharded_tqa.py"
```

### 6.2 拉最新代码

```bash
cd /home/ubuntu/lzz/MyAgent
git pull
git log --oneline -5
```

确认看到：

```text
0138797 fix: disable qwen3 thinking for vllm runs
```

### 6.3 确认模型服务还活着

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

期望看到：

```text
qwen3-32b-local
```

或者至少模型服务能返回列表。

### 6.4 healthcheck，确认不再出现 `<think>`

```bash
bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
```

期望输出内容直接接近：

```text
ok
```

如果仍然出现：

```text
<think>
```

说明当前 vLLM 版本可能没有吃到 `chat_template_kwargs`，后续要改为服务端启动参数或升级/调整 vLLM。不要直接全量跑。

### 6.5 重新跑 10 条 smoke

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

rm -rf outputs/server_runs/qwen3_32b_smoke

ENDPOINTS="http://127.0.0.1:8000/v1"

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_smoke \
  --limit-per-task 10
```

这里加了 `time`，用于判断真实耗时。30 条不应该再接近 1 小时。

### 6.6 查看 smoke 结果

```bash
cat outputs/server_runs/qwen3_32b_smoke/eval/wtq_qwen3-32b-local_eval.json
cat outputs/server_runs/qwen3_32b_smoke/eval/tabfact_qwen3-32b-local_eval.json
cat outputs/server_runs/qwen3_32b_smoke/eval/crt_qwen3-32b-local_eval.json
```

检查行数：

```bash
find outputs/server_runs/qwen3_32b_smoke/raw -name "*.jsonl" -exec wc -l {} \;
wc -l outputs/server_runs/qwen3_32b_smoke/merged/*.jsonl
```

预期每个数据集 10 行。

## 7. 如何判断是否跑完

前台 smoke：

```text
终端重新出现命令提示符 -> 跑完
看到 Traceback -> 失败
一直不返回 -> 还在跑或卡住
```

检查进程：

```bash
ps aux | grep code/tqa.py | grep -v grep
ps aux | grep run_sharded_tqa.py | grep -v grep
```

检查当前输出进度：

```bash
find outputs/server_runs/qwen3_32b_smoke/raw -name "*.jsonl" -exec wc -l {} \;
```

看具体日志：

```bash
tail -100 outputs/server_runs/qwen3_32b_smoke/logs/wtq/wtq_shard00.log
tail -100 outputs/server_runs/qwen3_32b_smoke/logs/tabfact/tabfact_shard00.log
tail -100 outputs/server_runs/qwen3_32b_smoke/logs/crt/crt_shard00.log
```

## 8. 通过 smoke 后的推荐实验

### 8.1 每数据集 50 条

```bash
rm -rf outputs/server_runs/qwen3_32b_50

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_50 \
  --limit-per-task 50
```

### 8.2 每数据集 200 条

```bash
rm -rf outputs/server_runs/qwen3_32b_200

nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_200 \
  --limit-per-task 200 \
  > logs/server/qwen3_32b_200_runner.log 2>&1 &
```

看进度：

```bash
tail -f logs/server/qwen3_32b_200_runner.log
find outputs/server_runs/qwen3_32b_200/raw -name "*.jsonl" -exec wc -l {} \;
```

### 8.3 全量

全量很大，建议只在 50/200 速度和准确率都可接受后跑：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_full \
  > logs/server/qwen3_32b_full_runner.log 2>&1 &
```

断点续跑：

```bash
nohup python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints "${ENDPOINTS}" \
  --model "${SERVED_MODEL_NAME}" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_full \
  --resume \
  > logs/server/qwen3_32b_full_resume.log 2>&1 &
```

## 9. 常见问题和处理

### 9.1 `vllm: command not found`

说明没激活 `lzz-agent`：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
which vllm
```

### 9.2 `scripts/server/install_server_deps.sh: No such file`

说明服务器上是旧 MyAgent 目录，没有切到新分支。之前用户遇到过。处理方式：

```bash
cd /home/ubuntu/lzz/MyAgent
git fetch origin
git checkout codex/selective-risk-collaboration
git pull
```

如果本地改动阻止 checkout，建议备份旧目录重新 clone：

```bash
cd /home/ubuntu/lzz
mv MyAgent MyAgent_backup_$(date +%Y%m%d_%H%M%S)
git clone -b codex/selective-risk-collaboration git@github.com:linzz007/MyAgent.git MyAgent
```

### 9.3 `FileNotFoundError ... raw/wtq/wtq_shard00_out.jsonl`

这是旧版本 bug，已由提交 `bdc5bce` 修复。执行：

```bash
git pull
```

### 9.4 `unrecognized arguments: --api_key_env`

`run_sharded_tqa.py` 用短横线：

```bash
--api-key-env LOCAL_VLLM_API_KEY
```

不是：

```bash
--api_key_env LOCAL_VLLM_API_KEY
```

### 9.5 `qwen_manager.sh status` 显示 API 不可访问

之前用户遇到过，但手动测试是通的：

```bash
curl -v http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

如果这两个通，可以暂时忽略 `qwen_manager.sh status` 的 API 检查。

### 9.6 结果很慢

优先检查是否还有 `<think>`：

```bash
bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
tail -50 outputs/server_runs/qwen3_32b_smoke/logs/wtq/wtq_shard00.log
```

如果还有 `<think>`，不要跑全量。先解决 Qwen3 thinking 关闭问题。

如果 no-thinking 仍然慢，可以考虑：

```text
1. 使用 Qwen2.5-14B-AWQ，两张卡各开一个服务，提高吞吐
2. 先用 50/200 子集做论文阶段性结果
3. 晚上挂全量，并用 --resume 断点续跑
```

## 10. 下一位 Codex 的判断原则

1. 不要直接全量跑。先确认 `0138797` 已拉取、healthcheck 无 `<think>`、10 条 smoke 速度正常。
2. 所有方法对比必须使用同一评价脚本和同一数据切分。
3. 不要把 `outputs/` 提交到 GitHub。
4. 如果要修改代码，修改后至少跑：

```bash
python -m py_compile code/tqa.py code/model_backends.py scripts/server/run_sharded_tqa.py
python -m unittest discover -s tests -v
```

5. 如果要继续优化性能，优先从 Qwen3 thinking、并发/分片、模型规模和高风险题二次校验开关入手，不要只靠 prompt 猜。

## 11. 用户当前最需要的答案

用户现在关心的是：

```text
为什么 30 条跑这么久？
是否应该关 thinking？
接下来是否需要重启模型？
下一步具体执行什么？
```

当前建议：

```text
不必先重启模型；
先 git pull 到 0138797；
healthcheck 验证无 <think>；
清理旧 smoke；
重新 time 跑 10 条/数据集；
如果速度恢复，再跑 50/200；
如果仍慢，再考虑换 14B-AWQ 或服务端禁用 thinking。
```
