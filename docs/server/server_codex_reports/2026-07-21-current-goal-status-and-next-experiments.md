# 2026-07-21 Current Goal Status and Next Experiments

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
当前提交：`88598c7 Record WTQ shortcutfix2 validations`
目标：继续排查当前实验问题，筛选可超过 MACT 的模型，并给出不跑 full 多天的正式实验路径。

## 1. Current Verdict

当前本地可用模型均已完成小样本筛选：

| model | local path | status | decision |
|---|---|---|---|
| Qwen3-32B | `/home/ubuntu/models/Qwen3-32B` | passed Gate-50 and frozen150 strict paired | current main model |
| Qwen3-14B-AWQ | `/home/ubuntu/models/Qwen3-14B-AWQ` | Gate-50 runs, but below MACT | no-go for expansion |
| Qwen2.5-14B-AWQ | `/home/ubuntu/models/Qwen2.5-14B-Instruct-AWQ` | Gate-50 runs, but below MACT | no-go for expansion |
| Qwen2.5-3B-Instruct | `/home/ubuntu/models/Qwen2.5-3B-Instruct` | Gate-50 runs, but clearly below MACT | no-go for expansion |

结论：

1. 当前没有未筛选的本地候选模型；继续多模型测试前，需要先下载或挂载新模型。
2. 当前 Qwen3-32B 服务健康，非沙箱 `healthcheck_vllm_pool.sh` 返回 `ok`。
3. 之前的问题没有复现：当前 blind200 / frozen150 guard 日志未检出 `Connection refused`、context length、APIConnectionError 或 shard 缺行。
4. 当前主证据足够支持阶段 gate：Qwen3-32B frozen150 strict paired 为 myAgent `342/450 = 0.7600` vs MACT `330/450 = 0.7333`，token ratio `0.6161`，myAgent failed/missing 为 `0`。
5. 不能写成“所有模型都超过 MACT”或“full dataset 已经稳定全面超过 MACT”；目前只有 Qwen3-32B 是主候选。
6. 本轮发现并修复了一个 vLLM 管理脚本风险：`pids/server/vllm_8000.pid` 已 stale，但 port `8000` 仍有真实 Qwen3-32B listener。`start_vllm_pool.sh` 已增加 stale pid + live port 检查，避免在已有服务上重复启动。

## 2. Current Resource Constraints

磁盘状态：

```text
overlay 200G, used 197G, available 3.2G, use 99%
```

主要占用：

| path | size | note |
|---|---:|---|
| `/home/ubuntu/models` | 86G | main model storage |
| `/home/ubuntu/miniconda3` | 26G | environment |
| `/home/ubuntu/lzz` | 5.2G | repos and local datasets |
| `/home/ubuntu/.vscode-server` | 4.0G | editor runtime |
| `/home/ubuntu/.cache/vllm` | 1.9G | compile cache |
| `/home/ubuntu/lzz/MyAgent/outputs/server_runs` | 241M | experiment outputs, not the disk bottleneck |
| `/home/ubuntu/lzz/MACT/outputs` | 6.8M | MACT outputs, not the disk bottleneck |

模型目录大小：

| model dir | size |
|---|---:|
| `/home/ubuntu/models/Qwen3-32B` | 62G |
| `/home/ubuntu/models/Qwen3-14B-AWQ` | 9.4G |
| `/home/ubuntu/models/Qwen2.5-14B-Instruct-AWQ` | 9.4G |
| `/home/ubuntu/models/Qwen2.5-3B-Instruct` | 5.8G |

判断：

- 删除 `outputs/server_runs` 不能解决空间问题，只能释放约 241M。
- 如果要下载新模型，需要先删除或迁移 no-go 模型目录，或清理 vLLM compile cache；这是 destructive action，应先由用户确认。
- 保留 Qwen3-32B，因为它是当前主模型。

GPU 状态：

| gpu | memory used / total | note |
|---:|---:|---|
| 0 | 1209 / 49140 MiB | mostly free |
| 1 | 17433 / 49140 MiB | occupied by unknown process/cache |
| 2 | 17497 / 49140 MiB | occupied by unknown process/cache |
| 3 | 3 / 49140 MiB | free |
| 4 | 3 / 49140 MiB | free |
| 5 | 45815 / 49140 MiB | Qwen3-32B service |
| 6 | 45815 / 49140 MiB | Qwen3-32B service |

Qwen3-32B current service:

```text
MODEL_ID=/home/ubuntu/models/Qwen3-32B
SERVED_MODEL_NAME=qwen3-32b-local
GPU_GROUPS="5,6"
BASE_PORT=8000
VLLM_MAX_MODEL_LEN=8192
```

服务管理修复验证：

```text
before fix:
  pids/server/vllm_8000.pid = 47338
  /proc/47338 missing
  actual vLLM process = 17816, port 8000, Qwen3-32B

after fix:
  start_vllm_pool.sh removed stale pid 47338
  detected port 8000 already has a listener
  skipped duplicate start
  healthcheck_vllm_pool.sh returned ok
```

后续长跑前不要只信 pid 文件；必须跑 healthcheck。

## 3. Model Screening Summary

Strict Gate-50 MACT reference:

| dataset | MACT correct | avg tokens |
|---|---:|---:|
| WTQ | 41/50 | 10,346.36 |
| TabFact | 48/50 | 10,442.78 |
| CRT | 31/50 | 12,998.10 |
| Overall | 120/150 | 11,262.41 |

Qwen3-32B current strict Gate-50 subtotal comes from the latest WTQ shortcutfix, TabFact shortcutfix, and CRT strict paired runs:

| dataset | myAgent correct | MACT correct | myAgent avg tokens | token ratio | status |
|---|---:|---:|---:|---:|---|
| WTQ | 41/50 | 41/50 | 6,246.82 | 0.6038 | tied MACT |
| TabFact | 49/50 | 48/50 | 2,631.46 | 0.2520 | beats MACT |
| CRT | 34/50 | 31/50 | 12,920.72 | 0.9940 | beats MACT, token almost tied |
| Overall | 124/150 | 120/150 | 7,266.33 | 0.6452 | gate passed |

Other local models:

| model | WTQ | TabFact | CRT | overall | avg tokens | avg seconds | failed/missing | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-14B-AWQ | 37/50 | 44/50 | 27/50 | 108/150 = 0.720 | 7,344.51 | 6.50s | 0/0 | no-go |
| Qwen2.5-14B-AWQ | 34/50 | 45/50 | 28/50 | 107/150 = 0.713 | 7,308.35 | 6.92s | 0/0 | no-go |
| Qwen2.5-3B-Instruct | 28/50 | 40/50 | 21/50 | 89/150 = 0.593 | 7,298.51 | 4.26s | 0/0 | no-go |

The 14B and 3B models are faster, but they do not reduce average tokens versus Qwen3-32B myAgent and lose too much accuracy. They should not enter Gate-100/150/200.

## 4. Current Qwen3-32B Evidence

Strict paired main evidence:

| scope | myAgent | MACT | accuracy delta | token ratio | accepted |
|---|---:|---:|---:|---:|---|
| frozen150 strict paired | 342/450 = 0.7600 | 330/450 = 0.7333 | +2.67 pp | 0.6161 | yes |

Current-code blind200 stress test:

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ shortcutfix2 | 131/200 | 0.6550 | 6,226.93 | 15.939s | 0 | 0 |
| TabFact current | 185/200 | 0.9250 | 2,426.89 | 10.755s | 0 | 0 |
| CRT current | 137/200 | 0.6850 | 10,838.25 | 24.899s | 0 | 0 |
| Overall | 453/600 | 0.7550 | 6,497.36 | 17.198s | 0 | 0 |

限制：

- blind200 目前是 myAgent-only stress test，没有同 split Qwen3 MACT 输出。
- 因此 blind200 不能写成 strict paired 超过 MACT，只能写“current-code 稳定、总体性能可用、token 仍明显低于 frozen150 MACT 均值”。

## 5. Next Experiments

### Option A: No New Model, Formal Paired 200

如果不下载新模型，下一步最有价值的是补跑 blind200 同 split MACT，和已经完成的 myAgent blind200 组成 formal paired 200/数据集主表。

已有 myAgent 输出：

```text
outputs/server_runs/qwen3_32b_current_blind200_20260721/
outputs/server_runs/qwen3_32b_current_blind200_wtq200_shortcutfix2_20260721/
```

要补跑的 MACT 输入：

```text
datasets_ready/blind_holdout_200_v1_2026-06-27/wtq.jsonl
datasets_ready/blind_holdout_200_v1_2026-06-27/tabfact.jsonl
datasets_ready/blind_holdout_200_v1_2026-06-27/crt.jsonl
```

预计成本：

| dataset | MACT frozen150 avg sec/sample | blind200 estimated wall |
|---|---:|---:|
| WTQ | 117.2s | about 6.5h |
| TabFact | 96.4s | about 5.4h |
| CRT | 173.9s | about 9.7h |
| Total sequential | - | about 21.6h |

这不是五天级别，但仍然是长跑。建议只在磁盘和服务状态确认后，用 `tmux` 或明确后台脚本执行，并全程 `--resume`。

### Option B: New Model Screening

若用户提供或允许下载新模型，流程固定为：

1. 先释放空间：删除或迁移 no-go 模型目录，保留 Qwen3-32B。
2. 下载/挂载新模型。
3. 启动单 endpoint smoke。
4. 跑 frozen Gate-50 myAgent-only。
5. 只有 overall 接近 Qwen3-32B 或超过 MACT Gate-50，才跑 MACT paired。
6. 若 Gate-50 不过，直接记录 no-go，不进入扩大实验。

Gate-50 命令模板：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

export LOCAL_VLLM_API_KEY=local-vllm-key-change-me
RUN_ROOT=outputs/server_runs/<model_tag>_gate50_$(date +%Y%m%d_%H%M%S)

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:<port>/v1 \
  --model <served_model_name> \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$RUN_ROOT" \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11262.41 \
  --resume
```

Gate acceptance:

| criterion | threshold |
|---|---|
| raw/merged/eval completeness | 50 rows per dataset |
| failed/missing | 0 for smoke; <= 2% for larger gate |
| logs | no connection/context/API errors |
| overall accuracy | >= MACT Gate-50 or close to Qwen3-32B |
| datasets | at least 2 datasets not below MACT |
| token | <= 0.75 MACT preferred; CRT can be watched separately |

### Option C: Ablation for Patent Mechanism Explanation

Do not expand ablations to full data. Use 50 or 100 samples only:

| ablation | command switch | purpose |
|---|---|---|
| legacy collaboration | `--collaboration-mode legacy` | show selective-risk policy contribution |
| no strong verification | `--disable-strong-verification` | show verifier contribution/cost |
| no deterministic shortcuts | `--disable-deterministic-shortcuts` | show semantic shortcut contribution |
| replan budget | `--max-replan 0/1/2` | show adaptive correction effect |

Recommended first ablation table for experts:

```text
frozen Gate-50, Qwen3-32B, WTQ/TabFact/CRT, same evaluator.
```

## 6. Recommended Immediate Action

当前不建议继续盲目跑本地模型，因为没有未筛选候选。下一步按优先级：

1. 若目标是尽快写专家/专利主表：补跑 blind200 MACT strict paired，预计约 22h sequential。
2. 若目标是继续找更优模型：先释放模型磁盘空间，再下载一个新候选，只跑 Gate-50。
3. 若目标是解释机制贡献：先跑 Qwen3-32B frozen Gate-50 ablation，不跑 full。

当前阶段可写入专家材料：

```text
在本地可用模型筛选中，Qwen3-32B 是唯一通过 strict paired stage gate 的候选；
Qwen3-14B-AWQ、Qwen2.5-14B-AWQ、Qwen2.5-3B 均可稳定运行但准确率不足。
Qwen3-32B 在 frozen150 strict paired 中以 342/450 对 330/450 超过 MACT，
平均 API token 为 MACT 的 61.6%，且 failed/missing 为 0。
```

不应写：

```text
所有模型都超过 MACT。
blind200 已严格证明超过 MACT。
full 数据集已经完成。
```
