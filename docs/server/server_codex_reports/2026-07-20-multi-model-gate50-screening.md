# 2026-07-20 Multi-Model Gate-50 Screening

服务器路径：`/home/ubuntu/lzz/MyAgent`  
分支：`codex/selective-risk-collaboration`  
当前提交：`d2519c7 Record Qwen3 CRT MACT gate`  
目标：用小样本同口径 gate 判断哪些本地模型值得进入扩大实验，不直接跑全量。

## 1. 结论

当前服务器本地可用模型只有：

| model | path | status |
|---|---|---|
| Qwen3-32B | `/home/ubuntu/models/Qwen3-32B` | Gate-50 通过，当前主模型 |
| Qwen2.5-3B-Instruct | `/home/ubuntu/models/Qwen2.5-3B-Instruct` | 可运行，但 Gate-50 准确率不达标 |

Qwen3-32B 是当前唯一值得进入 Gate-100/Gate-200 或正式分层样本实验的模型。Qwen2.5-3B 速度快，但准确率明显低于 MACT 和 Qwen3-32B，不建议继续跑 MACT 对照或扩大实验。

## 2. Qwen2.5-3B Current Frozen Gate-50

运行前状态：

- Qwen3-32B 已运行在 GPU `5,6`，port `8000`。
- Qwen2.5-3B 使用 GPU `0`，port `8010`。
- GPU 0 启动前显存占用约 `2450 MiB / 49140 MiB`。
- Qwen2.5-3B healthcheck 初始失败，启动 vLLM 后通过。

启动命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

bash scripts/server/start_vllm_pool.sh configs/server/qwen25_3b_1gpu_local.env
```

实验命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen25_3b_1gpu_local.env

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8010/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen25_3b_current_frozen_gate50_20260720 \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11262.41
```

计时：

```text
START: 2026-07-20 20:03:12 CST
END:   2026-07-20 20:13:55 CST
real:  10m43.115s
```

输出：

```text
outputs/server_runs/qwen25_3b_current_frozen_gate50_20260720/
```

完整性：

| dataset | raw | merged | eval samples | failed | missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 50 | 50 | 50 | 0 | 0 |
| TabFact | 50 | 50 | 50 | 0 | 0 |
| CRT | 50 | 50 | 50 | 0 | 0 |

日志扫描未检出 `Traceback`、`BadRequestError`、`context length`、`Connection refused`、`APIConnectionError`、`Exception` 或 `failed`。

## 3. Qwen2.5-3B Eval

| dataset | correct | acc | avg tokens | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 28/50 | 0.560 | 6,593.26 | 4.92 | 3.986 | 0 | 0 |
| TabFact | 40/50 | 0.800 | 2,507.82 | 3.66 | 2.274 | 0 | 0 |
| CRT | 21/50 | 0.420 | 12,794.44 | 5.98 | 6.519 | 0 | 0 |
| Overall | 89/150 | 0.593 | 7,298.51 | 4.85 | 4.260 | 0 | 0 |

## 4. Gate Comparison

Strict frozen Gate-50 comparison:

| system | WTQ | TabFact | CRT | overall | avg tokens | avg seconds |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-32B myAgent | 41/50 | 49/50 | 34/50 | 124/150 = 0.827 | 7,266.33 | 19.799 |
| Qwen2.5-3B myAgent | 28/50 | 40/50 | 21/50 | 89/150 = 0.593 | 7,298.51 | 4.260 |
| Qwen3-32B MACT | 41/50 | 48/50 | 31/50 | 120/150 = 0.800 | 11,262.41 | 125.622 |

Token ratios:

| comparison | ratio |
|---|---:|
| Qwen2.5-3B myAgent / Qwen3 MACT | 0.6480 |
| Qwen2.5-3B myAgent / Qwen3 myAgent | 1.0044 |
| Qwen3-32B myAgent / Qwen3 MACT | 0.6452 |

判断：

- Qwen2.5-3B 的平均 token 与 Qwen3-32B myAgent 基本相同，但准确率低 `23.3 pp`。
- Qwen2.5-3B 虽然比 Qwen3-32B 快约 `4.65x`，但 WTQ 和 CRT 明显不足。
- Qwen2.5-3B 没有进入 MACT 配对或扩大样本的价值。
- Qwen3-32B 仍是当前主实验模型。

## 5. Next Model Strategy

当前服务器没有本地 Qwen2.5-14B、Qwen3-14B、Qwen3-30B-A3B 或其他中大型候选模型目录；`configs/server/qwen14b_2gpu.env.example` 只是 example。

建议下一步：

1. 不再扩大 Qwen2.5-3B。
2. 若要继续多模型筛选，优先新增一个中等成本候选模型目录，例如 Qwen2.5-14B-Instruct-AWQ 或 Qwen3-14B，再跑同一个 frozen Gate-50 myAgent。
3. 在新增模型前，Qwen3-32B 可以进入 Gate-100/Gate-200 扩大验证。
4. 正式专利实验不建议全量三数据集每次重跑；建议使用 200/数据集或 300/数据集的分层样本作为主实验，保留全量为可选后台附录。

当前阶段可写入专利/专家材料的稳妥表述：

```text
在 Qwen3-32B 本地模型上，myAgent 在 frozen Gate-50 strict paired 小实验中
以 124/150 对 120/150 超过 MACT，平均 token 为 MACT 的 64.5%。
同一流程下，小模型 Qwen2.5-3B 虽可稳定运行但准确率不足，说明该方法需要
足够强的基础模型承载风险识别、程序生成和验证策略。
```

不要写：

```text
所有候选模型均超过 MACT。
Qwen2.5-3B 也适合作为正式实验主模型。
```
