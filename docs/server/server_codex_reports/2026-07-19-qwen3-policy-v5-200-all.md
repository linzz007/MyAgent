# 2026-07-19 Qwen3 Policy v5 200-Sample All-Dataset Validation

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
代码基线：`33663ab42366000fd94bc884f0f3d200d71bf756` (`Improve TabFact deterministic verification`)
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
任务：当前 myAgent policy v5 在 WTQ / TabFact / CRT 各 200 条同口径验证

> 2026-07-20 update: 本报告中的 MACT 对比是相对已有 Qwen3 MACT 50/数据集参考，不是同 frozen split 的正式配对结论。后续 frozen WTQ100 paired 诊断显示当前 Qwen3 policy v5 不能直接宣称严格超过 MACT；见 `docs/server/server_codex_reports/2026-07-20-qwen3-paired-diagnostic-and-plan.md`。

## 1. 结论

当前 myAgent 在三数据集合计 600 条上的 primary accuracy 为 **429/600 = 71.50%**。相对当前可用的 Qwen3 MACT 50/数据集参考基线 **100/150 = 66.67%**，总体准确率高 **+4.83 个百分点**。

总体 avg total tokens 为 **7,912.67**，是 MACT 参考均值 **11,460.75** 的 **69.04%**，低 **30.96%**。按用户给出的“约 70% MACT token 可接受”尺度看，总体 token 仍明显低于 MACT。

需要保留限制：服务器当前没有同一 200 条样本的 Qwen3 MACT 结果；因此这里的“超过 MACT”是相对现有 Qwen3 MACT 50/数据集参考，不是严格同 200 样本配对结论。

## 2. 运行命令

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_policy_v5_200_all \
  --limit-per-task 200 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

计时：

```text
START: 2026-07-19 14:19:30 CST
END:   2026-07-19 17:25:51 CST
real:  186m20.796s
```

vLLM healthcheck 在实验前后均返回 `ok`，模型名为 `qwen3-32b-local`。

## 3. 输出完整性

输出根目录：

```text
outputs/server_runs/qwen3_32b_policy_v5_200_all/
```

行数核对：

| dataset | raw rows | merged rows | eval num_samples | log tail |
|---|---:|---:|---:|---|
| WTQ | 200 | 200 | 200 | `Finished sample 200/200` |
| TabFact | 200 | 200 | 200 | `Finished sample 200/200` |
| CRT | 200 | 200 | 200 | `Finished sample 200/200` |

日志扫描未检出：

```text
Traceback
Connection refused
APIConnectionError
Error processing
Exception
FAILED/failed
```

日志中存在 pandas `SettingWithCopyWarning`，但未导致 shard 失败或缺行。

## 4. myAgent Eval 结果

| dataset | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 134/200 | 0.670 | 11,317.45 | 10,847.90 | 469.55 | 6.000 | 20.860 | 0 | 0 |
| TabFact | 166/200 | 0.830 | 2,680.65 | 2,348.30 | 332.35 | 3.875 | 12.833 | 0 | 0 |
| CRT | 129/200 | 0.645 | 9,739.92 | 9,218.18 | 521.74 | 5.365 | 22.179 | 0 | 0 |
| Overall | 429/600 | 0.715 | 7,912.67 | 7,471.46 | 441.21 | 5.080 | 18.624 | 0 | 0 |

补充分布：

| dataset | route distribution | risk distribution | strong applied | shortcut |
|---|---|---|---:|---:|
| WTQ | `COMPLEX=178, SIMPLE=22` | `high=179, light=12, medium=8, fallback=1` | 180/200 | 7/200 |
| TabFact | `COMPLEX=197, SIMPLE=3` | `high=163, medium=37` | 0/200 | 15/200 |
| CRT | `COMPLEX=198, SIMPLE=2` | `high=132, medium=64, light=3, fallback=1` | 133/200 | 8/200 |

## 5. MACT Qwen3 参考基线

当前可用 MACT 基线来自：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_50/
```

| dataset | correct | accuracy | avg tokens | avg prompt | avg completion | avg llm calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 23/50 | 0.460 | 10,946.00 | 8,343.76 | 2,602.24 | 3.760 | 97.567 | 1 | 3 |
| TabFact | 44/50 | 0.880 | 11,051.98 | 7,869.24 | 3,182.74 | 3.280 | 118.543 | 0 | 0 |
| CRT | 33/50 | 0.660 | 12,384.26 | 8,330.22 | 4,054.04 | 3.840 | 151.058 | 0 | 0 |
| Overall | 100/150 | 0.667 | 11,460.75 | 8,181.07 | 3,279.67 | 3.627 | 122.389 | 1 | 3 |

## 6. 对比判断

| dataset | accuracy delta | token ratio vs MACT | token delta |
|---|---:|---:|---:|
| WTQ | +21.0 pp | 103.39% | +3.39% |
| TabFact | -5.0 pp | 24.25% | -75.75% |
| CRT | -1.5 pp | 78.65% | -21.35% |
| Overall | +4.83 pp | 69.04% | -30.96% |

判断：

1. 总体准确率相对现有 Qwen3 MACT 50 参考是超过的：71.50% vs 66.67%。
2. 总体 token 仍明显低于 MACT，达到 MACT 参考均值的 69.04%。
3. 分数据集看，优势不均衡：WTQ 准确率明显更高但 token 略高于 MACT；TabFact token 大幅下降但 accuracy 低于 MACT；CRT accuracy 略低于 MACT，token 低约 21.35%。
4. 因为 MACT 缺少同 200 条配对结果，不能写成“同 200 样本稳定全面超过 MACT”。更稳妥表述是：当前 myAgent 在 200/数据集验证上的总体准确率高于现有 Qwen3 MACT 50/数据集参考，同时总体 token 约为 MACT 的 69%。

## 7. 本轮提交范围

本轮只提交本报告文件：

```text
docs/server/server_codex_reports/2026-07-19-qwen3-policy-v5-200-all.md
```

不提交实验输出和既有本地改动：

```text
outputs/server_runs/*
configs/server/qwen3_32b_2gpu_local.env.example
configs/server/qwen3_32b_2gpu_local.env.bak.20260709_191806
restart_qwen3_context_try.sh
```
