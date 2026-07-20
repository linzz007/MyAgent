# 2026-07-20 Qwen3 Current Frozen150 and Experiment Plan

服务器路径：`/home/ubuntu/lzz/MyAgent`  
分支：`codex/selective-risk-collaboration`  
提交：`6932c0f Record multi-model gate screening`  
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)  
目标：确认当前 myAgent 是否仍符合阶段要求，并给出不用全量跑完 WTQ/TabFact/CRT 的正式实验方案。

## 1. Current Understanding

当前项目目标不是单纯压 token，而是在同模型、同数据 split、同 evaluator 下，对标 MACT：

- 准确率总体至少不低于 MACT，最好略高。
- 平均 token 尽量控制在 MACT 的 `70%` 左右，`75%` 可作为 gate 上限。
- 不再继续优先优化 TabFact；优先判断三数据集总体是否值得进入更大样本和更多模型筛选。
- 不能把 first-N 200、frozen split、旧 policy v5、current shortcutfix、MACT 50 混成一个结论。

## 2. Previous Issues

本轮复核后，myAgent 侧以前担心的运行问题没有复现：

| issue | status | evidence |
|---|---|---|
| vLLM 未健康或 Connection refused | 未复现 | Qwen3 healthcheck 在 port `8000` 返回 `ok` |
| myAgent shard 缺行或提前结束 | 未复现 | tail100 三数据集 raw/merged/eval 都完整 |
| context length / BadRequestError | 未复现 | tail100 日志错误扫描为空 |
| TabFact strong verification 过度触发 | 已绕开 | 当前 TabFact 走 deterministic shortcut + selective，Gate-50 已过 |
| MACT batch partial 但 exit 0 | 仍是 MACT 侧风险 | 后续 MACT 继续只用 one-by-one runner |

注意：CRT 日志存在 pandas `SettingWithCopyWarning`，但没有 `Traceback`、`BadRequestError`、`context length`、`Connection refused`、`APIConnectionError`、`Exception` 或 failed row。

## 3. New Tail100 Run

为避免重复跑已完成的 frozen first50，本轮只补跑同一 frozen150 split 的第 `51-150` 条：

```text
outputs/server_runs/qwen3_32b_current_frozen150_tail100_20260720/input/
```

命令：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset outputs/server_runs/qwen3_32b_current_frozen150_tail100_20260720/input/wtq_tail100.jsonl \
  --tabfact-dataset outputs/server_runs/qwen3_32b_current_frozen150_tail100_20260720/input/tabfact_tail100.jsonl \
  --crt-dataset outputs/server_runs/qwen3_32b_current_frozen150_tail100_20260720/input/crt_tail100.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_current_frozen150_tail100_20260720 \
  --max-replan 2 \
  --mact-avg-tokens 11262.41
```

计时：

```text
START: 2026-07-20 20:37:16 CST
END:   2026-07-20 22:10:30 CST
real:  93m13.600s
```

完整性：

| dataset | raw | merged | eval samples | failed | missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 100 | 100 | 100 | 0 | 0 |
| TabFact | 100 | 100 | 100 | 0 | 0 |
| CRT | 100 | 100 | 100 | 0 | 0 |

Tail100 结果：

| dataset | correct | acc | avg tokens | avg seconds | avg calls | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 64/100 | 0.640 | 6,250.02 | 15.824 | 4.85 | 0 | 0 |
| TabFact | 82/100 | 0.820 | 2,670.51 | 12.084 | 3.94 | 0 | 0 |
| CRT | 63/100 | 0.630 | 12,265.77 | 27.981 | 5.86 | 0 | 0 |

相对旧 policy v5 frozen tail100：

| dataset | old tail100 | current tail100 | accuracy delta | old avg tokens | current avg tokens |
|---|---:|---:|---:|---:|---:|
| WTQ | 60/100 | 64/100 | +4 pp | 11,103.92 | 6,250.02 |
| TabFact | 76/100 | 82/100 | +6 pp | 2,622.57 | 2,670.51 |
| CRT | 57/100 | 63/100 | +6 pp | 10,452.23 | 12,265.77 |

## 4. Current Frozen150 Combined

合并来源：

- WTQ first50：`outputs/server_runs/qwen3_32b_current_gate50_wtq_shortcutfix_20260720/merged/wtq_qwen3-32b-local.jsonl`
- TabFact first50：`outputs/server_runs/qwen3_32b_current_gate50_tabfact_shortcutfix_20260720_b/merged/tabfact_qwen3-32b-local.jsonl`
- CRT first50：`outputs/server_runs/qwen3_32b_current_gate50_all_20260720_afterfix/merged/crt_qwen3-32b-local.jsonl`
- Tail100：`outputs/server_runs/qwen3_32b_current_frozen150_tail100_20260720/merged/`

合并输出：

```text
outputs/server_runs/qwen3_32b_current_frozen150_combined_20260720/
```

合并时已校验每个数据集的 `150` 个 id 顺序等于：

```text
datasets_ready/frozen_qwen3_eval_150_2026-07-19/{wtq,tabfact,crt}.jsonl
```

结果：

| dataset | merged rows | eval samples | correct | acc | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 150 | 150 | 105/150 | 0.700 | 6,248.95 | 16.138 | 0 | 0 |
| TabFact | 150 | 150 | 131/150 | 0.873 | 2,657.49 | 11.776 | 0 | 0 |
| CRT | 150 | 150 | 97/150 | 0.647 | 12,484.09 | 29.143 | 0 | 0 |
| Overall | 450 | 450 | 333/450 | 0.740 | 7,130.18 | 19.019 | 0 | 0 |

相对旧 policy v5 frozen150，current code 从 `301/450 = 0.669` 提升到 `333/450 = 0.740`，平均 token 从约 `8,361.82` 降到 `7,130.18`。

## 5. MACT Comparison Status

已完成的 strict paired Gate-50：

| system | WTQ | TabFact | CRT | overall | avg tokens |
|---|---:|---:|---:|---:|---:|
| myAgent current | 41/50 | 49/50 | 34/50 | 124/150 = 0.827 | 7,266.33 |
| MACT one-by-one | 41/50 | 48/50 | 31/50 | 120/150 = 0.800 | 11,262.41 |

Gate-50 paired 结论：

- myAgent 总体超过 MACT：`124/150` vs `120/150`。
- myAgent 平均 token 是 MACT 的 `64.52%`。
- McNemar p = `0.5572`，不能写成统计显著，只能写成 Gate-50 通过、值得扩大样本。

WTQ first100 strict paired 重新诊断：

| system | correct | acc | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| myAgent current | 78/100 | 0.780 | 6,111.74 | 16.090 | 0 | 0 |
| MACT one-by-one | 83/100 | 0.830 | 10,427.21 | 111.373 | 5 | 5 |

Paired：`both_correct=70, myAgent_only=8, MACT_only=13, both_wrong=9, McNemar p=0.3833`。  
Token ratio：`0.5861`。

这说明之前的 WTQ100 严重问题已经明显缓解：

- 旧 policy v5：myAgent `64/100` vs MACT `83/100`，accuracy gap `-19 pp`，token 还高于 MACT。
- 当前代码：myAgent `78/100` vs MACT `83/100`，accuracy gap `-5 pp`，token 约为 MACT 的 `58.6%`。

但它也说明当前还不能写成“WTQ100 已超过 MACT”。更稳妥的表述是：WTQ100 差距已收窄到 gate 边界，且 token 明显低于 MACT。

## 6. Can This Be Written To Expert Materials?

可以写入的内容：

```text
在 Qwen3-32B 本地模型上，myAgent 已完成 WTQ、TabFact、CRT 三数据集 frozen Gate-50 strict paired 评估，
总体准确率为 124/150，高于 MACT 的 120/150；平均 token 为 MACT 的 64.5%。
在扩大到 current frozen150 的 myAgent-only 验证中，三数据集合计 333/450，平均 token 为 7,130，
failed/missing 均为 0，说明当前工程链路稳定，且相对旧 policy v5 有明显提升。
```

不建议写入的内容：

```text
当前 Qwen3 myAgent 已在所有正式设置下全面超过 MACT。
当前 frozen150 strict paired 已经超过 MACT。
TabFact 已经不需要任何后续观察。
```

原因是 frozen150 的 MACT tail 尚未补齐；目前只有 Gate-50 三数据集 paired，以及 WTQ first100 paired。

## 7. Runtime Reality

按本轮和 Gate-50 观测的平均耗时估算，直接全量跑不现实：

| system | estimated full WTQ/TabFact/CRT runtime |
|---|---:|
| myAgent current | about 67.2 hours |
| MACT one-by-one | about 483.2 hours, about 20.1 days |

这还没有计入重启、排队、失败重试、上下文错误处理。全量不适合作为每次模型筛选或每次策略修改后的常规实验。

## 8. Recommended Experiment Plan

建议采用递进 gate，不直接跑 full dataset：

| stage | scope | run | purpose | stop rule |
|---|---|---|---|---|
| Smoke | 5-20 per dataset | myAgent only | 检查模型服务、格式、失败数、token 异常 | failed/missing > 0 或 token 爆炸就停 |
| Gate-50 | frozen first50 per dataset | myAgent first, then MACT only if myAgent 达标 | 快速筛模型和策略 | overall < MACT reference 或 token > 0.75 MACT 就不扩大 |
| Gate-100 | frozen first100 per dataset | selected model only | 检查 Gate-50 是否偶然 | 任一数据集低 MACT 超过 5 pp 就先诊断 |
| Frozen150 main | frozen150 per dataset | final candidates, strict paired | 专家/专利主表优先选择 | overall >= MACT, token <= 0.75 MACT, failed <= 2% |
| Stratified 200/300 | 按 dataset/risk/route/shortcut 分层抽样 | final model only | 正式材料增强说服力 | 只在 frozen150 通过后跑 |
| Full dataset | full WTQ/TabFact/CRT | optional background | 附录或长期后台任务 | 不作为常规 gate |

下一步最划算的是补齐 Qwen3 frozen150 的 MACT one-by-one，而不是跑全量：

- WTQ 还需要补 `101-150` 约 50 条。
- TabFact 还需要补 `51-150` 约 100 条。
- CRT 还需要补 `51-150` 约 100 条。
- 粗略总耗时约 9 小时，适合夜间跑。

如果 frozen150 strict paired 通过，就可以把 Qwen3-32B 作为当前主模型写入专家材料；如果不过，就不再烧全量时间，优先分析 disagreement subset。

## 9. Model Strategy

当前服务器本地可用模型里：

- Qwen3-32B：Gate-50 通过，当前唯一值得扩大到 frozen150 paired 的主候选。
- Qwen2.5-3B-Instruct：Gate-50 只有 `89/150 = 0.593`，不建议继续 MACT 对照或扩大。

如果后续新增模型，优先选择中等成本模型，例如 Qwen2.5-14B-Instruct-AWQ、Qwen3-14B 或 Qwen3-30B-A3B。每个新模型先跑同一个 frozen Gate-50 myAgent；只有通过 Gate-50 才跑 MACT paired。

## 10. Current Verdict

当前项目符合阶段性要求：

1. 工程链路稳定，当前 Qwen3 myAgent 能完整跑 WTQ/TabFact/CRT，tail100 和 combined frozen150 都没有 failed/missing。
2. Gate-50 strict paired 已经总体超过 MACT，token 明显低于 MACT。
3. 扩大到 frozen150 myAgent-only 后，结果相对旧 policy v5 明显提升。
4. WTQ100 旧问题明显缓解，但 current WTQ100 仍比 MACT 低 `5 pp`，不能宣称 WTQ 已全面超过。

因此建议：现在不要继续局部优化 TabFact，也不要跑 full dataset。先补齐 Qwen3 frozen150 的 MACT one-by-one paired；如果通过，再进入 200/300 分层正式样本和更多候选模型筛选。
