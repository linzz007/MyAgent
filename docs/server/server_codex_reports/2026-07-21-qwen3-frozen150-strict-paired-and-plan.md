# 2026-07-21 Qwen3 Frozen150 Strict Paired Result and Plan

服务器路径：`/home/ubuntu/lzz/MyAgent`  
分支：`codex/selective-risk-collaboration`  
提交：`e6f09bf Record Qwen3 current frozen150 plan`  
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)  
数据：`datasets_ready/frozen_qwen3_eval_150_2026-07-19/{wtq,tabfact,crt}.jsonl`

## 1. Short Verdict

当前工程链路已经稳定，之前主要运行问题没有在 myAgent 侧复现：

- Qwen3 vLLM healthcheck 持续返回 `ok`。
- myAgent frozen150 合并结果在 WTQ / TabFact / CRT 上均为 `150` 行，failed/missing 均为 `0`。
- TabFact 和 CRT 的 MACT tail one-by-one 均无 failed row。
- WTQ MACT 仍有上下文上限问题，但 one-by-one wrapper 已把失败保留为结果行，避免缺行或静默吞错。

性能结论要分两层写：

1. 核心阶段指标通过：myAgent overall `333/450 = 0.7400`，MACT `330/450 = 0.7333`；myAgent 平均 token 是 MACT 的 `61.79%`。
2. 预注册完整 acceptance 未通过：只有 CRT 高于 MACT，WTQ 低 `6.00 pp`，TabFact 低 `0.67 pp`，因此不能写成“所有数据集全面超过 MACT”。

可以写进专家/专利阶段材料的表述是：

```text
在 Qwen3-32B 本地同模型、同 frozen150 split、同 evaluator 的 strict paired 评估中，
myAgent 三数据集合计 333/450，略高于 MACT 的 330/450；平均 API token 为 MACT 的 61.8%，
且 myAgent failed/missing 为 0。结果说明风险自适应协作在总体准确率不降低的同时显著降低推理资源。
```

不建议写：

```text
当前版本已经在所有数据集上稳定全面超过 MACT，或已经统计显著优于 MACT。
```

overall McNemar p = `0.8408`，只能说明当前 frozen150 上总体略高，不支持统计显著超越。

## 2. Artifacts

myAgent combined:

```text
outputs/server_runs/qwen3_32b_current_frozen150_combined_20260720/merged/
```

MACT combined:

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_frozen150_combined_20260721/
```

主要机器结果：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_frozen150_combined_20260721/compare_current_frozen150_vs_mact.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_frozen150_combined_20260721/eval/{wtq,tabfact,crt}_mact_eval.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_frozen150_combined_20260721/eval/all_mact_eval.json
```

合并时已校验每个数据集 `150` 个 id 顺序等于 frozen split。

## 3. MACT Tail Completion

本轮补齐的是 MACT frozen150 缺口：

| dataset | input range | output rows | failed rows | wall time |
|---|---:|---:|---:|---:|
| WTQ | frozen rows 101-150 | 50 | 1 | 108m50.355s |
| TabFact | frozen rows 51-150 | 100 | 0 | 171m21.346s |
| CRT | frozen rows 51-150 | 100 | 0 | 291m21.976s |

WTQ failed row:

```text
id: nu-1330
reason: MACT produced 0 rows with returncode 0
root cause: 6145 input tokens + requested 2048 output tokens exceeds 8192 context; max input length 6144
```

这说明 MACT 的 `tqa.py` 仍会在部分样本上吞掉异常并返回 0；当前 one-by-one wrapper 的处理是正确的，因为它把该样本写成 failed row，保证 paired eval 不缺行。

## 4. Integrity Table

| dataset | system | rows | eval samples | failed | missing | avg calls | avg tokens | avg seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | myAgent | 150 | 150 | 0 | 0 | 4.83 | 6,248.95 | 16.14 |
| WTQ | MACT | 150 | 150 | 6 | 6 | 3.40 | 10,698.43 | 117.16 |
| TabFact | myAgent | 150 | 150 | 0 | 0 | 3.88 | 2,657.49 | 11.78 |
| TabFact | MACT | 150 | 150 | 0 | 0 | 3.20 | 10,722.56 | 96.41 |
| CRT | myAgent | 150 | 150 | 0 | 0 | 5.93 | 12,484.09 | 29.14 |
| CRT | MACT | 150 | 150 | 0 | 1 | 4.05 | 13,197.35 | 173.86 |
| Overall | myAgent | 450 | 450 | 0 | 0 | 4.88 | 7,130.18 | 19.02 |
| Overall | MACT | 450 | 450 | 6 | 7 | 3.55 | 11,539.45 | 129.15 |

## 5. Strict Paired Result

Paired column format: `both_correct / myAgent_only / MACT_only / both_wrong`.

| dataset | myAgent | MACT | delta | token ratio | my sec | MACT sec | paired | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| WTQ | 105/150 = 0.700 | 114/150 = 0.760 | -6.00 pp | 0.584 | 16.1 | 117.2 | 92 / 13 / 22 / 23 | 0.1755 |
| TabFact | 131/150 = 0.873 | 132/150 = 0.880 | -0.67 pp | 0.248 | 11.8 | 96.4 | 120 / 11 / 12 / 7 | 1.0000 |
| CRT | 97/150 = 0.647 | 84/150 = 0.560 | +8.67 pp | 0.946 | 29.1 | 173.9 | 70 / 27 / 14 / 39 | 0.0596 |
| Overall | 333/450 = 0.740 | 330/450 = 0.733 | +0.67 pp | 0.618 | 19.0 | 129.1 | 282 / 51 / 48 / 69 | 0.8408 |

Acceptance criteria from `code/compare_blind_results.py`:

| criterion | passed |
|---|---:|
| overall accuracy >= MACT | yes |
| at least two datasets >= MACT | no |
| token ratio <= 0.75 | yes |
| myAgent execution failure rate <= 0.02 | yes |
| full selective-risk acceptance | no |

## 6. What This Means

当前 Qwen3-32B 版本已经能支撑阶段性专家材料，但还不建议作为最终“全面优于 MACT”的主结论。

可以主张：

- 当前项目运行链路稳定，myAgent 没有 failed/missing。
- 在 frozen150 strict paired 上，总体准确率略高于 MACT。
- 平均 token 明显低于 MACT，约为 MACT 的 `61.8%`。
- CRT 上 myAgent 明显强于 MACT，是当前总体反超的主要来源。

需要保留限制：

- WTQ 仍低于 MACT `6.00 pp`，刚好超过原设计中“任一数据集不得低于 MACT 超过 5 pp”的边界。
- TabFact 基本持平但仍少 1 题。
- overall 只多 3 题，McNemar 不显著，不能写统计显著。
- MACT WTQ 有 6 个 failed/missing；这是真实运行风险，但不能用来宣称 myAgent 已在 WTQ 推理能力上超过 MACT。

## 7. Next Debugging Priority

不建议继续优先优化 TabFact。TabFact frozen150 只差 `1` 题，token ratio `0.248`，已经基本符合阶段目标。

更值得做的是 WTQ disagreement subset：

```text
WTQ paired: myAgent_only = 13, MACT_only = 22
```

下一步只分析这 35 个 discordant 样本，目标不是按样本硬编码，而是判断是否存在系统性问题：

- 表格压缩遗漏候选行或列。
- WTQ 多答案 denotation 处理仍弱。
- 数字/年份/排名类 shortcut 覆盖不足。
- 高风险样本预算仍不足，或者 fallback 触发条件太保守。

如果能用通用规则把 WTQ 多追回 2-4 题，同时不伤 CRT/TabFact，frozen150 完整 acceptance 很可能过线。

## 8. Experiment Plan Under Server-Time Constraints

不要全量跑 WTQ / TabFact / CRT，也不要每个模型都跑 MACT full。按当前实测速度，MACT one-by-one 全量会非常慢，尤其 TabFact 和 CRT 长尾明显。

推荐递进 gate：

| stage | scope | run | stop rule |
|---|---|---|---|
| Smoke | 20/数据集 | myAgent only | failed/missing > 0 或 token 异常就停 |
| Gate-50 | frozen first50/数据集 | myAgent；达标后再跑同模型 MACT | overall < MACT 参考或 token > 0.75 MACT 就不扩大 |
| Frozen150 | frozen150/数据集 | 只给 1-2 个候选模型跑 strict paired | overall >= MACT、token <= 0.75、failed <= 2%，且 per-dataset gate 不越界 |
| Disagreement fix | paired discordant subset | 只看错误类型，不全量重跑 | 只接受通用修复，不按 id/table 硬编码 |
| Formal sample | 150 或 200/数据集 | 最终模型 strict paired | 写专家/专利主表 |
| Full dataset | optional background | 只在最终稳定后后台跑 | 不作为模型筛选常规动作 |

多模型策略：

- Qwen3-32B：当前主候选，overall/token 过线，但 WTQ 需要小修或至少在报告里保留限制。
- Qwen2.5-3B-Instruct：已有 Gate-50 为 `89/150 = 0.593`，不建议继续 MACT paired。
- 新增 14B/30B/32B 类模型时，先跑同一 frozen Gate-50 myAgent；只有 myAgent 自身接近或超过当前 Qwen3 Gate-50，才值得跑该模型的 MACT paired。

正式实验建议：

1. 固定代码提交、env、模型名、dataset sha256。
2. 跑 frozen150 或 frozen200 strict paired，而不是 full。
3. 输出 per dataset、overall、token、耗时、failed/missing、route/risk strata、paired test。
4. 消融只跑 50 或 100/数据集，重点覆盖 `legacy collaboration`、`no-strong verification`、`no deterministic shortcuts`、`max-replan 0/1/2`，不把所有消融扩大到 full。

## 9. Current Bottom Line

当前项目已经满足“可以继续作为主线推进”的要求：工程稳定、overall 略高于 MACT、token 明显低于 MACT。

当前项目还没有满足“最终正式全面验收”的要求：per-dataset gate 未过，主要差距在 WTQ。

最划算的下一步不是跑 full，也不是继续 TabFact 消融，而是分析 WTQ frozen150 discordant 35 条，做通用修复后只重跑 WTQ Gate/Frozen paired，再决定是否进入 200/数据集正式样本。
