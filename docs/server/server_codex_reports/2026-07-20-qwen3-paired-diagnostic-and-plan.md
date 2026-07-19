# 2026-07-20 Qwen3 Paired Diagnostic and Experiment Plan

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
报告起点 HEAD：`fea1180470eeccb3d7fe578fb969f810c7abf692` (`Freeze Qwen3 paired evaluation split`)
关键代码基线：包含用户指定的 `33663ab42366000fd94bc884f0f3d200d71bf756` (`Improve TabFact deterministic verification`)
模型：`qwen3-32b-local`

## 1. 当前判断

工程可运行性：符合阶段要求。当前 myAgent 能在 WTQ / TabFact / CRT 三个数据集跑完 200 条和 frozen150，raw、merged、eval 行数完整，失败数为 0，vLLM healthcheck 正常。

正式实验结论：还不符合“严格同 split 超过 MACT”的要求。已有 first-N 200/数据集结果相对现有 MACT 50/数据集参考是总体更高且 token 更低，但 frozen split 上的 WTQ 前 100 条 paired 诊断显示 MACT 显著更好，不能写成当前项目已经证明超过 MACT。

建议：不要继续优先优化 TabFact。当前最需要处理的是 WTQ frozen split 上 myAgent 相比 MACT 的准确率和 token 双重劣势，然后再用 gate50/gate100 选择是否值得正式跑 150/200。

## 2. 已完成 myAgent 结果

### 2.1 first-N 200/数据集

报告：`docs/server/server_codex_reports/2026-07-19-qwen3-policy-v5-200-all.md`

| dataset | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ | 134/200 | 0.670 | 11,317.45 | 20.860 | 0 | 0 |
| TabFact | 166/200 | 0.830 | 2,680.65 | 12.833 | 0 | 0 |
| CRT | 129/200 | 0.645 | 9,739.92 | 22.179 | 0 | 0 |
| Overall | 429/600 | 0.715 | 7,912.67 | 18.624 | 0 | 0 |

相对现有 MACT 50/数据集参考：overall accuracy `71.50%` vs `66.67%`，overall avg tokens 是 MACT 的 `69.04%`。这只能作为阶段参考，不是正式配对结论。

### 2.2 frozen150

输出目录：

```text
outputs/server_runs/qwen3_32b_policy_v5_frozen150_20260719/
```

数据集：

```text
datasets_ready/frozen_qwen3_eval_150_2026-07-19/
```

| dataset | raw rows | merged rows | eval samples | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 150 | 150 | 150 | 90/150 | 0.600 | 11,245.03 | 22.091 | 0 | 0 |
| TabFact | 150 | 150 | 150 | 120/150 | 0.800 | 2,778.90 | 12.166 | 0 | 0 |
| CRT | 150 | 150 | 150 | 91/150 | 0.607 | 11,061.53 | 26.987 | 0 | 0 |
| Overall | 450 | 450 | 450 | 301/450 | 0.669 | 8,361.82 | 20.415 | 0 | 0 |

计时：

```text
START: 2026-07-19 21:51:04 CST
END:   2026-07-20 00:24:16 CST
real:  153m11.982s
```

## 3. MACT 问题和处理

直接 batch 跑 MACT 仍有问题：

| attempt | status | rows | observed issue |
|---|---|---:|---|
| `max_tokens=2048` | partial | 11 | `6145 input tokens + 2048 output` exceeds Qwen3 context; MACT `tqa.py` catches exception and exits 0 |
| `max_tokens=1024` | partial | 32 | `7169 input tokens + 1024 output` exceeds context; still partial exit 0 |

因此新增了 one-by-one runner：

```text
scripts/server/run_mact_one_by_one.py
tests/test_run_mact_one_by_one.py
```

它逐条调用 MACT，保证每个输入样本都有一行输出；context error、timeout 或异常会写成 failed row，计入 `num_failed_exec` 和 `num_missing_answer`，而不是让整批静默提前结束。runner 支持 `--limit`，后续可以直接跑 gate50/gate100。

## 4. frozen WTQ100 paired 诊断

为节省服务器时间，MACT one-by-one 跑到 frozen WTQ 前 100 条后手动停止。停止原因是该子实验已经足够暴露当前策略不达标风险；不是服务异常。停止时无残留 MACT 子进程。

输出：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_frozen150_20260719_onebyone_2048/wtq_mact.jsonl
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_frozen150_20260719_onebyone_2048/wtq_mact_eval_partial100.json
outputs/server_runs/qwen3_32b_policy_v5_frozen150_20260719/compare_wtq100_myagent_vs_mact_onebyone2048.json
```

| dataset/scope | system | rows | correct | accuracy | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ frozen first 100 | myAgent | 100 | 64/100 | 0.640 | 11,586.69 | 11,068.63 | 518.06 | 6.10 | 22.705 | 0 | 0 |
| WTQ frozen first 100 | MACT | 100 | 83/100 | 0.830 | 10,427.21 | 7,461.88 | 2,965.33 | 3.34 | 111.373 | 5 | 5 |

Paired counts:

| both correct | myAgent only | MACT only | both wrong | McNemar exact p | token ratio myAgent/MACT |
|---:|---:|---:|---:|---:|---:|
| 58 | 6 | 25 | 11 | 0.000878 | 1.111 |

解释：

1. frozen WTQ100 上 myAgent accuracy 低 MACT `19.0 pp`。
2. myAgent avg tokens 反而高于 MACT `11.1%`。
3. MACT 有 5 个 context/error failed row；这些 failed row 的 token 记为 0，使 MACT 平均 token 更低。这个口径对 myAgent 更严格，但即使如此准确率差距也很明显。
4. 所以当前 Qwen3 policy v5 不能作为正式“超过 MACT”的版本。

## 5. 之前问题是否还存在

| 问题 | 当前状态 |
|---|---|
| myAgent 连接 vLLM 失败或 shard 缺行 | 未复现；200/all 和 frozen150 都完整 |
| myAgent eval、merged 行数不一致 | 未复现；raw/merged/eval 都对齐 |
| TabFact strong verification token 过高 | 当前 policy v5 已避开；TabFact 200 strong applied 为 0 |
| MACT batch partial 但 exit 0 | 仍存在；已用 one-by-one runner 规避 |
| 当前项目能否直接写正式超过 MACT | 不能；frozen WTQ100 paired 诊断不支持 |

## 6. 后续实验方案

### Gate A: 服务和 myAgent smoke

每个候选模型只先跑 20 或 50 条/数据集，目标是检查服务、失败数、token 是否异常。

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/<model>_policy_v5_smoke50_all \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

### Gate B: 同 split MACT gate50

不要直接跑全量 MACT。用同一个 frozen split 的前 50 条/数据集做 paired gate。

```bash
python scripts/server/run_mact_one_by_one.py \
  --mact-root /home/ubuntu/lzz/MACT \
  --dataset-path datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --output-path /home/ubuntu/lzz/MACT/outputs/server_runs/<model>_gate50/wtq_mact.jsonl \
  --log-path /home/ubuntu/lzz/MACT/outputs/server_runs/<model>_gate50/logs/wtq_onebyone.log \
  --task wtq \
  --plan-model-name "$SERVED_MODEL_NAME" \
  --code-model-name "$SERVED_MODEL_NAME" \
  --model-provider openai_compatible \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key-env LOCAL_VLLM_API_KEY \
  --thinking disabled \
  --temperature 0 \
  --max-tokens 2048 \
  --api-timeout 180 \
  --api-max-retries 5 \
  --plan-sample 1 \
  --code-sample 1 \
  --max-step 3 \
  --max-actual-step 3 \
  --limit 50 \
  --temp-dir /home/ubuntu/lzz/MACT/outputs/server_runs/<model>_gate50/tmp
```

TabFact 用 `--task scitab`，CRT 用 `--task crt`，其余参数相同。

### Gate C: 是否进入正式 150/200

只有 gate50 同时满足这些条件才进入 150 或 200：

| criterion | threshold |
|---|---:|
| overall accuracy | myAgent >= MACT |
| overall avg tokens | myAgent / MACT <= 0.75 |
| failed rate | <= 2% |
| per-dataset accuracy | 任一数据集不得低于 MACT 超过 5 pp |
| dataset count | 至少两个数据集 myAgent >= MACT |

当前 Qwen3 policy v5 在 frozen WTQ100 已经不满足这些条件，因此不建议继续花服务器时间跑完整 MACT150/200，除非先修 WTQ 或换模型后重新 gate。

### Gate D: 正式实验规模

正式主结果建议只给最有希望的 1 到 2 个模型跑：

| scope | use case |
|---|---|
| frozen50 | 模型/策略筛选 |
| frozen100 | 候选确认 |
| frozen150 | 正式主结果，成本可控 |
| frozen200 | 时间充足时作为增强版主结果 |
| full dataset | 不建议作为常规实验；MACT 全量预计远超一周 |

消融实验只跑 50 或 100 条/数据集，不要跑全量：

- legacy collaboration
- no strong verification
- no deterministic shortcuts
- max-replan 0/1/2
- optional multiview validation

## 7. 下一步建议

1. 先分析 WTQ frozen100 中 `mact_only=25` 的样本，定位 myAgent 是检索、压缩、路由还是执行失败。
2. 若目标是尽快写专利/论文实验，可以保留当前方法论，但实验结论写成“当前框架可运行，初始策略在 reference gate 上有 token 优势；strict frozen paired 仍需 WTQ 修正”，不要写成最终超过 MACT。
3. 若要继续模型筛选，先用 gate50 跑新模型，不再跑全量；只有 gate50 过线才扩到 100/150。
