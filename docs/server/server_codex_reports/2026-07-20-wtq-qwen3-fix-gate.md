# 2026-07-20 Qwen3 WTQ Fix Gate and Experiment Plan

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
模型：`qwen3-32b-local` (`/home/ubuntu/models/Qwen3-32B`)
本轮起点：用户指定不要继续优先优化 TabFact，先确认当前工程是否可运行、WTQ 问题是否仍存在，以及后续怎样用较小成本跑正式实验。

## 1. 当前对齐结论

我理解当前项目目标如下：

1. myAgent 不是单纯省 token，而是通过问题类型识别、风险评分、表格压缩、选择性强校验、确定性 shortcut 等机制，在高风险题上自适应投入预算。
2. 实验主目标是同模型、同数据 split、同 evaluator 下对标 MACT：总体准确率至少不低于 MACT，token 尽量控制在 MACT 的 70% 左右，75% 可作为 gate 上限。
3. 不建议跑全量 WTQ/TabFact/CRT。全量条数约 `WTQ 4344 + TabFact 12779 + CRT 728`，MACT 又慢，正式实验应采用 frozen gate50 -> gate100 -> frozen150/200 的递进方案。

当前状态：

| 层级 | 判断 |
|---|---|
| 工程可运行性 | 基本符合要求。Qwen3 vLLM healthcheck 正常；myAgent shard 缺行、连接拒绝、eval/merged 不一致未复现。 |
| 当前策略是否能正式宣称超过 MACT | 还不能。旧 frozen WTQ100 paired 是 myAgent `64/100` vs MACT `83/100`，差距太大。 |
| 本轮修复后的 WTQ gate | 明显改善 token，accuracy 接近但仍略低于 MACT：WTQ50 direct-only `38/50`；单样本差值修复代入后投影 `39/50` vs MACT `41/50`。 |
| 后续是否值得继续 gate | 值得。WTQ token ratio 降到约 `0.596`；加上 latest verifier budget fix 后，预计仍低于 `0.75`，且准确率差距从旧 WTQ100 的 `-19 pp` 收窄到 WTQ50 投影的 `-4 pp`。 |

## 2. 之前问题是否还存在

| 问题 | 本轮状态 | 证据 |
|---|---|---|
| Qwen3 vLLM 未健康或连接拒绝 | 未复现 | `healthcheck_vllm_pool.sh` 返回 `ok`，port `8000` 正常。 |
| myAgent shard 静默失败、raw/merged/eval 缺行 | 未复现 | WTQ50 direct-only：raw `50`、merged `50`、eval `50`。 |
| 日志里 Traceback/APIConnectionError | 未复现 | WTQ50 log 扫描 `Traceback/Connection refused/APIConnectionError/Exception/failed = 0`。 |
| WTQ strong verifier 三路 direct/audit/program token 过高 | 已处理 | WTQ 非 forced strong verification 改成 direct-only。 |
| WTQ thinking verifier context 边界错误 | 已进一步处理 | 原 `nu-1328` 有 `6145 > 6144` context error；改为 verifier 单独 `max_tokens=512` 后单样本复跑无 BadRequest/context error。 |
| MACT batch partial 但 exit 0 | 仍是 MACT 侧问题 | 继续用 `scripts/server/run_mact_one_by_one.py` 规避，正式 gate 不用 MACT batch。 |

## 3. 本轮修复内容

代码改动集中在 WTQ 错误类型、evaluator 口径和 verifier 成本：

| 文件 | 改动 |
|---|---|
| `code/my_agents.py` | 新增 WTQ last chart/table entity、superlative owner shortcut；排除 election summary rows；WTQ direct-only verifier；pandas Series/DataFrame final value 安全归一；difference-between 负差值取绝对值；保护 deterministic shortcut 不被弱 verifier 覆盖。 |
| `code/selective_collaboration.py` | ThinkingSolver 支持 fenced/preamble JSON；压缩 evidence/candidate prompt；verifier 调用使用短输出预算 `max_tokens=512`。 |
| `code/model_backends.py` | OpenAI-compatible backend 新增 `.complete(..., max_tokens=...)`，允许局部覆盖 completion budget，不改变普通 `__call__` 默认 `2048`。 |
| `code/evaluate_results.py` | WTQ bool `True/False` 映射到 `yes/no`；清理 escaped quote/backslash。 |
| `tests/*` | 增加上述行为的回归测试。 |

## 4. WTQ 修复验证

### 4.1 WTQ50 direct-only 完整运行

输出：

```text
outputs/server_runs/qwen3_32b_policy_v5_patch_wtq50_directonly_20260720/
```

运行范围：`datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl` 前 50 条。

计时：

```text
START: 2026-07-20 05:35:05 CST
END:   2026-07-20 05:49:42 CST
real:  14m37.331s
```

完整性：

| item | rows |
|---|---:|
| raw | 50 |
| merged | 50 |
| eval num_samples | 50 |
| failed | 0 |
| missing | 0 |

myAgent vs MACT frozen first50：

| system | correct | accuracy | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| myAgent direct-only | 38/50 | 0.760 | 6,166.04 | 5,739.06 | 426.98 | 4.94 | 17.514 | 0 | 0 |
| MACT one-by-one | 41/50 | 0.820 | 10,346.36 | 7,317.46 | 3,028.90 | 3.42 | 113.586 | 3 | 3 |

Paired：

| both correct | myAgent only | MACT only | both wrong | McNemar p | token ratio |
|---:|---:|---:|---:|---:|---:|
| 34 | 4 | 7 | 5 | 0.5488 | 0.5960 |

解释：

- accuracy 仍低 MACT `6 pp`，不能宣称 WTQ 已超过。
- token 已明显低于 MACT，只是 MACT 有 3 个 failed/missing row，token 记 0；这个 token ratio 对 myAgent 不是宽松口径。

### 4.2 差值修复单样本验证与投影

修复项：WTQ 问句出现 `difference ... between` 时，负数差值 canonicalize 为绝对差值。

验证样本：

```text
nu-3395
question: what is the difference in balls between the first and fourth players?
gold: ["6"]
```

单样本输出：

| metric | value |
|---|---:|
| final_value | `6` |
| eval primary_accuracy | 1.0 |
| avg tokens | 6,903 |
| failed/missing | 0/0 |

把该单样本结果代入 WTQ50 direct-only 的同一 ID 位置后：

| system | correct | accuracy | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|
| myAgent projected after delta fix | 39/50 | 0.780 | 6,166.22 | 17.528 | 0 | 0 |
| MACT | 41/50 | 0.820 | 10,346.36 | 113.586 | 3 | 3 |

Paired 投影：

| both correct | myAgent only | MACT only | both wrong | McNemar p | token ratio |
|---:|---:|---:|---:|---:|---:|
| 35 | 4 | 6 | 5 | 0.7539 | 0.5960 |

注意：这是单样本实跑后的代入投影，不是完整重跑 WTQ50。

### 4.3 verifier context 修复单样本验证

原失败样本：

```text
nu-1328
question: how many times has a steam rail vehicle's speed been recorded to be above 200 kilometers per hour?
```

旧 WTQ50 direct-only 中，该样本主答案正确，但 `thinking_direct` 失败：

```text
6145 input tokens + 2048 output exceeds 8192 context; maximum input length 6144
```

本轮给 thinking verifier 单独使用 `max_tokens=512` 后复跑：

| metric | value |
|---|---:|
| final_value | `2` |
| eval primary_accuracy | 1.0 |
| `thinking_direct.is_valid` | true |
| `thinking_direct.failure` | empty |
| avg tokens | 12,208 |
| failed/missing | 0/0 |

这说明 context 边界问题已被修复。因为之前失败的 verifier 请求未计入 API usage，完整 WTQ50 未来重跑时 avg token 会略高。保守估计：6 个原 context failure 如果都成功计入，每题均值增加不超过约 `800` token，WTQ50 token ratio 仍约低于 `0.68`，仍在 `0.75` gate 内。

## 5. 验证命令

已通过：

```text
python tests/test_evaluate_results.py
Ran 15 tests: OK

python tests/test_selective_collaboration.py
Ran 8 tests: OK

python tests/test_myagent_pipeline.py
Ran 125 tests: OK

python tests/test_model_backends.py
Ran 11 tests: OK

python tests/test_server_runner.py
Ran 5 tests: OK

python tests/test_compare_blind_results.py
Ran 3 tests: OK

python -m py_compile \
  code/model_backends.py code/my_agents.py code/selective_collaboration.py \
  code/evaluate_results.py code/compare_blind_results.py scripts/server/run_sharded_tqa.py
exit 0
```

说明：曾尝试 `python -m unittest tests.test_myagent_pipeline...`，因为本仓库 `tests/` 不是 Python package，出现导入错误；最终按仓库现有方式 `python tests/test_myagent_pipeline.py` 通过。

## 6. 是否“当前项目已经符合要求”

我的判断：

1. 工程链路符合阶段要求：Qwen3 服务、runner、eval、compare、错误传播、单样本回归都能跑通。
2. 作为专利/论文的正式 MACT 对比结果，还不符合最终要求：当前没有三数据集 current-code frozen gate50 的完整 paired 结果，也不能把 WTQ50 投影写成正式超过 MACT。
3. 现在的策略已经值得进入下一轮小规模 gate：WTQ 从旧 frozen WTQ100 `-19 pp` 缩到投影 `-4 pp`，且 token 明显低于 MACT；TabFact/CRT 旧结果没有同类工程失败，但需要用 current code 重跑 gate50 统一口径。

## 7. 推荐后续实验方案

### Gate 0: 每次先做服务检查

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

bash scripts/server/healthcheck_vllm_pool.sh configs/server/qwen3_32b_2gpu_local.env
```

### Gate 1: current code 跑 myAgent frozen50/all

目的：确认当前提交后 WTQ/TabFact/CRT 三数据集同口径 50 条结果。

```bash
OUT=outputs/server_runs/qwen3_32b_current_gate50_all_$(date +%Y%m%d_%H%M%S)

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root "$OUT" \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

### Gate 2: MACT frozen50/all 用 one-by-one

不要用 MACT batch。每个数据集单独跑，失败行也会被写出并计入 eval。

WTQ 示例：

```bash
python scripts/server/run_mact_one_by_one.py \
  --mact-root /home/ubuntu/lzz/MACT \
  --dataset-path datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --output-path /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/wtq_mact.jsonl \
  --log-path /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/logs/wtq_onebyone.log \
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
  --temp-dir /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/tmp
```

TabFact 用 `--task scitab`，CRT 用 `--task crt`，其余参数相同。

### Gate 3: 是否扩大到 100/150

只有 gate50 同时满足以下条件才扩大：

| criterion | threshold |
|---|---:|
| overall accuracy | myAgent >= MACT |
| overall avg tokens | myAgent / MACT <= 0.75，目标约 0.70 |
| myAgent failed rate | <= 2% |
| per-dataset drop | 任一数据集不低于 MACT 超过 5 pp |
| dataset count | 至少两个数据集 myAgent >= MACT |

扩大顺序：

```text
current-code frozen50/all -> frozen100/all -> frozen150/all
```

不要直接跑 full dataset。正式主结果优先 frozen150；只有服务器时间充足，再补 frozen200。

### 模型筛选方案

当前本地可用模型里：

| model | status |
|---|---|
| Qwen3-32B | 当前主候选，值得继续 gate |
| Qwen2.5-3B-Instruct | 已跑 50/all，overall `84/150 = 0.56`，不建议继续 |

如果服务器后续新增模型，统一先跑 smoke20 或 gate50/all。只有 gate50 达标，才跑同模型 MACT gate50；不要每个模型都跑 MACT100/150。

### 消融实验方案

消融不要全量，建议只在最终主模型上跑 50 或 100 条：

| ablation | sample |
|---|---:|
| legacy collaboration | 50/all |
| no strong verification | 50/all |
| no deterministic shortcuts | 50/all |
| max-replan 0/1/2 | 50/all |
| optional multiview validation | 50/all 或 100/all |

消融报告只需要说明机制贡献，不要求每个消融都跑到 frozen150。

## 8. 可写入专利/论文的保守表述

可以写：

> 该系统采用风险自适应协作机制，对简单题低成本回答，对高风险题触发压缩证据下的独立校验，并通过确定性表格规则修正高频 WTQ 错误类型。小规模 frozen gate 显示，修复后的 WTQ token 明显低于 MACT，准确率差距已显著收窄，具备进入扩大配对实验的条件。

暂时不要写：

> 当前 Qwen3 myAgent 已经在所有正式设置下稳定超过 MACT。
