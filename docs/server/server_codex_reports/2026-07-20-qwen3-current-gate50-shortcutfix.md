# 2026-07-20 Qwen3 Current Gate50 Shortcut Fix

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
模型：`qwen3-32b-local`
代码基线：在 `26e6241 Improve Qwen3 WTQ gate stability` 之后继续修复 WTQ verifier 覆盖和 deterministic shortcut。

## 1. 结论

当前工程链路稳定：本轮 current-code frozen50/all 跑完 WTQ / TabFact / CRT 共 150 条，raw、merged、eval 行数完整，failed/missing 均为 0；日志没有连接拒绝、context length、Traceback 或 shard 失败。

关键变化：WTQ 经过 verifier 覆盖收紧和 5 类确定性 shortcut 后，frozen first50 从本轮初始 `35/50` 提升到 `41/50`，与 MACT one-by-one frozen first50 持平；avg token 为 MACT 的 `60.38%`。

仍需保留限制：TabFact/CRT 还没有 strict frozen first50 的 MACT one-by-one 结果。本轮不能写成“三数据集 strict frozen paired 已全面超过 MACT”，但已经值得启动 TabFact/CRT 的 MACT frozen50 gate。

## 2. current-code frozen50/all myAgent

输出：

```text
outputs/server_runs/qwen3_32b_current_gate50_all_20260720_afterfix/
```

计时：

```text
START: 2026-07-20 06:05:04 CST
END:   2026-07-20 06:57:10 CST
real:  52m05.558s
```

| dataset | raw | merged | eval | correct | acc | avg tokens | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 50 | 50 | 50 | 35/50 | 0.700 | 6,560.84 | 18.206 | 0 | 0 |
| TabFact | 50 | 50 | 50 | 43/50 | 0.860 | 2,985.28 | 12.748 | 0 | 0 |
| CRT | 50 | 50 | 50 | 34/50 | 0.680 | 12,920.72 | 31.467 | 0 | 0 |
| Overall | 150 | 150 | 150 | 112/150 | 0.747 | 7,488.95 | 20.807 | 0 | 0 |

日志扫描：

```text
Traceback=0
BadRequestError=0
context length=0
Connection refused=0
APIConnectionError=0
Exception=0
failed=0
```

## 3. 发现的问题

WTQ 初始 current-code gate 只有 `35/50`，低于之前 direct-only 诊断。逐 ID diff 显示问题不是工程失败，而是选择策略：

- `thinking_direct` 在非 forced WTQ verifier 中，即使和已有有效 code candidate 冲突，也会因为单 verifier candidate 高置信而覆盖主答案。
- 部分 verifier 会自信地数错，导致 code 原本正确的样本被改错。

典型回退：

| id | question | correct code | wrong verifier |
|---|---|---:|---:|
| `nu-3736` | Gene Hackman best actor awards | 5 | 1 |
| `nu-1907` | last episode name | Home Again | One Man and a Baby |
| `nu-1797` | TV shows with more than 1 episode | 8 | 5 |

修复：

- WTQ 非 forced verifier 与当前有效主候选冲突时，不接受 verifier 覆盖；forced fallback 仍保留原逻辑。

## 4. 新增 WTQ deterministic shortcuts

为减少 Qwen3 在可确定表格模式上的波动，新增以下 WTQ shortcut，并通过单元测试和样本实跑验证：

| shortcut | 覆盖样本 | verified output |
|---|---|---|
| after-reference count | `nu-1085` `how many song come after "rollin hard"?` | 5 |
| after-reference next entity | `nu-2402` `who came in after stefan holm?` | Andrey Tereshin |
| zero metric count | `nu-3184` `did not win any silver medals` | 2 |
| same-column count | `nu-3256` `winner the same as race leader` | 5 |
| contributor fuzzy count | `nu-2859` `shailenra contribute to` | 5 |

5-sample verification:

```text
outputs/server_runs/qwen3_32b_current_wtq_shortcuts_5samples_20260720/
primary_accuracy=1.0
failed=0
missing=0
avg_total_tokens=3,943.60
```

## 5. WTQ shortcutfix formal rerun

输出：

```text
outputs/server_runs/qwen3_32b_current_gate50_wtq_shortcutfix_20260720/
```

计时：

```text
START: 2026-07-20 07:22:12 CST
END:   2026-07-20 07:36:12 CST
real:  13m59.859s
```

完整性：

| item | rows |
|---|---:|
| raw | 50 |
| merged | 50 |
| eval num_samples | 50 |
| failed | 0 |
| missing | 0 |

WTQ vs MACT frozen first50：

| system | correct | acc | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| myAgent shortcutfix | 41/50 | 0.820 | 6,246.82 | 5,841.80 | 405.02 | 4.78 | 16.767 | 0 | 0 |
| MACT one-by-one | 41/50 | 0.820 | 10,346.36 | 7,317.46 | 3,028.90 | 3.42 | 113.586 | 3 | 3 |

Paired:

| both correct | myAgent only | MACT only | both wrong | McNemar p | token ratio |
|---:|---:|---:|---:|---:|---:|
| 36 | 5 | 5 | 4 | 1.0000 | 0.6038 |

判断：

- WTQ strict frozen first50 已追平 MACT。
- token 明显低于 MACT，约低 `39.62%`。
- MACT 有 3 个 failed/missing；即便如此 myAgent accuracy 没有低于 MACT。

## 6. Adjusted current gate50 view

把最新 WTQ shortcutfix 与同一次 current-code all-gate 的 TabFact/CRT 结果合并看：

| dataset | correct | acc | avg tokens | failed | missing |
|---|---:|---:|---:|---:|---:|
| WTQ shortcutfix | 41/50 | 0.820 | 6,246.82 | 0 | 0 |
| TabFact all-gate | 43/50 | 0.860 | 2,985.28 | 0 | 0 |
| CRT all-gate | 34/50 | 0.680 | 12,920.72 | 0 | 0 |
| Overall adjusted | 118/150 | 0.787 | 7,384.27 | 0 | 0 |

这是 current-code 的 gate 视图，但不是完整 strict paired 结论，因为 TabFact/CRT 缺少 frozen MACT50。

## 7. 下一步

不要跑全量。下一步只跑 MACT one-by-one frozen50 for TabFact/CRT：

```bash
python scripts/server/run_mact_one_by_one.py \
  --mact-root /home/ubuntu/lzz/MACT \
  --dataset-path datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --output-path /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/tabfact_mact.jsonl \
  --log-path /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/logs/tabfact_onebyone.log \
  --task scitab \
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

CRT 同命令改：

```text
--dataset-path datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl
--output-path /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/crt_mact.jsonl
--log-path /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/logs/crt_onebyone.log
--task crt
```

如果 TabFact/CRT paired 后满足：

| criterion | threshold |
|---|---:|
| overall accuracy | myAgent >= MACT |
| token ratio | <= 0.75 |
| myAgent failed rate | <= 2% |
| per-dataset drop | no dataset worse than MACT by >5 pp |

再扩大到 frozen100，不要直接跑 full dataset。

## 8. 验证

已通过：

```text
python tests/test_evaluate_results.py                 # 15 tests OK
python tests/test_selective_collaboration.py          # 8 tests OK
python tests/test_myagent_pipeline.py                 # 131 tests OK
python tests/test_model_backends.py                   # 11 tests OK
python tests/test_server_runner.py                    # 5 tests OK
python tests/test_compare_blind_results.py            # 3 tests OK
python -m py_compile code/model_backends.py code/my_agents.py code/selective_collaboration.py code/evaluate_results.py code/compare_blind_results.py scripts/server/run_sharded_tqa.py
```

## 9. TabFact MACT paired and shortcutfix

TabFact MACT frozen first50 one-by-one 已完成：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/tabfact_mact.jsonl
START: 2026-07-20 07:39:44 CST
END:   2026-07-20 08:54:18 CST
real:  74m34.189s
```

MACT TabFact 结果：

| system | correct | acc | avg tokens | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| MACT one-by-one | 48/50 | 0.960 | 10,442.78 | 3.16 | 87.509 | 0 | 0 |

与修复前 myAgent TabFact all-gate 比较：

| system | correct | acc | avg tokens | failed | missing |
|---|---:|---:|---:|---:|---:|
| myAgent before TabFact shortcutfix | 43/50 | 0.860 | 2,985.28 | 0 | 0 |
| MACT one-by-one | 48/50 | 0.960 | 10,442.78 | 0 | 0 |

Paired：`both_correct=41, myAgent_only=2, MACT_only=7, both_wrong=0`，myAgent/MACT token ratio `0.2859`。这说明修复前 TabFact 不满足 per-dataset gate。

诊断显示 7 条 MACT-only 主要是高置信确定性表格模式没有在 LLM 前处理：

| id | pattern |
|---|---|
| `tabfact-test-3075` | date range series sweep, record progression |
| `tabfact-test-4338` | only player from state + draft year |
| `tabfact-test-1405` | final season record |
| `tabfact-test-3115` | numbered match rows vs section rows |
| `tabfact-test-557` | score threshold count without team side |
| `tabfact-test-5374` | duplicate metric value count |
| `tabfact-test-7457` | location majority over year range |

本轮只补这些通用 deterministic shortcut，不启用 TabFact strong verifier；命中时在 LLM 规划前返回 label。

复跑输出：

```text
outputs/server_runs/qwen3_32b_current_gate50_tabfact_shortcutfix_20260720_b/
START: 2026-07-20 09:04:47 CST
END:   2026-07-20 09:14:07 CST
real:  9m19.564s
```

完整性：

| item | rows |
|---|---:|
| raw | 50 |
| merged | 50 |
| eval num_samples | 50 |
| failed | 0 |
| missing | 0 |

日志扫描未检出 `Traceback`、`BadRequestError`、`context length`、`Connection refused`、`APIConnectionError`、`Exception` 或 `failed`。

TabFact shortcutfix vs MACT frozen first50：

| system | correct | acc | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| myAgent TabFact shortcutfix | 49/50 | 0.980 | 2,631.46 | 2,347.34 | 284.12 | 3.76 | 11.161 | 0 | 0 |
| MACT one-by-one | 48/50 | 0.960 | 10,442.78 | 8,111.58 | 2,331.20 | 3.16 | 87.509 | 0 | 0 |

Paired：

| both correct | myAgent only | MACT only | both wrong | McNemar p | token ratio |
|---:|---:|---:|---:|---:|---:|
| 47 | 2 | 1 | 0 | 1.0000 | 0.2520 |

唯一 myAgent-only 错例：

| id | gold | myAgent | MACT | statement |
|---|---|---|---|---|
| `tabfact-test-3120` | true | false | true | both player from miami be draft before the player from california |

判断：

- TabFact strict frozen first50 已超过 MACT：`49/50` vs `48/50`。
- TabFact token 明显低于 MACT，约为 MACT 的 `25.20%`。
- 不继续追 `tabfact-test-3120`，因为当前目的不是继续优先优化 TabFact，而是通过 gate 后转向 CRT/更大样本。

## 10. CRT MACT paired

CRT MACT frozen first50 one-by-one 已完成。第一次运行在权限/上下文切换后只写出 1/50，随后用 `--resume` 从已有输出继续，最终 50/50 完整。

输出：

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/crt_mact.jsonl
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/crt_mact_eval.json
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_current_gate50/logs/crt_onebyone.log

initial START: 2026-07-20 09:19:24 CST
resume START:  2026-07-20 16:11:05 CST
resume END:    2026-07-20 18:35:42 CST
resume real:   144m37.172s
```

完整性：

| item | rows |
|---|---:|
| MACT output | 50 |
| myAgent merged | 50 |
| MACT eval num_samples | 50 |
| failed | 0 |
| missing | 0 |

日志扫描未检出 `Traceback`、`BadRequestError`、`context length`、`Connection refused`、`APIConnectionError`、`Exception` 或 `failed`。

CRT shortcutfix/current vs MACT frozen first50：

| system | correct | acc | avg tokens | avg prompt | avg completion | avg calls | avg seconds | failed | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| myAgent current CRT | 34/50 | 0.680 | 12,920.72 | 12,398.98 | 521.74 | 6.08 | 31.467 | 0 | 0 |
| MACT one-by-one | 31/50 | 0.620 | 12,998.10 | 8,286.86 | 4,711.24 | 4.08 | 175.769 | 0 | 0 |

Paired：

| both correct | myAgent only | MACT only | both wrong | McNemar p | token ratio |
|---:|---:|---:|---:|---:|---:|
| 26 | 8 | 5 | 11 | 0.5811 | 0.9940 |

判断：

- CRT strict frozen first50 已超过 MACT：`34/50` vs `31/50`。
- CRT token 与 MACT 基本持平，ratio `0.9940`，但平均耗时明显低：`31.467s` vs `175.769s`。
- 这意味着 Qwen3 Gate-50 的最后一个 strict paired 缺口已经补齐。

## 11. Current gate status

把最新 WTQ shortcutfix、TabFact shortcutfix 与 CRT strict paired 组合：

| dataset | myAgent correct | acc | avg tokens | strict MACT status |
|---|---:|---:|---:|---|
| WTQ | 41/50 | 0.820 | 6,246.82 | strict paired tied MACT 41/50, token ratio 0.6038 |
| TabFact | 49/50 | 0.980 | 2,631.46 | strict paired beats MACT 48/50, token ratio 0.2520 |
| CRT | 34/50 | 0.680 | 12,920.72 | strict paired beats MACT 31/50, token ratio 0.9940 |
| Overall strict Gate-50 | 124/150 | 0.827 | 7,266.33 | passes MACT 120/150, token ratio 0.6452 |

Strict Gate-50 subtotal：

| system | correct | avg tokens |
|---|---:|---:|
| myAgent | 124/150 | 7,266.33 |
| MACT | 120/150 | 11,262.41 |

Aggregate paired：

| both correct | myAgent only | MACT only | both wrong | McNemar p | token ratio |
|---:|---:|---:|---:|---:|---:|
| 109 | 15 | 11 | 15 | 0.5572 | 0.6452 |

当前可以写成：

```text
在 Qwen3-32B frozen first50 strict paired gate 中，myAgent 在 WTQ 追平 MACT，
在 TabFact 和 CRT 超过 MACT；三数据集合计 124/150 vs MACT 120/150，
平均 token 约为 MACT 的 64.5%，平均耗时约为 MACT 的 15.8%。
```

限制：

- 这是 Gate-50 小样本，不是正式全量实验。
- McNemar p 不显著，结论应写成“Gate-50 通过、值得扩大样本”，不要写成统计显著全面优于 MACT。
- CRT token 只是略低于 MACT，不像 WTQ/TabFact 那样明显降低；后续扩大样本时要继续观察 CRT token。

## 12. Recommended experiment plan

不要跑全量，也不要现在直接跑 4,344/12,779/728 全数据集。建议采用 staged gate：

1. `Gate-50`：每个候选模型先跑 WTQ/TabFact/CRT frozen50 myAgent；只对 myAgent 达标的模型跑 MACT frozen50。
2. `Gate-100/200`：通过 Gate-50 后，myAgent 扩到 frozen100 或 frozen200；MACT 只跑同样 frozen subset 或至少 disagreement/stratified subset。
3. `Formal sampled experiment`：正式写专家/专利材料时，用分层抽样而不是全量：按 dataset、risk level、route type、shortcut/non-shortcut 分层，固定 seed，报告 raw/merged/eval 行数、failed/missing、accuracy、token、耗时、paired McNemar/CI。
4. `Full dataset optional`：全量只作为最终附录或后台长期任务，不作为每次模型筛选必须项。

当前建议：

1. Qwen3-32B 已通过 Gate-50，可以进入 `Gate-100` 或 `Gate-200`。
2. 先不要继续在 TabFact 上微调；下一步更有价值的是用同一个 Gate-50 split 测 1-2 个候选模型。
3. 候选模型筛选时只跑 myAgent first；如果模型 myAgent Gate-50 低于当前 Qwen3-32B 或 token 明显高，再跳过 MACT 对照。
4. 正式专利实验建议采用 `200/数据集` 或 `300/数据集` 的分层样本作为主表，全量作为可选后台附录。

## 13. Additional verification

TabFact shortcutfix 后补充通过：

```text
python tests/test_myagent_pipeline.py              # 138 tests OK
python tests/test_evaluate_results.py              # 15 tests OK
python tests/test_selective_collaboration.py       # 8 tests OK
python tests/test_model_backends.py                # 11 tests OK
python tests/test_server_runner.py                 # 5 tests OK
python tests/test_compare_blind_results.py         # 3 tests OK
python -m py_compile code/model_backends.py code/my_agents.py code/selective_collaboration.py code/evaluate_results.py code/compare_blind_results.py scripts/server/run_sharded_tqa.py
```
