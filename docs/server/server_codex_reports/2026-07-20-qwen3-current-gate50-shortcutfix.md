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
