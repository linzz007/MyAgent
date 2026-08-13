# Current Baseline Experiment PRD

Last updated: 2026-08-13 CST

Audience: server-side Codex agent controlling `/home/ubuntu/lzz/MyAgent` and `/home/ubuntu/lzz/MACT`.

This is the latest experiment PRD. It replaces the previous broad baseline plan. The user has limited time, so the plan below distinguishes necessary experiments from optional strengthening work. Do not expand the experiment scope unless the user or supervisor explicitly asks for it.

## 0. Executive Decision

The immediate goal is not to keep optimizing Qwen3 or to reproduce every recent table-QA paper. The immediate goal is to finish a master's-thesis-level and patent-supporting experiment package with:

1. at least three baselines,
2. three datasets,
3. one main model,
4. accuracy / token / time / failure-rate reporting,
5. a small but clear ablation table for the patent mechanisms.

If the P0 items in this PRD are complete, the experiment body can be considered complete for the current thesis/patent stage. P1 items strengthen the report. P2 items are not necessary now.

## 1. Current Evidence Already Available

Qwen3-32B MyAgent-vs-MACT strict staged paired evidence is already positive:

- WTQ: MyAgent `76/100` vs MACT `74/100`.
- TabFact: MyAgent `91/100` vs MACT `87/100`.
- CRT: MyAgent `65/100` vs MACT `62/100`.
- Overall: MyAgent `232/300` vs MACT `223/300`.
- Overall token ratio: `0.5662`.

Interpretation:

- This already supports the claim that Qwen3-32B + MyAgent can exceed MACT on current staged paired samples while using fewer tokens.
- Do not continue single-dataset Qwen3 tuning unless a concrete defect blocks the final tables.
- The next necessary work is baseline packaging, not more Qwen3 score chasing.

## 2. Priority Levels

| Priority | Meaning | Default action |
|---|---|---|
| P0 | Necessary for the current thesis/patent experiment body | Must do |
| P1 | Useful strengthening if time remains | Do only after all P0 is complete |
| P2 | Optional or expensive | Do not do unless explicitly requested |

## 3. P0 Necessary Experiments

### P0.1 Main Baseline Comparison

Run the main comparison under the same conditions:

- Model: Qwen3-32B local.
- Datasets: WTQ, TabFact, CRT.
- Sample size: `200` rows per dataset.
- Sample policy: same sample IDs for every method.
- Evaluation: same evaluator and same answer-normalization policy.

Required methods:

| Method | Role | Required reason |
|---|---|---|
| MyAgent | Proposed method | Thesis/patent method |
| MACT | Strong baseline | Main paper target |
| Direct-CoT | Weak baseline | Proves the model cannot solve the task well by direct prompting alone |
| Single-Agent Pandas / I2R-style | Medium baseline | Proves the gain is not just from giving one agent a table/code tool |

This gives three baselines: MACT, Direct-CoT, and Single-Agent Pandas. That satisfies the supervisor requirement of at least three baselines.

Main result table format:

| Method | WTQ Acc | TabFact Acc | CRT Acc | Overall Acc | Avg Token | Avg Time | Fail/Missing |
|---|---:|---:|---:|---:|---:|---:|---:|

### P0.2 Efficiency Reporting

Do not run a separate efficiency experiment. Compute these metrics from P0.1:

- average token per sample,
- average time per sample,
- failure / missing-answer count,
- token ratio to MACT.

Efficiency is required in the final report because the patent method claims selective collaboration and compression, not just higher accuracy.

### P0.3 MyAgent Ablation

Run MyAgent ablations separately from baseline comparisons.

Minimum sample size:

- `50` rows per dataset if time is tight.
- `100` rows per dataset if the first ablation table is noisy.

Minimum ablation variants:

| Variant | Mechanism supported | Priority |
|---|---|---|
| No question-type routing | problem type differentiation | P0 |
| No dual scoring / risk scoring | dual scoring mechanism and formula weights | P0 |
| No strong verification / selective collaboration | high-risk collaboration and second check | P0 |
| No table compression / evidence retention | table information compression and evidence retention | P0 if switch exists; otherwise P1 with limitation documented |

The ablation table must report accuracy and token. It can use fewer samples than the main comparison because its purpose is mechanism evidence, not leaderboard-level comparison.

### P0.4 Existing Multi-Model Boundary Summary

Do not rerun smaller models now. Summarize existing no-go evidence:

- Qwen3-14B-AWQ: already tested as no-go for MyAgent Gate-50.
- Qwen2.5-14B-Instruct-AWQ: already tested as no-go for MyAgent Gate-50.
- Qwen2.5-3B-Instruct: already tested as no-go for MyAgent Gate-50.

Use this as boundary evidence: smaller/quantized models run but accuracy is not enough, so Qwen3-32B remains the main model.

## 4. P1 Strengthening Work

Do these only after all P0 items are complete.

### P1.1 Recent-Method Baseline

Add one recent table-reasoning baseline only if time remains or the supervisor explicitly wants a recent method:

- POS-style SQL baseline, preferred.
- SynTQA-style baseline, fallback if POS is blocked.

Run policy:

- First run Gate-50: WTQ / TabFact / CRT, `50` rows each.
- Only expand to `200` rows per dataset if the runner is stable and the result is useful.

Do not let POS/SynTQA block P0 completion.

### P1.2 Larger Sample

Only consider `500` rows per dataset after P0.1 Formal-200 is complete and stable.

### P1.3 Difficulty Split

If existing MyAgent risk/type labels are easy to reuse, report simple vs hard question accuracy. This is useful for thesis explanation but not mandatory for completion.

## 5. P2 Not Necessary Now

Do not do these by default:

- full official datasets with thousands of rows,
- MACT paired runs for every model,
- all baselines on all models,
- reproduction of every MACT paper baseline,
- reproduction of every recent table-QA paper,
- repeated Qwen3-14B / Qwen2.5-14B / Qwen2.5-3B runs unless a new reason is documented.

These are expensive and not required for the current graduation/patent objective.

## 6. Runner Requirements

Every baseline runner must write one JSONL row per input row and preserve enough fields for the common evaluator.

Required output fields or equivalents:

- `id`
- dataset / task name
- prediction / answer
- gold answer copied from input
- token usage if available
- elapsed seconds if available
- failure status if the sample cannot be answered

Before any 200-row run, each new baseline must pass a 5-row smoke test on WTQ, TabFact, and CRT.

## 7. Immediate Server To-Do List

### Step 1: Sync Latest Code

```bash
cd /home/ubuntu/lzz/MyAgent
git fetch origin
git checkout codex/selective-risk-collaboration
git pull --ff-only origin codex/selective-risk-collaboration
```

Read this PRD first:

```bash
sed -n '1,260p' docs/server/server_codex_reports/current-baseline-experiment-execution-plan.md
```

### Step 2: Audit Existing Runner Support

```bash
cd /home/ubuntu/lzz/MyAgent
rg -n "direct|cot|pandas|sql|baseline|ablation|evaluate_results|run_sharded" code scripts tests docs
```

Report:

- whether Direct-CoT runner already exists,
- whether Single-Agent Pandas runner already exists,
- whether ablation switches already exist,
- which scripts generate the final summary table.

### Step 3: Implement Only Missing P0 Runners

Implement only what is needed for P0:

- Direct-CoT runner if missing,
- Single-Agent Pandas runner if missing,
- ablation switches or wrapper flags if missing.

Do not change MyAgent policy logic while implementing baselines. Baseline work should be isolated.

### Step 4: Smoke Test

Run each new baseline on 5 rows per dataset and produce:

- raw JSONL,
- eval JSON,
- a short markdown summary,
- logs for failures.

Stop and report if any runner cannot produce exactly one output row per input row.

### Step 5: Formal-200 Main Experiment

Run:

- MyAgent,
- MACT,
- Direct-CoT,
- Single-Agent Pandas,

on WTQ / TabFact / CRT, `200` rows per dataset, same sample IDs.

Then generate the main table with accuracy, token, time, and failure rate.

### Step 6: Ablation

Run the P0 ablations on `50` rows per dataset first. Expand to `100` only if the result is too noisy or contradictory.

## 8. Completion Definition

The current experiment stage is complete when all of the following exist:

1. Main Formal-200 table for MyAgent, MACT, Direct-CoT, and Single-Agent Pandas on WTQ / TabFact / CRT.
2. Efficiency metrics from the same Formal-200 runs.
3. MyAgent ablation table covering at least three core mechanisms.
4. Existing multi-model no-go/boundary summary included in the final report.
5. A short final experiment summary markdown that states what was run, what was not run, and why.

If these five items are done, do not continue running optional experiments without user approval.

## 9. Stop Conditions

Stop and report instead of continuing if:

- a baseline cannot produce one output row per input row,
- evaluator compatibility is unclear,
- token accounting is unavailable and cannot be approximated consistently,
- a run would repeat a known no-go model,
- a run would expand beyond P0 before P0 is complete,
- a full-dataset run is proposed without explicit user approval.

## 10. Final Report Tables

Required final tables:

1. Main baseline comparison: MyAgent vs MACT vs Direct-CoT vs Single-Agent Pandas.
2. Efficiency comparison: accuracy, token, time, failure rate.
3. Ablation comparison: remove MyAgent mechanisms.
4. Multi-model boundary summary: existing smaller-model no-go results.

Optional final tables:

1. POS/SynTQA recent baseline.
2. Difficulty split.
3. Larger sample or full dataset.

## 11. Wording Boundaries

Allowed:

- "Under Qwen3-32B and same-ID paired staged samples, MyAgent exceeds MACT on WTQ, TabFact, and CRT with lower overall token usage."
- "The final experiment compares MyAgent with at least three baselines: MACT, Direct-CoT, and Single-Agent Pandas."
- "Existing smaller-model tests are treated as boundary evidence, not as proof that all models benefit equally."

Not allowed unless new evidence is produced:

- "All full datasets are completed."
- "All tested models beat MACT."
- "All baselines were run on all models."
- "MACT paper-reported numbers are directly comparable to our same-sample table."
- "Every recent table-QA paper has been reproduced."

## 12. Codex Preparation Checkpoint

Last preparation update: 2026-08-12 15:12 CST.

Current server condition:

- GPU is occupied, so no online model or experiment run was started in this preparation step.
- MyAgent branch is `codex/selective-risk-collaboration`.
- MyAgent preparation commit base before this update: `56a9d6344618`.
- MACT preparation commit base before this update: `60dff4a28d3d`.

Runner audit result:

| Item | Status | Evidence / action |
|---|---|---|
| MyAgent runner | existing | `scripts/server/run_sharded_tqa.py`; supports WTQ/TabFact/CRT, same evaluator, token/time/failure fields |
| MACT runner | existing | `scripts/server/run_mact_one_by_one.py`; preserves one output row per input row, including failure rows |
| Direct-CoT runner | implemented in preparation | `scripts/server/run_baseline_tqa.py --baseline direct_cot` |
| Single-Agent Pandas runner | implemented in preparation | `scripts/server/run_baseline_tqa.py --baseline single_agent_pandas` |
| Final summary table generator | implemented in preparation | `scripts/server/summarize_baseline_experiment.py` |
| Existing ablation switches | partially available | `--collaboration-mode legacy`, `--disable-strong-verification`, `--disable-deterministic-shortcuts` |
| Missing ablation switches | still pending | no explicit no-question-routing switch; no explicit no-table-compression/evidence-retention switch |

Prepared MACT run package:

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/
```

Prepared fixed inputs:

| Slice | WTQ | TabFact | CRT |
|---|---:|---:|---:|
| smoke5 | 5 rows | 5 rows | 5 rows |
| formal200 | 200 rows, `nu-0` to `nu-199` | 200 rows, `tabfact-test-0` to `tabfact-test-199` | 200 rows, `crt-0` to `crt-199` |
| ablation50 | 50 rows | 50 rows | 50 rows |

Generated scripts in the MACT run package:

| Script | Purpose |
|---|---|
| `healthcheck_services.sh` | Check configured Qwen3 endpoint(s) before any run |
| `run_smoke_direct_cot.sh` | 5-row Direct-CoT smoke for all three datasets |
| `run_smoke_single_agent_pandas.sh` | 5-row Single-Agent Pandas smoke for all three datasets |
| `run_formal_myagent.sh` | Formal-200 MyAgent run on fixed inputs |
| `run_formal_direct_cot.sh` | Formal-200 Direct-CoT run on fixed inputs |
| `run_formal_single_agent_pandas.sh` | Formal-200 Single-Agent Pandas run on fixed inputs |
| `run_mact_wtq_formal200.sh` | Formal-200 MACT WTQ run |
| `run_mact_tabfact_formal200.sh` | Formal-200 MACT TabFact run using MACT `scitab` task |
| `run_mact_crt_formal200.sh` | Formal-200 MACT CRT run |
| `run_ablation_legacy50.sh` | 50-row MyAgent legacy/no-selective-risk ablation |
| `run_ablation_no_strong50.sh` | 50-row MyAgent no-strong-verification ablation |
| `run_ablation_no_deterministic_shortcuts50.sh` | 50-row MyAgent no-deterministic-shortcut ablation |
| `run_eval_and_summary.sh` | Evaluate MACT outputs and generate `summary/main_baseline_summary.md` |
| `checkpoint_to_git.sh` | Force-add this ignored MACT output directory, commit, and push |

Validation already completed without GPU:

- `python -m py_compile scripts/server/run_baseline_tqa.py scripts/server/summarize_baseline_experiment.py scripts/server/prepare_baseline_experiment_run.py`
- `python -m unittest tests.test_run_baseline_tqa tests.test_server_runner tests.test_run_mact_one_by_one -v`
- `bash -n` passed for all generated scripts in the MACT run package.
- Input row counts verified: formal `200 * 3`, ablation `50 * 3`, smoke `5 * 3`.
- Dry-run command generation verified for both Direct-CoT and Single-Agent Pandas style runners.

Next action when GPU becomes available:

```bash
cd /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505
export LOCAL_VLLM_API_KEY=local-vllm-key-change-me
# Adjust BASELINE_ENDPOINTS if the Qwen3 service uses different ports.
bash healthcheck_services.sh
bash run_smoke_direct_cot.sh
bash run_smoke_single_agent_pandas.sh
```

Continue only if each smoke run produces exactly 5 merged rows per dataset and valid eval JSON. Then run the Formal-200 scripts and finally:

```bash
bash run_eval_and_summary.sh
bash checkpoint_to_git.sh "results: checkpoint qwen3 baseline formal200"
```

## 13. 2026-08-13 Execution Checkpoint

Current execution state:

- Qwen3-32B local service is running on two vLLM endpoints and should be kept resident unless switching models:
  - `http://127.0.0.1:8000/v1`, GPUs `4,5`
  - `http://127.0.0.1:8001/v1`, GPUs `6,7`
- GPU `0,1,2,3` services were stopped at the user's request. Do not use GPUs `0,1,2,3` for the current experiment unless the user explicitly changes this constraint. Recheck after stopping showed no compute processes on GPUs `0,1,2,3`; only the four VLLM workers on GPUs `4,5,6,7` remained visible in `nvidia-smi --query-compute-apps`.
- Served model name: `qwen3-32b-local`.
- API key env: `LOCAL_VLLM_API_KEY=local-vllm-key-change-me`.
- Main result package remains:

```text
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/
```

Git checkpoints already pushed:

| Repo | Commit | Content |
|---|---|---|
| MyAgent | `120b04e` | Stabilized `single_agent_pandas` runner with one code-repair round and tests |
| MyAgent | `1f577ae` | Fixed baseline JSON output serialization for pandas `Timedelta`/`Timestamp` scalar values |
| MyAgent | `ccb23eb` | Passed `--thinking disabled/enabled` through `run_sharded_tqa.py` to `code/tqa.py` |
| MyAgent | `244a26f` | Added `scripts/server/run_mact_sharded_one_by_one.py` to shard MACT one-by-one execution across endpoints without changing MACT reasoning parameters |
| MyAgent | `009fc8e` | Added `configs/server/qwen3_32b_2x2gpu_0123_local.env` for Qwen3 endpoints on GPUs `0,1` and `2,3` |
| MACT | `5358fc0` | Smoke outputs for Direct-CoT and Single-Agent Pandas |
| MACT | `56cf7d5` | Direct-CoT Formal-200 raw, merged, eval, logs |
| MACT | `cbffd44` | Single-Agent Pandas WTQ Formal-200 raw, merged, eval; partial TabFact checkpoint |
| MACT | `67f0075` | Single-Agent Pandas TabFact Formal-200 raw, merged, eval; partial CRT checkpoint |
| MACT | `a6f6dd0` | Complete Single-Agent Pandas Formal-200 raw, merged, eval, logs |
| MACT | `9e93a81` | MyAgent WTQ Formal-200 partial raw/log checkpoint |
| MACT | `092d853` | MyAgent WTQ Formal-200 raw, merged, eval, logs; partial TabFact checkpoint |
| MACT | `bc6cca7` | MyAgent TabFact Formal-200 partial raw/log checkpoint |
| MACT | `71c54d9` | MyAgent TabFact Formal-200 raw, merged, eval, logs; partial CRT checkpoint |
| MACT | `f32b4ba` | MyAgent CRT Formal-200 partial raw/log checkpoint |
| MACT | `fb7fa89` | Complete MyAgent Formal-200 raw, merged, eval, logs |
| MACT | `3f92f05` | MACT Formal-200 partial checkpoint: WTQ 9/200, TabFact 15/200, logs and temp sample traces |
| MACT | `e118532` | MACT Formal-200 partial checkpoint plus sharded helper scripts: WTQ 17/200, TabFact 25/200 |
| MACT | `4a8a874` | MACT Formal-200 partial checkpoint: WTQ 51/200, TabFact 50/200, CRT shard outputs 6/200 |
| MACT | `03964eb` | MACT Formal-200 partial checkpoint after stopping GPUs `0,1,2,3`: WTQ 103/200, TabFact 96/200, CRT stopped shard traces 79/200 |
| MACT | `28613d9` | MACT Formal-200 partial checkpoint on GPUs `4,5,6,7` only: WTQ 120/200, TabFact 119/200; CRT final still pending |
| MACT | `b795b33` | MACT Formal-200 partial checkpoint on GPUs `4,5,6,7` only: WTQ 140/200, TabFact 135/200; CRT final still pending |
| MACT | `baf73f0` | MACT Formal-200 partial checkpoint on GPUs `4,5,6,7` only: WTQ 150/200, TabFact 143/200; CRT final still pending |
| MACT | `4812733` | MACT Formal-200 partial checkpoint on GPUs `4,5,6,7` only: WTQ 153/200, TabFact 150/200; CRT final still pending |

Completed Formal-200 baseline:

| Method | Dataset | Merged rows | Accuracy | Avg token | Avg time | Fail/Missing |
|---|---|---:|---:|---:|---:|---:|
| Direct-CoT | WTQ | 200 | 0.630 | 852.47 | 2.286s | 1/1 |
| Direct-CoT | TabFact | 200 | 0.745 | 645.63 | 2.739s | 0/0 |
| Direct-CoT | CRT | 200 | 0.555 | 639.91 | 2.629s | 0/0 |
| Single-Agent Pandas | WTQ | 200 | 0.690 | 1185.12 | 6.126s | 6/11 |
| Single-Agent Pandas | TabFact | 200 | 0.795 | 938.19 | 7.512s | 4/4 |
| Single-Agent Pandas | CRT | 200 | 0.620 | 1099.42 | 9.166s | 12/13 |
| MyAgent | WTQ | 200 | 0.705 | 6326.39 | 16.663s | 0/0 |
| MyAgent | TabFact | 200 | 0.810 | 2796.52 | 13.400s | 0/0 |
| MyAgent | CRT | 200 | 0.665 | 10430.63 | 23.134s | 0/0 |

Last completed baseline run:

```bash
cd /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505
export LOCAL_VLLM_API_KEY=local-vllm-key-change-me
bash run_formal_single_agent_pandas.sh
```

`single_agent_pandas` completion note: CRT shard01 first hit a pandas `Timedelta` JSON serialization boundary. The runner was fixed at MyAgent `1f577ae`; `--resume` completed only the missing CRT rows. This is an output-format robustness fix, not a baseline strategy change.

Current in-flight run:

| Method | Dataset | Script | Endpoint | Current state |
|---|---|---|---|---|
| MACT | WTQ | `run_mact_wtq_formal200.sh` | `http://127.0.0.1:8000/v1`, GPUs `4,5` | running; latest synced checkpoint `4812733` has 153/200 rows |
| MACT | TabFact | `run_mact_tabfact_formal200.sh` | `http://127.0.0.1:8001/v1`, GPUs `6,7` | running; latest synced checkpoint `4812733` has 150/200 rows |
| MACT | CRT | `run_mact_crt_formal200_sharded.sh` | stopped; previously used `http://127.0.0.1:8002/v1` and `http://127.0.0.1:8003/v1` | stopped at user request; 79/200 shard rows are retained as traces only and must not be used as final Formal-200 CRT |

Operational notes for the next Codex page:

- Do not stop the two Qwen3-32B vLLM services on GPUs `4,5` and `6,7` unless switching models or the user explicitly allows releasing those GPUs.
- Do not use GPUs `0,1,2,3` for the current experiment. They were intentionally released.
- MACT is running through `scripts/server/run_mact_one_by_one.py`, which is resumable and writes one JSONL row only after each sample finishes.
- MACT can now also be run through `scripts/server/run_mact_sharded_one_by_one.py` for faster execution across multiple endpoints. This changes only experiment scheduling: each shard still calls the same one-sample MACT runner with the same MACT parameters, then merges rows back in original order.
- The MACT wrapper does not have a per-sample timeout. A temporarily unchanged output file is not enough to call the run stuck; check GPU utilization, temp sample output size, and `logs/mact_*_formal200.log`.
- If one endpoint finishes early, keep its model resident and use that endpoint for the next MACT dataset or remaining MACT work.

New helper scripts added to the MACT run package:

| Script | Use |
|---|---|
| `run_mact_crt_formal200_sharded.sh` | Run MACT CRT Formal-200 through the sharded one-by-one wrapper; for the current constraint use only 4567 endpoints by setting `BASELINE_ENDPOINTS` to `http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1` after WTQ/TabFact release them |
| `run_mact_wtq_formal200_sharded_resume.sh` | Resume MACT WTQ Formal-200 through the sharded wrapper after the original WTQ runner has stopped; do not run concurrently against the same WTQ output file |

Continue P0 from the current state:

1. Let `run_mact_wtq_formal200.sh` and `run_mact_tabfact_formal200.sh` continue on GPUs `4,5` and `6,7`, with MACT result checkpoints around 150/200 rows when practical.
2. After WTQ and/or TabFact frees a 4567 endpoint, run the final MACT CRT Formal-200 on 4567 only. The stopped 0123 CRT shard traces are diagnostic artifacts, not final comparison data.
3. Run `bash run_eval_and_summary.sh` after all three MACT Formal-200 datasets finish.
4. Run `bash checkpoint_to_git.sh "results: checkpoint qwen3 baseline formal200 summary"`.
5. Run the three prepared ablation-50 scripts and checkpoint again.
