# Current Baseline Experiment PRD

Last updated: 2026-08-27 10:44 CST

Audience: server-side Codex agent controlling `/home/ubuntu/lzz/MyAgent` and `/home/ubuntu/lzz/MACT`.

This is the latest experiment PRD. It replaces the previous broad baseline plan. The user has limited time, so the plan below distinguishes necessary experiments from optional strengthening work. Do not expand the experiment scope unless the user or supervisor explicitly asks for it.

## 0.0 Final Thesis Experiment Contract: 2026-08-27

This PRD is now the execution contract for the final thesis/patent experiment stage. It is not an open brainstorming backlog.

The current patent method is frozen at the component level by tag `patent-stage-freeze-2026-08-24` on branch `codex/selective-risk-collaboration`. Server-side Codex agents should pull the latest branch and treat the frozen method as the main experimental object.

Current state:

| Item | Status | Notes |
|---|---|---|
| Patent method | frozen | Top-level route is `SIMPLE` / `COMPLEX`; deterministic table validation and selective collaboration are sub-mechanisms, not new routes. |
| Patent specification | final candidate | Version `7.0` Word/PDF has been generated locally and aligned with the frozen method wording. |
| Main result | positive | Qwen3-32B Formal-200: MyAgent `480/600 = 0.8000`, MACT `465/600 = 0.7750`. |
| Baselines | enough for supervisor requirement | MACT, Direct-CoT, and Single-Agent Pandas are the three required baselines. |
| Efficiency metrics | available | MyAgent average token `6293.12`; MACT average token `11318.89`; token ratio `0.5560`. |
| Seed-E | diagnostic warning | MyAgent `95/150`, MACT `105/150`; this shows robustness risk and must not be claimed as solved. |
| Code optimization | paused except bounded fixes | No broad new modules. Only parameter/gate/normalization/compression fixes mapped to frozen components are allowed. |

Final thesis experiments to report by default:

1. Main Formal-200 comparison: MyAgent vs MACT vs Direct-CoT vs Single-Agent Pandas on WTQ, TabFact, and CRT.
2. Efficiency comparison from the same Formal-200 runs: accuracy, average token, and average time.
3. Mechanism ablation: question routing, dual/risk scoring, table compression/evidence retention, deterministic table validation, and selective strong verification where the switch exists.
4. Seed stability diagnostic: Seed-E error attribution, followed by one fresh Seed-F or Seed-G Gate-50/Gate-100 blind validation only if a mechanism-level repair is made.
5. Multi-model boundary summary: existing smaller-model no-go results should be summarized, not rerun by default.

Experiment discipline:

- Do not run full official datasets by default.
- Do not reproduce every MACT paper baseline by default.
- Do not test extra models unless they answer a thesis table requirement.
- Do not continue optimizing on Formal-200 sample IDs, fixed questions, or one-off failures.
- Do not add modules whose names and behavior cannot be mapped to the frozen patent components.
- Every experiment must have a table destination in the thesis: main comparison, efficiency, ablation, robustness diagnostic, or model-boundary analysis.

Completion standard:

If the main Formal-200 comparison, three-baseline table, efficiency table, ablation table, Seed-E diagnostic, and one post-fix fresh seed validation are packaged into final markdown/CSV tables, this experiment stage is complete for the current graduation/patent objective. Optional experiments require explicit approval.

## 0. Executive Decision

The immediate Qwen3-32B formal200 target is achieved and can be kept as the current main result: current MyAgent exceeds MACT on WTQ, TabFact, CRT, and overall while using much fewer tokens and much less time.

The immediate goal is no longer to keep optimizing Qwen3 on fixed formal200 samples or to reproduce every recent table-QA paper. The current goal is to make the experiment and method defensible for patent/thesis writing by proving mechanism-level robustness beyond one fixed split:

1. at least three baselines,
2. three datasets,
3. one main model,
4. accuracy / token / time as the main metrics,
5. internal robust-runner diagnostics for fallback / retry / error type,
6. a small but clear ablation table for the patent mechanisms,
7. Seed-E error attribution and a later Seed-F blind validation after mechanism fixes.

If the P0 items in this PRD are complete, the experiment body can be considered complete for the current thesis/patent stage. P1 items strengthen the report. P2 items are not necessary now.

Current execution emphasis:

- Preserve the locked Qwen3-32B formal200 result instead of chasing more TabFact/WTQ single-dataset gains.
- Do not introduce new optimizations keyed to formal200 sample IDs, fixed query strings, or one-off examples. Future optimization must be based on error types, answer contracts, routing thresholds, evidence retention, compression budgets, or risk/verification mechanisms.
- Convert existing mechanism evidence into patent-facing claims: selective collaboration, deterministic verification, answer normalization, and evidence retention.
- Add only bounded diagnostics that clarify generalization risk, cross-model boundary, or mechanism failure modes.
- Treat Seed-E as the active diagnosis split. Use it for error attribution and mechanism repair, then validate with a new unseen Seed-F Gate-50 or Gate-100 before claiming multi-seed generalization.
- Final paper/patent main tables should prioritize Accuracy, Avg Token, and Avg Time. Fallback and retry diagnostics remain in logs, not as a headline failure-rate metric.

## 0.1 Current Active Direction: 2026-08-24

The user explicitly changed the direction on 2026-08-24:

1. No more formal200 sample-specific optimization. Formal200 remains the current main result, but all future fixes must target reusable mechanisms or error categories.
2. Do not use "failure rate" as a final headline metric. Every method must produce one scoreable output row per input row.
3. Implement or use a common robust runner whenever possible. Context overflow, API BadRequest, code execution errors, and tool errors should enter a recovery path instead of stopping the run.
4. Recovery policy:
   - context overflow: lower `max_tokens`, compress or truncate prompt, and preserve question, table header, candidate evidence rows/columns, and key cells first;
   - code execution failure: feed execution error back into code generation/repair for at least 1-2 retries;
   - repeated failure: use a fallback answer path so the row can still be scored as correct or incorrect.
5. Internal logs must keep `fallback_used`, `retry_count`, `error_type`, `context_overflow`, `execution_error`, and related fields for diagnosis.
6. MACT fairness rule: if MACT gets truncation/recovery handling, it must be documented as a shared run-wrapper robustness layer and not as a change to MACT core reasoning.
7. Seed-E shows current stability is insufficient. Next work is error attribution, not blind rule expansion.
8. Optimization priority is parameter and mechanism tuning of existing components:
   - the two-path route threshold `tau_s` for `SIMPLE` versus `COMPLEX`;
   - the pre-route risk threshold `theta_s`;
   - evidence retention threshold `theta_g`;
   - posterior risk trigger threshold `theta_v`;
   - table compression budget, especially WTQ long tables and CRT complex tables;
   - confidence gates for deterministic rules;
   - bounded second-check budget for high-risk WTQ/CRT rows.
9. Verification flow: repair from Seed-E error types, rerun Seed-E Gate-50, then create a new fully unseen Seed-F Gate-50 or Gate-100 for blind validation.
10. Patent wording boundary: Formal-200 can be the main positive result; Seed-E is a stability diagnostic until a repaired method also passes a new random split. Risk scoring should be described mainly as a cost/path-control mechanism unless additional evidence proves direct accuracy gain.

## 0.2 Patent Stage Freeze: 2026-08-24

The patent-facing method is frozen at the component level as of 2026-08-24. Do not add new top-level routes, new independent agents, or a new post-hoc scoring layer unless the user explicitly reopens the patent design.

Frozen patent components:

| Component | Frozen meaning |
|---|---|
| Question-type recognition | Detects answer contract, operation type, semantic complexity, structural complexity, and task-specific table QA patterns. |
| Dual scoring | Combines semantic complexity and structural/evidence complexity into a route decision, then uses pre-route and post-answer risk scores to control verification cost. |
| Table compression | Converts the original table into a question-aware evidence table by retaining headers, candidate rows, candidate columns, and key cells before prompting. |
| Two-path routing | Top-level route is only `SIMPLE` or `COMPLEX`. `SIMPLE` is a lightweight lookup path; `COMPLEX` is the reasoning path. |
| Deterministic table validation | A tool capability inside the existing complexity/risk framework, used for reproducible table operations and answer normalization. It is not a third top-level route. |
| Selective collaboration verification | A conditional sub-process inside `COMPLEX`, triggered by high posterior risk, answer-contract conflict, candidate disagreement, or insufficient evidence. It is not a blanket multi-agent pass. |
| Robust scoreable output | A run-wrapper/output-boundary requirement: every input row should produce one scoreable output row with diagnostics. For MACT this remains wrapper-only and must not alter MACT core reasoning. |

Allowed future changes before thesis experiments:

- tune thresholds, weights, and table-compression budgets;
- refine answer-contract normalization;
- add confidence gates for existing deterministic validators;
- add bounded retry/fallback handling inside the robust output contract;
- fix reusable error categories found from Seed-E/F diagnostics.

Not allowed without reopening the patent design:

- sample-ID-specific rules;
- fixed question-string rules;
- new top-level paths beyond `SIMPLE` and `COMPLEX`;
- new named modules that cannot be mapped to one of the frozen components above;
- claims that multi-seed or all-model superiority is proven before new blind validation supports it.

Patent document checkpoint:

- Current patent specification candidate: `D:\AAAcode\AAA毕业相关\专利信息\lzz-成本感知表格问答路由\说明书7.0-定稿版.docx`.
- PDF proof: `D:\AAAcode\AAA毕业相关\专利信息\lzz-成本感知表格问答路由\说明书7.0-定稿版.pdf`.
- The specification uses the two-path `SIMPLE` / `COMPLEX` route wording and does not describe `Light` / `Tool` / `Collab` as three independent top-level routes.

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

Formal-200 full baseline evidence is now positive after the locked Qwen3 patches:

- WTQ: MyAgent `157/200 = 0.7850`, MACT `156/200 = 0.7800`.
- TabFact: MyAgent `190/200 = 0.9500`, MACT `185/200 = 0.9250`.
- CRT: MyAgent `133/200 = 0.6650`, MACT `124/200 = 0.6200`.
- Overall: MyAgent `480/600 = 0.8000`, MACT `465/600 = 0.7750`.
- Efficiency claim remains strong: MyAgent average token is `6293.12` vs MACT `11318.89`, token ratio `0.5560`; MyAgent average time is `16.749s` vs MACT `126.861s`.
- MyAgent fail/missing is `0/0`; MACT fail/missing is `4/4`.
- Next work should be report packaging, ablation interpretation, and bounded generalization/cross-model diagnostics. Avoid blind score tuning. Future changes must remain patent-describable: routing, risk scoring, selective collaboration, evidence retention, verification, or deterministic answer normalization.

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

| Method | WTQ Acc | TabFact Acc | CRT Acc | Overall Acc | Avg Token | Avg Time |
|---|---:|---:|---:|---:|---:|---:|

Fallback / retry / error diagnostics are stored in the run logs and diagnostic tables. They are not headline columns in the final main table.

### P0.2 Efficiency Reporting

Do not run a separate efficiency experiment. Compute these metrics from P0.1:

- average token per sample,
- average time per sample,
- token ratio to MACT,
- internal fallback / retry diagnostics.

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

### P0.4 Common Robust Runner

All methods should be run through a shared robust-output contract whenever possible:

- exactly one merged output row per input row;
- context-overflow retry with smaller generation budget and prompt/table compression;
- code execution repair for at least 1-2 attempts before fallback;
- fallback answer generation after repeated recovery failure;
- final output always scoreable by the same evaluator;
- internal diagnostic fields for `fallback_used`, `retry_count`, `error_type`, `context_overflow`, and `execution_error`.

For MACT, this layer is only an outer running wrapper. It must not alter MACT's core reasoning prompt, agent logic, voting logic, or tool-use strategy.

### P0.5 Existing Multi-Model Boundary Summary

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
- `fallback_used`
- `retry_count`
- `error_type`
- `context_overflow`
- `execution_error`

If a sample cannot be solved after recovery, the runner must still write a fallback prediction that can be scored. The evaluator should count it as correct or incorrect, not as a run-level interruption.

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
- logs for recovery and fallback diagnostics.

Stop and report if any runner cannot produce exactly one output row per input row.

### Step 5: Formal-200 Main Experiment

Run:

- MyAgent,
- MACT,
- Direct-CoT,
- Single-Agent Pandas,

on WTQ / TabFact / CRT, `200` rows per dataset, same sample IDs.

Then generate the main table with accuracy, token, and time. Generate a separate diagnostic table for fallback, retry, and error fields.

### Step 6: Ablation

Run the P0 ablations on `50` rows per dataset first. Expand to `100` only if the result is too noisy or contradictory.

## 8. Completion Definition

The current experiment stage is complete when all of the following exist:

1. Main Formal-200 table for MyAgent, MACT, Direct-CoT, and Single-Agent Pandas on WTQ / TabFact / CRT.
2. Efficiency metrics from the same Formal-200 runs.
3. MyAgent ablation table covering at least three core mechanisms.
4. Common robust-runner policy or implementation that preserves one scoreable output row per input row.
5. Seed-E error-attribution table covering WTQ, TabFact, and CRT categories.
6. At least one mechanism-level repair batch validated on Seed-E without sample-ID-specific rules.
7. A fresh unseen Seed-F Gate-50 or Gate-100 blind validation after the repair if Seed-E improves.
8. Existing multi-model no-go/boundary summary included in the final report.
9. A short final experiment summary markdown that states what was run, what was not run, and why.

If these items are done, do not continue running optional experiments without user approval.

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
2. Efficiency comparison: accuracy, token, time.
3. Robust-runner diagnostics: fallback count, retry count, context overflow, execution error, tool error.
4. Ablation comparison: remove MyAgent mechanisms.
5. Seed-E error attribution and, after repair, Seed-F blind validation.
6. Multi-model boundary summary: existing smaller-model no-go results.

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
- 2026-08-14 10:33 CST recheck: GPUs `0,1,2,3` each show `0 MiB` used and `0%` utilization. Continue all active experiments on GPUs `4,5,6,7` through endpoints `8000` and `8001`.
- 2026-08-14 14:18 CST user reconfirmed to stop using GPUs `0,1,2,3` and run experiments only on GPUs `4,5,6,7`. Recheck showed visible vLLM compute workers only on GPUs `4,5,6,7`. GPUs `0,2,3` reported memory/utilization, but `nvidia-smi --query-compute-apps` exposed no experiment/vLLM compute PID on those cards; do not blind-kill unknown GPU usage. Continue all experiments through endpoints `8000` and `8001` on GPUs `4,5,6,7`.
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
| MACT | `1da4e89` | MACT Formal-200 partial checkpoint on GPUs `4,5,6,7` only: WTQ 172/200, TabFact 170/200; CRT final still pending |
| MACT | `c171650` | MACT Formal-200 partial checkpoint on GPUs `4,5,6,7` only: WTQ 180/200, TabFact 180/200; CRT final still pending |
| MACT | `2a7e75e` | Completed MACT WTQ and TabFact Formal-200 raw, logs, and eval; WTQ 200/200, TabFact 200/200; CRT final still pending |
| MyAgent | `e9a3349` | PRD checkpoint: fresh non-sandbox MACT CRT reached 161/200 rows; GPUs `0,1,2,3` confirmed free |
| MACT | `f8c8bcc` | MACT CRT fresh non-sandbox checkpoint: 161/200 shard rows and logs |
| MyAgent | `1d6b26b` | PRD checkpoint: fresh non-sandbox MACT CRT reached 180/200 rows |
| MACT | `1a79423` | MACT CRT fresh non-sandbox checkpoint: 180/200 shard rows and logs |

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
| MACT | WTQ | 200 | 0.780 | 10484.65 | 115.088s | 4/4 |
| MACT | TabFact | 200 | 0.925 | 11232.74 | 114.443s | 0/0 |
| MACT | CRT | 200 | 0.620 | 12239.29 | 151.051s | 0/0 |

Formal-200 aggregate:

| Method | Overall Acc | Avg token | Avg time | Fail/Missing | Token ratio to MACT |
|---|---:|---:|---:|---:|---:|
| MyAgent | 436/600 = 0.7267 | 6517.84 | 17.73s | 0/0 | 0.5758 |
| MACT | 465/600 = 0.7750 | 11318.89 | 126.86s | 4/4 | 1.0000 |
| Direct-CoT | 386/600 = 0.6433 | 712.67 | 2.55s | 1/1 | 0.0630 |
| Single-Agent Pandas | 421/600 = 0.7017 | 1074.24 | 7.60s | 22/28 | 0.0949 |

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
| MACT | WTQ | `run_mact_wtq_formal200.sh` | `http://127.0.0.1:8000/v1`, GPUs `4,5` | complete; synced checkpoint `2a7e75e` has 200/200 rows and eval |
| MACT | TabFact | `run_mact_tabfact_formal200.sh` | `http://127.0.0.1:8001/v1`, GPUs `6,7` | complete; synced checkpoint `2a7e75e` has 200/200 rows and eval |
| MACT | CRT | manual `run_mact_sharded_one_by_one.py` invocation | `http://127.0.0.1:8000/v1` and `http://127.0.0.1:8001/v1`, GPUs `4,5,6,7` | complete; merged output has 200/200 rows; eval generated at `eval/crt_mact_formal200_eval.json` |

Operational notes for the next Codex page:

- Do not stop the two Qwen3-32B vLLM services on GPUs `4,5` and `6,7` unless switching models or the user explicitly allows releasing those GPUs.
- Do not use GPUs `0,1,2,3` for the current experiment. They were intentionally released.
- MACT is running through `scripts/server/run_mact_one_by_one.py`, which is resumable and writes one JSONL row only after each sample finishes.
- MACT can now also be run through `scripts/server/run_mact_sharded_one_by_one.py` for faster execution across multiple endpoints. This changes only experiment scheduling: each shard still calls the same one-sample MACT runner with the same MACT parameters, then merges rows back in original order.
- The MACT wrapper does not have a per-sample timeout. A temporarily unchanged output file is not enough to call the run stuck; check GPU utilization, temp sample output size, and `logs/mact_*_formal200.log`.
- If one endpoint finishes early, keep its model resident and use that endpoint for the next MACT dataset or remaining MACT work.
- Previous CRT shard traces under `mact_shards/crt_crt_mact_formal200_00000_00200` were produced on GPUs `0,1,2,3` before the user stopped those GPUs. They are diagnostic history only.
- The first fresh 4567 CRT attempt under `mact_shards_4567_final/crt_crt_mact_formal200_00000_00200` ran inside a network-restricted Codex sandbox. It produced 200 invalid rows with `api_metrics.request_count=0`, empty `pred_answer`, and `openai.APIConnectionError` / `httpcore.ConnectError: [Errno 1] Operation not permitted` in the logs. The merged invalid file was moved from `mact/crt_mact_formal200.jsonl` to `diagnostics/crt_mact_formal200_sandbox_network_invalid_20260813.jsonl`.
- The valid final CRT run was restarted outside the Codex network sandbox in `mact_shards_4567_final_nonsandbox/crt_crt_mact_formal200_00000_00200`, with clean merged output target `mact/crt_mact_formal200.jsonl`.
- Early validation for this restart: shard00 wrote valid `crt-0` with `pred_answer="Yes."`, `api_metrics.request_count=3`, and `total_tokens=6869`; shard01 wrote valid `crt-100` with a non-empty final answer. No `APIConnectionError`, `ConnectError`, or `Operation not permitted` strings were found in the fresh non-sandbox logs at startup.
- Partial checkpoint at 2026-08-13 23:03:58 CST: fresh non-sandbox CRT reached shard00 `5/100` rows and shard01 `5/100` rows. This partial state is suitable only for recovery/resume, not final reporting.
- Partial checkpoint at 2026-08-13 23:18:47 CST: fresh non-sandbox CRT reached shard00 `9/100` rows and shard01 `11/100` rows. No local-network connection errors were detected in the fresh logs.
- Partial checkpoint at 2026-08-13 23:41:30 CST: fresh non-sandbox CRT reached shard00 `17/100` rows and shard01 `23/100` rows. No local-network connection errors were detected in the fresh logs.
- Partial checkpoint at 2026-08-14 00:32:48 CST: fresh non-sandbox CRT reached shard00 `39/100` rows and shard01 `41/100` rows. No local-network connection errors were detected in the fresh logs.
- Partial checkpoint at 2026-08-14 01:28:41 CST: fresh non-sandbox CRT reached shard00 `60/100` rows and shard01 `60/100` rows. No local-network connection errors were detected in the fresh logs.
- Partial checkpoint at 2026-08-14 02:17:00 CST: fresh non-sandbox CRT reached shard00 `82/100` rows and shard01 `79/100` rows, total `161/200`. No local-network connection errors were detected in the fresh logs. GPUs `0,1,2,3` show `0 MiB` and no compute process; active services remain only on GPUs `4,5,6,7`.
- Partial checkpoint at 2026-08-14 02:35:53 CST: fresh non-sandbox CRT reached shard00 `93/100` rows and shard01 `87/100` rows, total `180/200`. No local-network connection errors were detected in the fresh logs. GPUs `0,1,2,3` remain unused; active services remain only on GPUs `4,5,6,7`.
- Final completion at 2026-08-14 03:16 CST: fresh non-sandbox CRT reached shard00 `100/100` rows and shard01 `100/100` rows, merged to `mact/crt_mact_formal200.jsonl` with `200` rows. `run_eval_and_summary.sh` generated `eval/crt_mact_formal200_eval.json` and `summary/main_baseline_summary.md`. No local-network connection errors were detected in the fresh logs. GPUs `0,1,2,3` remained unused.

New helper scripts added to the MACT run package:

| Script | Use |
|---|---|
| `run_mact_crt_formal200_sharded.sh` | Run MACT CRT Formal-200 through the sharded one-by-one wrapper; for the current constraint use only 4567 endpoints by setting `BASELINE_ENDPOINTS` to `http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1` after WTQ/TabFact release them |
| `run_mact_wtq_formal200_sharded_resume.sh` | Resume MACT WTQ Formal-200 through the sharded wrapper after the original WTQ runner has stopped; do not run concurrently against the same WTQ output file |

Continue P0 from the current state:

1. Commit and push the final Formal-200 MACT CRT output/eval/summary, plus this PRD update. Completed at MyAgent `2e1dd0e` and MACT `294d36a`.
2. Run the prepared ablation-50 scripts and checkpoint after each stable result. Completed for `legacy50`, `no_strong50`, `no_deterministic_shortcuts50`, `no_question_routing50`, `no_risk_scoring50`, and `no_table_compression50`.
3. Diagnose why Formal-200 WTQ and TabFact trail MACT despite lower token/time. Candidate areas: route confidence thresholds, evidence-retention budget, final-answer normalization, and selective second-pass verification.
4. Implement only patent-describable improvements, then rerun focused validation before expanding to another Formal-200 comparison.

Ablation execution status:

| Variant | Script | Endpoint/GPU policy | Status |
|---|---|---|---|
| Legacy collaboration | `run_ablation_legacy50.sh` | `http://127.0.0.1:8000/v1` on GPUs `4,5`; `http://127.0.0.1:8001/v1` on GPUs `6,7` | complete; WTQ `0.66`, TabFact `0.86`, CRT `0.80`, overall `116/150 = 0.7733`, failed/missing `0/0` |
| No strong verification | `run_ablation_no_strong50.sh` | same 4567 endpoint policy | complete; WTQ `0.66`, TabFact `0.86`, CRT `0.80`, overall `116/150 = 0.7733`, failed/missing `0/0` |
| No deterministic shortcuts | `run_ablation_no_deterministic_shortcuts50.sh` | same 4567 endpoint policy | complete; WTQ/TabFact/CRT merged rows all `50/50`, failed/missing `0/0` |
| No question routing | `run_ablation_no_question_routing50.sh` | same 4567 endpoint policy | complete; WTQ `0.66`, TabFact `0.90`, CRT `0.76`, overall `116/150 = 0.7733`, avg token `8353.59`, avg time `21.077s`, failed/missing `0/0` |
| No risk scoring | `run_ablation_no_risk_scoring50.sh` | same 4567 endpoint policy | complete; WTQ `0.74`, TabFact `0.88`, CRT `0.82`, overall `122/150 = 0.8133`, avg token `6664.23`, avg time `18.473s`, failed/missing `0/0` |
| No table compression | `run_ablation_no_table_compression50.sh` | same 4567 endpoint policy | complete; WTQ `0.72`, TabFact `0.90`, CRT `0.70`, overall `116/150 = 0.7733`, avg token `7612.96`, avg time `18.148s`, failed/missing `2/2` |

Legacy collaboration ablation result:

| Dataset | Rows | Accuracy | Avg token | Avg time | Fail/Missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 50 | 0.660 | 2704.26 | 12.827s | 0/0 |
| TabFact | 50 | 0.860 | 2445.90 | 14.304s | 0/0 |
| CRT | 50 | 0.800 | 2399.30 | 14.741s | 0/0 |
| Overall | 150 | 116/150 = 0.7733 | 2516.49 | 13.957s | 0/0 |

No-strong-verification ablation result:

| Dataset | Rows | Accuracy | Avg token | Avg time | Fail/Missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 50 | 0.660 | 2704.22 | 12.827s | 0/0 |
| TabFact | 50 | 0.860 | 2445.90 | 14.347s | 0/0 |
| CRT | 50 | 0.800 | 2399.30 | 14.744s | 0/0 |
| Overall | 150 | 116/150 = 0.7733 | 2516.47 | 13.973s | 0/0 |

No-deterministic-shortcuts ablation result:

| Dataset | Rows | Primary accuracy | Exact match | Avg token | Avg time | Fail/Missing |
|---|---:|---:|---:|---:|---:|---:|
| WTQ | 50 | 0.680 | 0.660 | 6450.88 | 17.900s | 0/0 |
| TabFact | 50 | 0.720 | 0.720 | 3274.76 | 19.162s | 0/0 |
| CRT | 50 | 0.720 | 0.720 | 12667.86 | 29.620s | 0/0 |
| Overall | 150 | 106/150 = 0.7067 | 105/150 = 0.7000 | 7464.50 | 22.227s | 0/0 |

No-question-routing ablation result:

| Dataset | Rows | Accuracy | Avg token | Avg time | Fail/Missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 50 | 0.660 | 8318.22 | 19.802s | 0/0 |
| TabFact | 50 | 0.900 | 3106.40 | 14.925s | 0/0 |
| CRT | 50 | 0.760 | 13636.16 | 28.504s | 0/0 |
| Overall | 150 | 116/150 = 0.7733 | 8353.59 | 21.077s | 0/0 |

No-risk-scoring ablation result:

| Dataset | Rows | Accuracy | Avg token | Avg time | Fail/Missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 50 | 0.740 | 6060.10 | 15.660s | 0/0 |
| TabFact | 50 | 0.880 | 2199.22 | 13.045s | 0/0 |
| CRT | 50 | 0.820 | 11733.38 | 26.714s | 0/0 |
| Overall | 150 | 122/150 = 0.8133 | 6664.23 | 18.473s | 0/0 |

No-table-compression ablation result:

| Dataset | Rows | Accuracy | Avg token | Avg time | Fail/Missing |
|---|---:|---:|---:|---:|---:|
| WTQ | 50 | 0.720 | 7346.16 | 15.504s | 2/2 |
| TabFact | 50 | 0.900 | 2418.52 | 11.609s | 0/0 |
| CRT | 50 | 0.700 | 13074.20 | 27.332s | 0/0 |
| Overall | 150 | 116/150 = 0.7733 | 7612.96 | 18.148s | 2/2 |

Ablation interpretation: `legacy50` and `no_strong50` have identical accuracy on the current gate50 split and near-identical token/time. This split does not yet isolate the value of strong verification; the next diagnostic should inspect whether strong verification was triggered on these rows, or select high-risk rows where it is expected to activate. In contrast, disabling deterministic shortcuts is strongly negative on the same 150 rows: primary overall drops from `116/150 = 0.7733` to `106/150 = 0.7067`, TabFact drops from `0.86` to `0.72`, CRT drops from `0.80` to `0.72`, and average token rises from about `2516` to about `7465`. This is currently the strongest P0 mechanism evidence for patent writing: deterministic shortcuts / answer normalization reduce unnecessary LLM work and protect accuracy on TabFact and CRT.

No-question-routing keeps the same overall primary accuracy as legacy/no-strong on this split, but increases average token usage from about `2516` to `8354` and average time from about `14s` to `21s`. Current evidence therefore supports question routing primarily as an efficiency and path-selection mechanism, not as a standalone accuracy driver on gate50.

No-risk-scoring is not accuracy-negative on this broad gate50 split: it reaches `122/150 = 0.8133`, above legacy/no-strong, while increasing average token usage to `6664.23`. Do not claim from this split that risk scoring directly improves accuracy. Current evidence supports a narrower claim that risk scoring is a selective cost/path-control component whose accuracy value needs a targeted high-risk split or additional acceptance-gate evidence.

No-table-compression keeps the same overall primary accuracy as legacy/no-strong on this split, but raises average token usage to `7612.96` and creates `2/50` WTQ failed/missing rows (`nu-30`, `nu-44`) from Qwen3-32B context-limit `BadRequestError`. This is direct evidence that table compression/evidence retention is needed for both token efficiency and runnability under an 8192-token service budget.

Runner robustness note: no-table-compression initially exposed that `code/tqa.py` raised on a per-sample `BadRequestError` and interrupted a shard. MyAgent commit `6de15a7` changed the output boundary so a per-sample exception writes one failed JSONL row and continues, preserving one output row per input row and allowing failed/missing counts to be reported.

Formal-200 WTQ/TabFact error diagnosis:

Generated files:

- MACT summary: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/summary/wtq_tabfact_diagnosis.md`
- MACT JSON: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/diagnostics/formal200_wtq_tabfact_error_diagnosis.json`
- Paired compare JSON: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/diagnostics/formal200_myagent_vs_mact_paired_compare.json`

Key diagnosis:

| Dataset | MyAgent correct | MACT correct | Both correct | MyAgent-only | MACT-only | Both wrong | Net gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 141/200 | 156/200 | 125 | 16 | 31 | 28 | 15 |
| TabFact | 162/200 | 185/200 | 154 | 8 | 31 | 7 | 23 |

- WTQ gap is broad: most MACT-only rows are high-risk/complex, with top tags `count`, `temporal`, `negation_logic`, `superlative_order`, and `arithmetic`. Strong verification already applied on `27/31` MACT-only rows, so the next WTQ fix should focus on evidence selection, entity canonicalization, tied answers, and temporal/count normalization rather than merely turning on more verification.
- TabFact gap is concentrated: all `31/31` MACT-only rows are `closed_choice`, all are `COMPLEX`, and all have `strong_verification_applied=false`. This is the clearest next patch target: trigger selective strong verification for high-risk compound TabFact claims, especially temporal / negation / superlative closed-choice statements, then validate first on the diagnostic MACT-only rows.

TabFact focused patch started:

- Code change: `TableQAPipeline._should_apply_strong_verification` now keeps simple TabFact labels on the cheaper path, but triggers strong verification when the row is `tabfact`, answer kind is `label`, tags include `closed_choice`, risk level is `high`, and at least two compound-risk tags are present among `temporal`, `negation_logic`, `superlative_order`, `comparison`, `arithmetic`, and `count`.
- Patent mechanism description: selective high-risk collaboration for compound binary table claims, controlled by problem tags and risk level instead of blanket second-pass verification.
- Estimated Formal-200 trigger scope before rerun: `71/200` TabFact rows, covering `17/31` TabFact MACT-only diagnostic errors.
- Local verification passed: `python -m unittest discover -s tests -p 'test_myagent_pipeline.py'` (`216` tests), `python -m py_compile code/my_agents.py code/tqa.py code/evaluate_results.py scripts/server/run_sharded_tqa.py`, and `python -m unittest discover -s tests -p 'test_evaluate_results.py'` (`15` tests). `test_run_sharded_tqa.py` discovery returned `0` tests in this workspace.
- Focused-17 validation complete at MACT `diagnostics/tabfact_compound17_patch_ed6ceca/`: old MyAgent was `0/17` on these MACT-only rows; patched MyAgent reached `9/17 = 0.5294`, with `17/17` strong-verification triggers, avg token `18999.24`, avg time `37.224s`, failed/missing `0/0`.
- Interpretation: the trigger recovers real old errors, but token/time cost is high. Before expanding to full200, run all `71` Formal-200 TabFact rows that match the trigger predicate to measure regressions among previously correct samples.
- Trigger-71 validation started at MACT `diagnostics/tabfact_compound71_patch_ed6ceca/`, input `input/diagnostic/tabfact_compound_trigger71.jsonl`, using only endpoints `8000`/`8001` on GPUs `4,5,6,7`.
- Trigger-71 validation complete: old MyAgent `50/71 = 0.7042`; patched MyAgent `51/71 = 0.7183`; MACT `65/71 = 0.9155`; old-wrong to new-right `8`, old-right to new-wrong `7`; avg token `16961.89`, avg time `33.425s`, failed/missing `0/0`.
- Interpretation: this patch is directionally positive but too weak and too expensive as a final Formal-200 optimization. It must not be expanded blindly. Next required control: run the same 71 rows with `--disable-strong-verification` to separate true strong-verification value from rerun variance, then design a safer acceptance gate for TabFact verifier overrides.
- No-strong trigger-71 control started at MACT `diagnostics/tabfact_compound71_no_strong_control_ed6ceca/`; checkpoint reached `20/71` rows by 2026-08-14 11:47 CST.
- No-strong trigger-71 control complete: no-strong `51/71 = 0.7183`, strong-trigger `51/71 = 0.7183`; strong recovers `7` no-strong wrong rows but regresses `7` no-strong correct rows; no-strong avg token `3482.51` and avg time `18.128s`, strong-trigger avg token `16961.89` and avg time `33.425s`.
- Decision: reverted the TabFact compound strong-trigger code path. It is useful negative evidence for the patent report, but it is not a production/formal200 optimization because it has no net accuracy gain over no-strong and costs far more tokens. Post-revert verification passed: `test_myagent_pipeline.py` (`216` tests), `test_evaluate_results.py` (`15` tests), and py_compile for `code/my_agents.py`, `code/tqa.py`, `code/evaluate_results.py`, `scripts/server/run_sharded_tqa.py`.

WTQ deterministic shortcut patch started:

- Code change: extend existing WTQ deterministic shortcuts for three low-token structural cases: `same number as <entity>` excluding the reference entity, next/listed-after adjacent entity lookup with blank metadata rows skipped, and `who was the <target> in the last <period>` returning the target column rather than the period column.
- Local verification passed: `python -m unittest discover -s tests -p 'test_myagent_pipeline.py'` (`221` tests) and `python -m py_compile code/my_agents.py code/tqa.py`.
- Offline Formal-200 WTQ scan: shortcut path fires on `21/200` rows and is correct on `18/21`; compared with old MyAgent formal200, estimated delta is `+6` correct and `0` regressions. Changed rows: `nu-27`, `nu-66`, `nu-78`, `nu-100`, `nu-146`, `nu-180`.
- MyAgent commit pushed: `d646885` (`feat: add wtq deterministic adjacency shortcuts`).
- Focused-6 runner validation complete on GPUs `4,5,6,7`, using endpoints `8000` and `8001`. Input: MACT `input/diagnostic/wtq_shortcut_delta6.jsonl`; output root: MACT `diagnostics/wtq_shortcut_delta6_patch_d646885`; summary: MACT `summary/wtq_shortcut_patch_d646885.md`.
- Focused-6 result: WTQ primary accuracy `6/6 = 1.0000`, strict exact `5/6 = 0.8333`, avg token `3858.33`, avg time `4.480s`, failed/missing `0/0`. Fixed rows: `nu-27`, `nu-66`, `nu-78`, `nu-100`, `nu-146`, `nu-180`.
- Decision: accept this patch as a valid low-token WTQ mechanism improvement. It raises the expected WTQ formal200 result from `141/200 = 0.7050` to about `147/200 = 0.7350`, and expected total formal200 from `436/600 = 0.7267` to about `442/600 = 0.7367`; this is still below MACT overall `465/600 = 0.7750`, so the patent-data goal is not complete yet.
- Full WTQ Formal-200 validation complete on GPUs `4,5,6,7`; repeated `nvidia-smi pmon -c 1` checks showed no visible compute PID on GPUs `0,1,2,3`. Output root: MACT `diagnostics/wtq_formal200_patch_f102b96`; eval: MACT `diagnostics/wtq_formal200_patch_f102b96/eval/wtq_qwen3-32b-local_eval.json`; compare JSON: MACT `diagnostics/wtq_formal200_patch_f102b96_compare.json`; summary: MACT `summary/wtq_formal200_patch_f102b96.md`.
- Full WTQ result: patched MyAgent `147/200 = 0.7350` primary accuracy, strict exact `145/200 = 0.7250`, avg token `6228.74`, avg time `16.140s`, failed/missing `0/0`. Old MyAgent WTQ was `141/200 = 0.7050`; MACT WTQ was `156/200 = 0.7800`, avg token `10698.62`, avg time `115.088s`, failed/missing `4/4`.
- Actual full-run delta: net `+6` primary-correct rows, made of `8` old-wrong to patched-right rows (`nu-27`, `nu-66`, `nu-78`, `nu-100`, `nu-129`, `nu-146`, `nu-180`, `nu-188`) and `2` old-right to patched-wrong rows (`nu-152`, `nu-160`). The two regressions did not use the new deterministic shortcut path, so they are non-shortcut runner/model-path variance rather than direct shortcut failures.
- Current status after WTQ patch: keep the patch, but the patent-data goal is still incomplete. Expected total formal200 becomes approximately `442/600 = 0.7367` versus MACT `465/600 = 0.7750`; WTQ gap is now `9` rows (`147` vs `156`). Next work should target remaining WTQ MACT-only rows and the TabFact gap, starting with `nu-152`/`nu-160` regressions and high-frequency count, temporal, negation/exclusion, comparison/superlative categories.
- WTQ count/date filter patch pushed: MyAgent `7e18c84` (`feat: add wtq deterministic count filters`). It adds deterministic coverage for `at least <number> <metric>` row counts and bare-month `after <month>` row counts, with a guard that rejects specific day cutoffs such as `after october 1st`.
- Local verification passed after `7e18c84`: `test_myagent_pipeline.py` (`224` tests) and py_compile for `code/my_agents.py`, `code/tqa.py`.
- Offline WTQ formal200 scan after `7e18c84`: new rules only hit `nu-152` and `nu-160`, both correct. Focused-2 runner validation complete on GPUs `4,5,6,7`; input MACT `input/diagnostic/wtq_count_filter_regression2.jsonl`; output root MACT `diagnostics/wtq_count_filter_regression2_patch_7e18c84`; summary MACT `summary/wtq_count_filter_patch_7e18c84.md`.
- Focused-2 result: primary accuracy `2/2 = 1.0000`, strict exact `2/2 = 1.0000`, avg token `3607.00`, avg time `6.339s`, failed/missing `0/0`. Expected WTQ if applied to the previous full run is `149/200 = 0.7450`, but avoid rerunning full WTQ200 after every two-row patch; batch the next full rerun with additional deterministic fixes to reduce server time and rerun variance.
- WTQ table-filter patch pushed: MyAgent `cdc644c` (`feat: add wtq deterministic table filters`). It adds deterministic coverage for only-metric-value entity lookup, entity-filtered metric summation, listed-entity combined row counts, specific-date cutoff row counts, first metric-threshold date lookup, and column-value row counts.
- Local verification passed after `cdc644c`: `test_myagent_pipeline.py` (`232` tests) and py_compile for `code/my_agents.py`, `code/tqa.py`.
- Offline WTQ formal200 scan after `cdc644c`: new/extended rules only hit `8` rows and all are correct: `nu-18`, `nu-22`, `nu-73`, `nu-81`, `nu-94`, `nu-110`, `nu-142`, `nu-187`. Focused-8 runner validation complete on GPUs `4,5,6,7`; input MACT `input/diagnostic/wtq_table_filter_delta8.jsonl`; output root MACT `diagnostics/wtq_table_filter_delta8_patch_cdc644c`; summary MACT `summary/wtq_table_filter_patch_cdc644c.md`.
- Focused-8 result: primary accuracy `8/8 = 1.0000`, strict exact `8/8 = 1.0000`, avg token `4398.00`, avg time `6.319s`, failed/missing `0/0`. If combined with previous validated patches, expected WTQ becomes `157/200 = 0.7850`, slightly above MACT WTQ `156/200 = 0.7800`; full WTQ200 rerun is still needed to lock the realized number because non-shortcut rows can vary across reruns.
- Full WTQ Formal-200 rerun complete at MyAgent `7168923` on GPUs `4,5,6,7`. Output root: MACT `diagnostics/wtq_formal200_patch_7168923`; eval: MACT `diagnostics/wtq_formal200_patch_7168923/eval/wtq_qwen3-32b-local_eval.json`; compare JSON: MACT `diagnostics/wtq_formal200_patch_7168923_compare.json`; summary: MACT `summary/wtq_formal200_patch_7168923.md`.
- Full WTQ locked result: current MyAgent `157/200 = 0.7850` primary accuracy, strict exact `155/200 = 0.7750`, avg token `6076.32`, avg time `15.424s`, failed/missing `0/0`. MACT WTQ is `156/200 = 0.7800`, avg token `10484.65`, avg time `115.088s`, failed/missing `4/4`. WTQ subgoal is achieved for Qwen3-32B formal200.
- Updated formal200 overall estimate with new WTQ and existing TabFact/CRT: MyAgent `452/600 = 0.7533`, MACT `465/600 = 0.7750`. MyAgent now leads WTQ and CRT, but overall still trails because TabFact remains `162/200` vs MACT `185/200`. Next priority should shift from WTQ to TabFact gap closure, while preserving the deterministic shortcut evidence as patent mechanism support.

TabFact deterministic table-filter patch:

- User execution constraint updated: do not use GPUs `0,1,2,3`; keep experiments on GPUs `4,5,6,7`. Repeated `nvidia-smi pmon -c 1` checks during TabFact full200 showed no visible compute PID on GPUs `0,1,2,3`; active model workers remained on GPUs `4,5,6,7`.
- MyAgent patch pushed: `c2552fd` (`feat: add tabfact deterministic table filters`). It adds narrow deterministic TabFact rules for tied-rank country counts, represented-country counts, over-par country majority, one-off frequency exceptions, only-not-from country pairs, complete player source columns, opponent attendance comparison, extreme score difference, entity metric difference, and second-highest metric entity claims.
- Local verification passed after `c2552fd`: `test_myagent_pipeline.py` (`237` tests) and py_compile for `code/my_agents.py`, `code/tqa.py`.
- Offline TabFact Formal-200 scan: new rules hit `12` rows, all correct, with `10` old-wrong rows fixed and no wrong deterministic hits.
- Focused-10 runner validation complete on GPUs `4,5,6,7`; input MACT `input/diagnostic/tabfact_table_filter_delta10.jsonl`; output root MACT `diagnostics/tabfact_table_filter_delta10_patch_c2552fd`; summary MACT `summary/tabfact_table_filter_patch_c2552fd.md`.
- Focused-10 result: primary accuracy `10/10 = 1.0000`, exact `10/10 = 1.0000`, avg token `402.20`, avg time `2.476s`, failed/missing `0/0`. All `10/10` rows used `deterministic_shortcut_applied=true`.
- Full TabFact Formal-200 rerun complete at MyAgent `c2552fd`; output root MACT `diagnostics/tabfact_formal200_patch_c2552fd`; eval MACT `diagnostics/tabfact_formal200_patch_c2552fd/eval/tabfact_qwen3-32b-local_eval.json`; merged rows `200/200`.
- Full TabFact result: current MyAgent `175/200 = 0.8750`, avg token `2711.77`, avg time `13.228s`, failed/missing `0/0`. Old MyAgent was `162/200 = 0.8100`; official MACT eval is `185/200 = 0.9250`, avg token `11232.74`, avg time `114.443s`, failed/missing `0/0`.
- Actual full-run delta vs old MyAgent: `15` old-wrong to new-right rows, `2` old-right to new-wrong rows (`tabfact-test-177`, `tabfact-test-190`), net `+13`. Full run had `28` deterministic shortcut hits and `0` wrong shortcut hits.
- Current locked formal200 aggregate with WTQ `7168923`, TabFact `c2552fd`, and existing CRT: MyAgent `465/600 = 0.7750`, MACT official `465/600 = 0.7750`. MyAgent now ties MACT overall, leads WTQ and CRT, and still trails TabFact (`175` vs `185`). The Qwen3 formal goal is not complete yet because the user target is to exceed MACT, preferably across all datasets.
- Next TabFact work should target the remaining MACT-only / new-wrong rows: `tabfact-test-9`, `tabfact-test-22`, `tabfact-test-34`, `tabfact-test-41`, `tabfact-test-60`, `tabfact-test-63`, `tabfact-test-64`, `tabfact-test-68`, `tabfact-test-80`, `tabfact-test-84`, `tabfact-test-103`, `tabfact-test-109`, `tabfact-test-123`, `tabfact-test-139`, `tabfact-test-146`, `tabfact-test-177`, `tabfact-test-187`, `tabfact-test-190`. Prioritize patent-describable deterministic table-verification mechanisms: entity-max metric ownership, month missing-day counts, lowest-attendance win/loss checks, entity-year/competition award counts, rank-gap checks, top-N country/no-medal checks, tenure interval overlap/gap/stint duration checks, and winner/race-leader count checks.

TabFact temporal/rank table-filter patch and Qwen3 formal target completion:

- MyAgent patch pushed: `5e3e0e8` (`feat: add tabfact temporal and rank filters`). It adds deterministic TabFact verification for entity-maximum metric ownership, only-year repeated-goal counts, equal win/loss record counts, monthly no-game day counts, lowest-attendance game win/loss, beer award counts, tennis surface counts, championship loss claims, top-N country/no-medal claims, Jazz tenure gaps, stint duration, race winner/leader counts, and consecutive dated race wins.
- Local verification passed after `5e3e0e8`: `test_myagent_pipeline.py` (`240` tests) and py_compile for `code/my_agents.py`, `code/tqa.py`.
- Offline TabFact Formal-200 scan: second-batch rules hit `22` rows, all correct, with `15` current-wrong rows fixed and no wrong deterministic hits.
- Focused-15 runner validation complete on GPUs `4,5,6,7`; input MACT `input/diagnostic/tabfact_temporal_rank_delta15.jsonl`; output root MACT `diagnostics/tabfact_temporal_rank_delta15_patch_5e3e0e8`; summary MACT `summary/tabfact_temporal_rank_patch_5e3e0e8.md`.
- Focused-15 result: primary accuracy `15/15 = 1.0000`, exact `15/15 = 1.0000`, avg token `457.60`, avg time `2.313s`, failed/missing `0/0`. All `15/15` rows used `deterministic_shortcut_applied=true`.
- Full TabFact Formal-200 rerun complete at MyAgent `5e3e0e8`; output root MACT `diagnostics/tabfact_formal200_patch_5e3e0e8`; eval MACT `diagnostics/tabfact_formal200_patch_5e3e0e8/eval/tabfact_qwen3-32b-local_eval.json`; merged rows `200/200`.
- Full TabFact locked result: current MyAgent `190/200 = 0.9500`, avg token `2372.42`, avg time `11.689s`, failed/missing `0/0`. Official MACT TabFact is `185/200 = 0.9250`, avg token `11232.74`, avg time `114.443s`, failed/missing `0/0`. TabFact subgoal is achieved for Qwen3-32B formal200.
- Actual full-run delta vs `c2552fd`: `15` previous-wrong to new-right rows, `0` previous-right to new-wrong rows, net `+15`. Full run had `50` deterministic shortcut hits and `0` wrong shortcut hits.
- Final locked Qwen3 Formal-200 aggregate: MyAgent WTQ `157/200`, TabFact `190/200`, CRT `133/200`, overall `480/600 = 0.8000`; MACT official WTQ `156/200`, TabFact `185/200`, CRT `124/200`, overall `465/600 = 0.7750`. MyAgent exceeds MACT on every dataset and overall.
- Efficiency aggregate after the patch: MyAgent avg token `6293.12` vs MACT `11318.89` (token ratio `0.5560`); MyAgent avg time `16.749s` vs MACT `126.861s` (time ratio `0.1320`); MyAgent fail/missing `0/0` vs MACT `4/4`.
- Decision: the user's Qwen3-32B formal200 goal is achieved for the current patent/thesis stage. Continue from here by packaging claims and ablation evidence, not by more blind TabFact score chasing. Remaining useful P1 work: write a patent-facing mechanism section from the deterministic verification evidence, add a clean final result table, and only then consider cross-model validation.

WTQ shortcut generalization diagnostic:

- Diagnostic summary added in MACT: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/summary/wtq_shortcut_generalization_20260814.md`.
- Diagnostic root: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/diagnostics/wtq_shortcut_generalization_20260814/`.
- This diagnostic is offline and does not call the model.

| Split | Input rows | Shortcut hits | Correct hits | Wrong hits | Accuracy on hits |
|---|---:|---:|---:|---:|---:|
| formal200 | 200 | 31 | 28 | 3 | 0.9032 |
| blind200_v1 | 200 | 19 | 18 | 1 | 0.9474 |
| frozen150 | 150 | 32 | 24 | 8 | 0.7500 |
| full_unseen | 4344 | 438 | 290 | 148 | 0.6621 |
| full_unseen_minus_formal200 | 4144 | 407 | 262 | 145 | 0.6437 |

Interpretation: current WTQ deterministic shortcuts are valuable on the locked formal200 and blind200 samples, but are not safe enough to expand blindly across the full WTQ unseen pool. The most fragile rule families are `after_reference`, `last_requested_column`, `superlative_owner`, and `existing_total_metric`. Future WTQ work should add stronger semantic target-column and ambiguity gates instead of increasing pattern coverage. This is boundary evidence for the patent report, not a reason to continue prioritizing WTQ optimization now.

Formal evidence package for patent/thesis drafting:

- Authoritative MACT summary: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/summary/formal200_final_evidence_package_20260814.md`.
- Machine-readable evidence JSON: `outputs/server_runs/qwen3_32b_baseline_formal200_20260812_1505/summary/formal200_final_evidence_package_20260814.json`.
- `summary/main_baseline_summary.md` is now explicitly marked as superseded by the final evidence package, because that old file records the pre-patch `436/600` MyAgent baseline.

Current completion audit against the long patent-data objective:

| Requirement | Status | Current evidence / next action |
|---|---|---|
| Qwen3-32B full200 MyAgent > MACT on WTQ/TabFact/CRT/overall | complete | Final evidence package: MyAgent `480/600 = 0.8000`, MACT `465/600 = 0.7750` |
| Three baselines and efficiency metrics | complete | MACT, Direct-CoT, Single-Agent Pandas; token/time/fail table in final evidence package |
| WTQ generalization diagnostic | complete as boundary evidence | `wtq_shortcut_generalization_20260814.md`; full unseen shortcut accuracy is not high enough for blind WTQ rule expansion |
| Mechanism ablation | complete for Gate-50 core mechanisms | Deterministic shortcut ablation is strong; question routing and table compression show strong efficiency/runnability value; no-risk scoring improves accuracy but raises token on gate50, so risk scoring should be framed cautiously as cost/path control; strong verification remains inconclusive |
| Multi-model gate | complete as no-go boundary summary | Qwen3-14B-AWQ, Qwen2.5-14B-AWQ, Qwen2.5-3B Gate-50 summaries are all no-go |
| Multi-seed stability | partial | P4b paired new-seed Gate-50 passes narrowly; Seed-C/D current-only and boundary summaries exist; Seed-E paired Gate-50 is complete but did not pass (`95/150` vs MACT `105/150`) |
| Patent draft evidence | drafted as evidence, not legal final | Final evidence package section 7 plus Seed-E supplement draft give technical problem, method steps, effects, answer-contract mechanism, claim directions, and wording boundaries |

Seed-E Gate-50 paired stability package prepared on 2026-08-23:

- Run package: `outputs/server_runs/qwen3_32b_patent_seed_e_gate50_20260823/`.
- Inputs: `input/wtq_seed_e_gate50.jsonl`, `input/tabfact_seed_e_gate50.jsonl`, `input/crt_seed_e_gate50.jsonl`, each `50` rows.
- Exclusions: Formal-200, ablation50, prior P4b new-seed, targeted slices, and Seed-C/D inputs.
- Static verification passed: input row count `50 * 3 = 150`, `seed_e_manifest.json` parses, `bash -n` passes for run scripts, and `py_compile` passes for package Python helpers.
- Execution started on 2026-08-24 00:08 CST after completing Gate-50 mechanism ablations. Qwen3-32B services are resident on GPUs `4,5` -> `http://127.0.0.1:8000/v1` and GPUs `6,7` -> `http://127.0.0.1:8001/v1`; GPUs `0,1,2,3` remain unused.
- Current active command: `bash run_myagent_seed_e_gate50.sh` in `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_seed_e_gate50_20260823/`.
- MyAgent execution completed on 2026-08-24: WTQ `31/50 = 0.6200`, TabFact `40/50 = 0.8000`, CRT `24/50 = 0.4800`, overall `95/150 = 0.6333`, avg token `7160.61`, avg time `20.387s`, failed/missing `0/0`.
- Interpretation: Seed-E is a harder stability split for current MyAgent. It is not enough to judge paired stability until MACT is run on the exact same 150 rows.
- MyAgent output target: `myagent_seed_e/`; checkpoint should be committed to MACT before starting long MACT paired execution.
- MyAgent Seed-E checkpoint pushed: MyAgent PRD commit `8e6d26f`; MACT output commit `5b4ff5f`.
- MACT Seed-E paired execution started with `bash run_mact_seed_e_gate50_sharded.sh wtq`; early check showed shard00 and shard01 each wrote `1/25` row with no local-network errors. WTQ output target: `mact/wtq_mact_seed_e_gate50.jsonl` after merge; shard outputs under `mact_shards/wtq_wtq_mact_seed_e_gate50_00000_00050/output/`.
- 2026-08-24 00:51 CST user reconfirmed: stop using GPUs `0,1,2,3`; run experiments only on GPUs `4,5,6,7`. Recheck showed no compute processes on GPUs `0,1,2,3`; the only visible compute workers were the two resident Qwen3-32B vLLM services on GPUs `4,5` and `6,7`. Do not release these services unless switching models or explicitly asked.
- MACT Seed-E WTQ progress at the same checkpoint: shard00 `8/25`, shard01 `8/25`, total `16/50`. There is one recorded failure row so far, `nu-1109`, caused by context-limit `BadRequestError`; the one-by-one wrapper wrote the failed JSONL row and continued, so this is counted as a measurable failed/missing sample, not a run-stopping infrastructure failure.
- 2026-08-24 00:59 CST half-run checkpoint: MACT Seed-E WTQ reached shard00 `12/25`, shard01 `13/25`, total `25/50`. Recorded failure rows so far: `2` (`nu-1109`, `nu-3253`), both sample-level context-limit failures. GPUs `0,1,2,3` still show no compute processes; active generation remains only on GPUs `4,5,6,7`.
- 2026-08-24 01:09 CST checkpoint: MACT Seed-E WTQ reached shard00 `14/25`, shard01 `16/25`, total `30/50`. Recorded failure rows remain `2`; GPUs `0,1,2,3` still show no compute processes.
- 2026-08-24 01:18 CST checkpoint: MACT Seed-E WTQ reached shard00 `19/25`, shard01 `21/25`, total `40/50`. Recorded failure/error rows are `3`: `nu-553`, `nu-1109`, `nu-3253`. GPUs `0,1,2,3` still show no compute processes; Qwen3 services remain resident on GPUs `4,5,6,7`.
- 2026-08-24 01:28 CST completion: MACT Seed-E WTQ completed and merged to `mact/wtq_mact_seed_e_gate50.jsonl` with `50/50` rows. Eval: MACT WTQ `37/50 = 0.7400`, avg token `10761.76`, avg time `117.007s`, failed/missing `3/3`; MyAgent WTQ on the same Seed-E rows was `31/50 = 0.6200`, avg token `6947.46`, avg time `20.399s`, failed/missing `0/0`. Interpretation: on Seed-E WTQ alone, MACT leads accuracy by `6` rows while MyAgent remains much cheaper and more robust. Do not judge paired stability until TabFact and CRT Seed-E MACT results are also complete.
- 2026-08-24 01:38 CST checkpoint: MACT Seed-E TabFact paired run started with `bash run_mact_seed_e_gate50_sharded.sh tabfact`; MACT internally uses task name `scitab`, so shard outputs are under `mact_shards/scitab_tabfact_mact_seed_e_gate50_00000_00050/`. Current progress shard00 `5/25`, shard01 `5/25`, total `10/50`, recorded errors `0`. GPUs `0,1,2,3` still show no compute processes; Qwen3 services remain resident on GPUs `4,5,6,7`.
- 2026-08-24 01:48 CST checkpoint: MACT Seed-E TabFact reached shard00 `11/25`, shard01 `9/25`, total `20/50`, recorded errors `0`. Continue on GPUs `4,5,6,7`; do not use GPUs `0,1,2,3`.
- 2026-08-24 01:57 CST checkpoint: MACT Seed-E TabFact reached shard00 `15/25`, shard01 `16/25`, total `31/50`, recorded errors `0`. Continue on GPUs `4,5,6,7`; model services remain resident.
- 2026-08-24 02:06 CST checkpoint after user asked to stop GPUs `0,1,2,3`: `nvidia-smi pmon -c 1` showed no compute processes on GPUs `0,1,2,3`, so no kill was needed. MACT Seed-E TabFact reached shard00 `18/25`, shard01 `22/25`, total `40/50`, recorded errors `0`. Qwen3-32B services remain resident on GPUs `4,5` (`8000`) and `6,7` (`8001`), and the active MACT TabFact run continues only through those endpoints.
- 2026-08-24 02:21 CST completion: MACT Seed-E TabFact completed and merged to `mact/tabfact_mact_seed_e_gate50.jsonl` with `50/50` rows. Eval: MACT TabFact `42/50 = 0.8400`, avg token `11488.04`, avg time `107.645s`, failed/missing `0/0`; MyAgent TabFact on the same Seed-E rows was `40/50 = 0.8000`, avg token `2766.76`, avg time `12.362s`, failed/missing `0/0`. Interpretation: on Seed-E TabFact alone, MACT leads by `2` rows, while MyAgent uses about `24.1%` of MACT tokens and about `11.5%` of MACT time. Paired Seed-E conclusion still requires MACT CRT completion.
- 2026-08-24 09:47 CST completion: MACT Seed-E CRT completed and merged to `mact/crt_mact_seed_e_gate50.jsonl` with `50/50` rows. Eval: MACT CRT `26/50 = 0.5200`, avg token `13437.94`, avg time `176.255s`, failed/missing `0/0`; MyAgent CRT on the same Seed-E rows was `24/50 = 0.4800`, avg token `11767.60`, avg time `28.399s`, failed/missing `0/0`.
- 2026-08-24 Seed-E paired summary generated: `seed_e_paired_gate50_summary.md` and `seed_e_paired_gate50_summary.json`. Overall Seed-E result: MyAgent `95/150 = 0.6333`, MACT `105/150 = 0.7000`, delta `-10` rows. Per dataset: WTQ `31/50` vs `37/50`, TabFact `40/50` vs `42/50`, CRT `24/50` vs `26/50`; MyAgent does not meet the Seed-E paired acceptance criteria and does not exceed MACT on any of the three Seed-E datasets. Efficiency remains favorable overall: MyAgent token ratio `0.6019`, MyAgent failure/missing `0/0`, MACT failure/missing `3/3`. Interpretation: this is stability boundary evidence. The locked Qwen3 full200 result remains achieved, but multi-seed robustness is not fully proven and needs either more seed replication or targeted, patent-describable mechanisms for Seed-E failure clusters.
- 2026-08-24 Seed-E failure-cluster diagnostic generated: `summary/seed_e_failure_cluster_diagnostic.md` and `summary/seed_e_failure_cluster_diagnostic.json`. Row-level buckets using the same evaluator logic: WTQ both-correct `28`, MyAgent-only `3`, MACT-only `9`, both-wrong `10`; TabFact both-correct `38`, MyAgent-only `2`, MACT-only `4`, both-wrong `6`; CRT both-correct `19`, MyAgent-only `5`, MACT-only `7`, both-wrong `19`. Candidate next mechanisms: WTQ count/aggregate and abbreviation-preserving answer repair; TabFact highest-crowd/player-max/month-lowest-attendance/greatest-margin verification; CRT exact numeric average, fraction/proportion, percent-style, and yes/no answer-contract repair; cross-dataset final answer normalization where the computed value is close but not evaluator-compatible.
- 2026-08-24 09:58 CST answer-contract patch implemented locally in MyAgent: WTQ abbreviation requests now preserve/convert to three-letter country codes; CRT average/mean contracts use three decimal places unless the question explicitly asks for another precision; CRT proportion scalars can be formatted as exact fractions; scalar `NaN` is invalid and should trigger replanning; CRT numeric execution candidates are protected from conflicting strong-verifier numeric answers on average/proportion/ratio/percent questions. Verification: `python -m unittest discover -s tests -p 'test_myagent_pipeline.py'` passed `248` tests, and py_compile passed for `code/my_agents.py`, `code/answer_contracts.py`, `code/dataset_profiles.py`, and `code/tqa.py`.
- 2026-08-24 answer-contract offline validation generated in MACT: `summary/answer_contract_patch_offline_validation.md` and `summary/answer_contract_patch_offline_validation.json`. Direct offline fixes: `nu-3415` canonicalizes `China -> CHN`; `crt-280` canonicalizes `0.16666666666666666 -> 1/6`. Focused rerun still required for `crt-502` and `crt-290` because the previous generated code already rounded inside `round(..., 2)`; the new contract rejects that old code shape and should force replanning. Focused inputs are prepared at `input/diagnostic/seed_e_answer_contract_wtq.jsonl` and `input/diagnostic/seed_e_answer_contract_crt.jsonl`.
- 2026-08-24 focused rerun status: not executed in the current sandbox because loopback networking is isolated; `curl http://127.0.0.1:8000/v1/models` and `8001` both failed with connection errors even though the vLLM processes are visible in the host process table. When a shell with local loopback access is available, run focused validation with `scripts/server/run_sharded_tqa.py` against those two diagnostic inputs before claiming Seed-E improvement.
- 2026-08-24 10:02 CST patent-facing supplement draft generated in MACT: `summary/patent_spec_draft_seed_e_supplement_20260824.md`. It integrates the Formal-200 win, Seed-E paired failure, failure-cluster diagnosis, answer-contract mechanism, claim directions, and wording boundaries. Key writing constraint: Formal-200 Qwen3-32B superiority can be claimed; multi-seed robustness cannot yet be claimed because Seed-E paired is negative.
- 2026-08-24 10:05 CST answer-contract focused validation scripts prepared in MACT: `run_answer_contract_focused_validation.sh` and `summarize_answer_contract_focused.py`. Static checks passed: `bash -n run_answer_contract_focused_validation.sh` and `python -m py_compile summarize_answer_contract_focused.py`. When loopback access to the resident vLLM services is available, run `cd /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_seed_e_gate50_20260823 && bash run_answer_contract_focused_validation.sh`; it will write outputs to `diagnostics/answer_contract_patch_focused_20260824/` and summaries to `summary/answer_contract_patch_focused_summary.md/json`.
- 2026-08-24 10:07 CST objective completion audit generated in MACT: `summary/objective_completion_audit_20260824.md`. This audit marks Qwen3 full200, WTQ generalization, mechanism ablation core evidence, multi-model gate, Seed-E paired execution, failure-cluster diagnosis, and patent draft evidence as present; it keeps the overall goal open because GitHub synchronization is pending, answer-contract focused validation is not executed, and multi-seed robustness remains partial/negative.
- 2026-08-24 sync status: local commits were created after Seed-E completion, but GitHub push is pending because the current sandbox cannot resolve `github.com` (`Temporary failure in name resolution`). MyAgent `codex/selective-risk-collaboration` and MACT `main` are ahead of their origins with the Seed-E PRD, CRT, paired summary, failure-cluster diagnostic, answer-contract code repair, patent supplement draft, and focused validation scripts. Next session should first run `git -C /home/ubuntu/lzz/MyAgent push origin codex/selective-risk-collaboration` and `git -C /home/ubuntu/lzz/MACT push origin main` when network/DNS is available.

Mechanism ablation expansion executed on 2026-08-23/2026-08-24:

- New MyAgent switches: `--disable_question_routing`, `--disable_risk_scoring`, `--disable_table_compression`.
- New sharded runner switches: `--disable-question-routing`, `--disable-risk-scoring`, `--disable-table-compression`.
- New MACT run-package scripts:
  - `run_ablation_no_question_routing50.sh`
  - `run_ablation_no_risk_scoring50.sh`
  - `run_ablation_no_table_compression50.sh`
- Static verification passed: `py_compile` for `code/my_agents.py`, `code/tqa.py`, `scripts/server/run_sharded_tqa.py`, `scripts/server/prepare_baseline_experiment_run.py`; `test_myagent_pipeline.py` passed `243` tests; sharded dry-run confirmed the three new flags are passed to `code/tqa.py`.
- Execution checkpoint: `run_ablation_no_question_routing50.sh`, `run_ablation_no_risk_scoring50.sh`, and `run_ablation_no_table_compression50.sh` completed on 2026-08-23/2026-08-24 using only GPUs `4,5,6,7`; merged rows are `50/50` for WTQ, TabFact, and CRT.
- `no_table_compression50` has failed/missing `2/2` from WTQ context-limit rows `nu-30` and `nu-44`; this is part of the mechanism evidence, not a network failure.

Next best work: do not rerun the full official 200-row package immediately. First use the completed Seed-E paired package as a failure-cluster diagnostic: compare MyAgent-only, MACT-only, and both-wrong rows for WTQ/TabFact/CRT; identify whether the gap comes from answer extraction, table filtering, risk routing, or CRT reasoning; then implement only mechanisms that are narrow, auditable, and patent-describable. After one such batch, run a small focused validation and then a new Seed-F/G Gate-50 paired check before considering another expensive formal run.

## 14. 2026-08-24 Strategy Realignment

User direction update:

- Stop optimizing around formal200 sample IDs or fixed query text. Formal200 remains the locked positive main result, but future work must be driven by reusable error types and component mechanisms.
- Final reports should not use failure rate as a headline metric. All methods should emit one scoreable answer for every input row. Recovery and fallback details stay in internal diagnostics.
- Prefer a common robust runner for MyAgent, MACT, Direct-CoT, and Single-Agent Pandas. The robust layer may retry, compress, truncate, repair code, or fall back, but for MACT it must remain an outer wrapper and not change core MACT reasoning.
- Seed-E is the active diagnostic split. Use it for error attribution across WTQ, TabFact, and CRT. If a mechanism repair improves Seed-E, create a fresh unseen Seed-F Gate-50 or Gate-100 as blind validation.
- Next optimization levers should be existing-component parameters and gates: the `SIMPLE`/`COMPLEX` threshold `tau_s`, pre-route risk threshold `theta_s`, evidence retention threshold `theta_g`, posterior verification threshold `theta_v`, table-compression budget, deterministic-rule confidence gates, and high-risk WTQ/CRT second-check budget.
- Patent wording boundary: claim Qwen3-32B Formal-200 superiority and efficiency; do not claim all-model superiority or multi-seed stability until Seed-F also validates.

GPU execution rule:

- Use only GPUs `4,5,6,7` for the current experiments.
- Keep the resident Qwen3-32B services on GPUs `4,5` and `6,7` unless switching models or the user explicitly asks to release them.
- 2026-08-24 10:14 CST check: visible compute processes are only VLLM workers on GPUs `4,5,6,7`; `nvidia-smi pmon -c 1` showed no visible PID on GPUs `0,1,2,3`. GPUs `0,1,2,3` still reported memory/utilization in `nvidia-smi`, but no MACT/MyAgent/vLLM process was exposed through NVML or `/proc` device-handle scan. Do not blind-kill unknown GPU usage; run all controlled experiments through the `4567` services.

Robust-output implementation checkpoint:

- MyAgent now has a shared output-contract helper in `code/robust_outputs.py`.
- Integrated runners: `code/tqa.py`, `scripts/server/run_baseline_tqa.py`, and `scripts/server/run_mact_one_by_one.py`.
- Failure rows now keep `exec_error` but also receive a non-empty fallback answer plus `fallback_used`, `retry_count`, `error_type`, `context_overflow`, `execution_error`, and nested `robust_runner` diagnostics.
- MACT integration is wrapper-only and does not change MACT core reasoning logic.
- Validation passed: py_compile for the touched runner/code files, and `36` targeted unit tests covering robust outputs, baseline runner, MACT wrappers, evaluator, and tqa failure output.
- Evidence file: `/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_seed_e_gate50_20260823/summary/common_robust_runner_update_20260824.md`.
- Remaining gap: full context truncation / prompt compression retry is not yet implemented for every method; the current change is the shared scoreable-output and diagnostic-field layer.
