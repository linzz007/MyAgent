# Current Baseline Experiment PRD

Last updated: 2026-08-12 CST

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

