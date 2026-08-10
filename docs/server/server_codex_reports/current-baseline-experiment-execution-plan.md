# Current Baseline Experiment Execution Plan

Last updated: 2026-08-10 CST

Audience: server-side Codex agent controlling `/home/ubuntu/lzz/MyAgent` and `/home/ubuntu/lzz/MACT`.

This document supersedes ad-hoc "keep testing Qwen3" instructions. The goal is now to turn the existing MyAgent-vs-MACT development evidence into a thesis/patent-grade experiment package with at least three baselines, controlled cost, and reproducible same-sample evaluation.

## 1. Current Decision

The Qwen3-32B MyAgent-vs-MACT strict target is already satisfied for the current staged paired evidence:

- WTQ: MyAgent `76/100` vs MACT `74/100`.
- TabFact: MyAgent `91/100` vs MACT `87/100`.
- CRT: MyAgent `65/100` vs MACT `62/100`.
- Overall: MyAgent `232/300` vs MACT `223/300`.
- Overall token ratio: `0.5662`.

Do not continue spending GPU time on single-dataset Qwen3 tuning unless a concrete thesis/patent gap is identified. The next work item is baseline expansion and formal experiment packaging.

## 2. Thesis-Level Baseline Set

Use the following method ladder. It gives weak, medium, recent, and strong baselines without turning the experiment into an unbounded benchmark survey.

| Method | Role | Required | Implementation priority |
|---|---|---:|---:|
| Direct-CoT | Weak baseline: one LLM directly answers from serialized table and question | yes | P0 |
| Single-Agent Pandas / I2R-style | Medium baseline: one agent generates executable Pandas code with retry, no MyAgent risk collaboration | yes | P0 |
| POS-style SQL baseline | Recent 2024 baseline: decompose question into atomic SQL-like steps | yes, unless too costly | P1 |
| SynTQA-style fallback | Recent 2024 fallback: route between Text-to-SQL and end-to-end table QA | optional fallback if POS is blocked | P1 fallback |
| MACT | Strong baseline and primary paper target | yes | already partially available; only extend when same-ID paired evidence is needed |
| MyAgent | Proposed method | yes | already available |

Important distinction:

- Baselines prove MyAgent is better than other methods.
- Ablations prove MyAgent's internal modules are useful.
- Ablations do not replace the "at least three baselines" requirement.

## 3. Datasets and Evaluation Policy

Use the same three datasets already adapted in this project:

- WTQ
- TabFact
- CRT

All main-table comparisons must satisfy:

- same sample IDs,
- same model,
- same answer evaluator,
- same token accounting where possible,
- same output schema or an adapter into the common schema consumed by `code/evaluate_results.py`.

MACT paper-reported numbers may be cited in related work or discussion, but they must not be mixed into the strict main result table because model, prompt, split, and evaluator conditions differ.

## 4. Execution Stages

### Stage A: Runner Readiness

Implement or verify runners for:

1. Direct-CoT baseline.
2. Single-Agent Pandas baseline.
3. POS-style SQL baseline, or SynTQA-style fallback if POS is blocked.

Each runner must write one JSONL row per input row and include at least:

- `id`
- `dataset` or task name
- `answer` / prediction field accepted by the evaluator
- gold answer field preserved from input
- token usage if available
- elapsed seconds if available
- failure status if the sample cannot be answered

Do not start a large run until all three candidate baselines pass a 5-row smoke test on WTQ, TabFact, and CRT.

### Stage B: Gate-50 Baseline Screening

Run each method on WTQ / TabFact / CRT, `50` rows per dataset:

- Direct-CoT
- Single-Agent Pandas
- POS-style SQL or SynTQA-style fallback
- MyAgent
- MACT only if the current same-ID MACT outputs are missing for the selected sample

Gate-50 goals:

- confirm output schema and evaluator compatibility,
- estimate accuracy, token, runtime, and failure rate,
- remove a baseline only if it is technically blocked or produces unusable outputs.

For thesis reporting, even weak baselines can remain if they are stable and fairly evaluated.

### Stage C: Formal Main Experiment

After Gate-50 passes, run the formal comparison:

- Default size: `200` rows per dataset.
- Methods: Direct-CoT, Single-Agent Pandas, POS/SynTQA, MACT, MyAgent.
- Model: Qwen3-32B local unless the supervisor explicitly asks for a different main model.

This produces the main thesis table:

| Method | WTQ Acc | TabFact Acc | CRT Acc | Overall Acc | Avg Token | Avg Time | Fail/Missing |
|---|---:|---:|---:|---:|---:|---:|---:|

Only consider `500` rows per dataset or full datasets after the `200`-row formal comparison is stable and the supervisor asks for stronger statistical evidence.

### Stage D: Ablation Experiment

Run MyAgent ablations separately from baseline comparisons. Recommended sample size:

- `50` rows per dataset if GPU time is tight.
- `100` rows per dataset if results are noisy.

Recommended ablation variants:

1. No question-type routing.
2. No dual scoring / risk scoring.
3. No table compression or evidence retention.
4. No strong verification / selective collaboration.

The ablation table should explain patent mechanisms:

- problem type differentiation,
- dual scoring mechanism,
- table information compression,
- high-risk collaboration or second verification,
- answer canonicalization / deterministic audit.

### Stage E: Multi-Model Boundary Experiment

Do not run every baseline on every model. That is too expensive and not necessary for a master's thesis.

Recommended model set:

- Qwen3-32B: main model, full baseline comparison.
- Qwen3-14B-AWQ: already tested as no-go for MyAgent Gate-50; cite as boundary unless rerun is explicitly required.
- Qwen2.5-14B-Instruct-AWQ: already tested as no-go for MyAgent Gate-50; cite as boundary unless rerun is explicitly required.
- Qwen2.5-3B or 7B: small-model boundary only.

If a genuinely new model appears, use the existing model-gate protocol:

1. Gate-10 smoke.
2. MyAgent-only Gate-50.
3. Gate-150 only if Gate-50 is close to Qwen3-32B.
4. Paired-200 against MACT only if Gate-150 remains competitive.

Do not rerun known no-go models unless the PRD or user explicitly says why the rerun is needed.

## 5. Immediate Server To-Do List

### P0: Synchronize Code

```bash
cd /home/ubuntu/lzz/MyAgent
git fetch origin
git checkout codex/selective-risk-collaboration
git pull --ff-only origin codex/selective-risk-collaboration
```

Then inspect:

```bash
sed -n '1,220p' docs/server/server_codex_reports/current-baseline-experiment-execution-plan.md
sed -n '1,120p' docs/server/server_codex_reports/current-qwen3-mact-experiment-prd.md
```

### P1: Audit Existing Baseline Runner Support

Check whether the project already has reusable runner pieces:

```bash
cd /home/ubuntu/lzz/MyAgent
rg -n "direct|cot|pandas|sql|baseline|ablation|evaluate_results|run_sharded" code scripts tests docs
```

Report which of the three new baselines already exists and which must be implemented.

### P2: Implement Missing Baseline Runners

Implement only the minimal stable runner surface first:

- one script per baseline, or one script with `--method direct_cot|pandas_agent|pos_sql`;
- shared dataset loading via existing adapted JSONL;
- output JSONL compatible with `code/evaluate_results.py`;
- resume support if practical;
- smoke command in docs or generated README.

Do not change MyAgent policy logic during baseline runner implementation.

### P3: Smoke Test

Run each new baseline on 5 rows per dataset. The expected deliverable is:

- raw output JSONL,
- evaluation JSON,
- a short summary markdown,
- errors/logs if any.

Only after this should Gate-50 start.

### P4: Gate-50

Run 50 rows per dataset for the baseline set. The expected deliverable is one summary table with:

- WTQ / TabFact / CRT / overall accuracy,
- average token,
- average time,
- failed/missing count,
- notes on schema or evaluator issues.

### P5: Formal-200

Run 200 rows per dataset only after Gate-50 outputs are complete and stable.

## 6. Stop Conditions

Stop and report instead of continuing if:

- a baseline cannot produce one output row per input row,
- evaluator cannot compare the baseline fairly,
- token accounting is unavailable and cannot be approximated consistently,
- a new model is not actually available,
- the run would repeat a known no-go model without a new reason.

## 7. Final Thesis/Patent Tables

Target final report tables:

1. Main baseline comparison: MyAgent vs Direct-CoT vs Single-Agent Pandas vs POS/SynTQA vs MACT.
2. Efficiency comparison: accuracy, token, time, failure rate.
3. Ablation comparison: remove each patent mechanism.
4. Multi-model boundary: Qwen3-32B main plus no-go/limited evidence for smaller models.
5. Optional difficulty split: simple vs hard questions based on existing risk/type classifier.

## 8. Recommended Wording Boundary

Allowed:

- "Under Qwen3-32B and same-ID paired staged samples, MyAgent exceeds MACT on WTQ, TabFact, and CRT with lower overall token usage."
- "The method is evaluated against at least three baselines: direct prompting, single-agent tool use, a recent SQL/table-reasoning baseline, and MACT."
- "Smaller local models show lower accuracy and are treated as boundary evidence."

Not allowed unless new evidence is produced:

- "All full datasets are completed."
- "All tested models beat MACT."
- "MACT paper-reported numbers are directly comparable to our same-sample table."
- "Every recent table-QA paper has been reproduced."

