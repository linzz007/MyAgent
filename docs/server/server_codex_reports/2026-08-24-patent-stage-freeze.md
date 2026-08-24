# Patent Stage Freeze: 2026-08-24

This document freezes the current patent-facing MyAgent direction so later server-side experiments do not keep adding unrelated layers.

## 1. Freeze Decision

As of 2026-08-24, the patent method is frozen at the component level. The current code and patent specification should be treated as the baseline design for final patent drafting and thesis experiments.

The top-level method is:

1. normalize the table and detect the answer contract;
2. compute semantic complexity and structural/evidence complexity;
3. compress the table into a question-aware evidence table;
4. route the problem to `SIMPLE` or `COMPLEX`;
5. solve the problem through lightweight lookup or complex reasoning;
6. compute posterior risk;
7. trigger selective collaboration verification only when the answer is high-risk or inconsistent;
8. normalize the final answer and emit one scoreable output row.

## 2. Frozen Patent Components

| Component | Meaning in the patent |
|---|---|
| Question-type recognition | Determines answer form, operation type, task family, semantic complexity, and structural complexity. |
| Dual scoring | Uses complexity scoring for route selection and risk scoring for verification/cost control. |
| Table compression | Preserves headers, candidate rows, candidate columns, and question-relevant cells while reducing irrelevant table content. |
| Two-path route | Uses only `SIMPLE` and `COMPLEX` as top-level routes. |
| Deterministic table validation | Reproducible table operations and answer normalization inside the existing route/risk framework. This is not a new route. |
| Selective collaboration verification | A conditional verification sub-process inside `COMPLEX`, triggered by high posterior risk or answer conflict. |
| Robust scoreable output | Runner/output contract that keeps every row measurable and records diagnostic fields. |

## 3. Allowed Future Work

Only the following changes are allowed without reopening the patent design:

- adjust formula weights and thresholds;
- adjust table-compression budget and evidence retention threshold;
- refine answer-contract normalization;
- refine confidence gates for deterministic table validators;
- improve high-risk verification triggers inside `COMPLEX`;
- add retry/fallback behavior inside the robust output contract;
- fix reusable error categories found by Seed-E/Seed-F diagnostics.

## 4. Forbidden Before Reopening

Do not add:

- rules keyed to sample IDs;
- rules keyed to exact fixed question strings;
- new top-level routes beyond `SIMPLE` and `COMPLEX`;
- new named layers that cannot be mapped to a frozen component;
- broad claims of all-model or multi-seed superiority without fresh blind validation.

## 5. Patent Specification Checkpoint

Current candidate final patent files:

- `D:\AAAcode\AAA毕业相关\专利信息\lzz-成本感知表格问答路由\说明书7.0-定稿版.docx`
- `D:\AAAcode\AAA毕业相关\专利信息\lzz-成本感知表格问答路由\说明书7.0-定稿版.pdf`

The patent wording is aligned with the current code direction:

- old `Light` / `Tool` / `Collab` wording has been removed as a top-level route;
- formula (4) uses the two-path `SIMPLE` / `COMPLEX` route;
- deterministic table validation is described as a tool capability under the existing route/risk framework;
- selective collaboration verification is described as a conditional high-risk verification sub-process;
- Seed-E instability is not claimed as solved in the patent text.

## 6. Experiment Boundary

The positive Qwen3-32B Formal-200 result remains the current main evidence:

- MyAgent: `480/600 = 0.8000`;
- MACT: `465/600 = 0.7750`;
- Direct-CoT: `386/600 = 0.6433`;
- Single-Agent Pandas: `421/600 = 0.7017`;
- MyAgent average token: `6293.12`;
- MACT average token: `11318.89`;
- token ratio to MACT: `0.5560`.

Seed-E is diagnostic boundary evidence, not the final positive claim. Future optimization should start from Seed-E failure clusters and must be validated on a fresh unseen Seed-F/G split before being used as a generalization claim.
