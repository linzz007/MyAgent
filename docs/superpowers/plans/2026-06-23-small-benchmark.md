# Small Cross-Project Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a fair 18-sample-per-dataset comparison of MACT and myAgent using dataset-aware accuracy and provider-returned API token usage.

**Architecture:** A deterministic sampler creates three shared JSONL inputs from the fully adapted datasets. The existing common evaluator gains dataset-specific correctness functions, while myAgent's API callable records the same usage counters MACT already emits. Both projects run unchanged primary configurations, then repeated systemic failures are fixed with TDD and the same samples are rerun.

**Tech Stack:** Python 3, unittest, pandas, OpenAI-compatible DeepSeek API, JSONL

---

### Task 1: Deterministic diverse sampler

**Files:**
- Create: `code/sample_benchmark.py`
- Create: `tests/test_sample_benchmark.py`

- [x] **Step 1: Write failing tests**

Test that `select_records(records, sample_size, seed, dataset)` is deterministic, selects 18 records when available, avoids repeated table ids before reuse, balances TabFact labels, and includes both explicit Yes/No and general CRT records.

- [x] **Step 2: Confirm RED**

Run `python -m unittest discover -s tests -p "test_sample_benchmark.py" -v` and require an import failure for the missing module.

- [x] **Step 3: Implement sampler and CLI**

Read an adapted JSONL, categorize records by dataset, shuffle with `random.Random(seed)`, select across categories and table ids, then write UTF-8 JSONL plus a JSON summary of table/category counts.

- [x] **Step 4: Confirm GREEN**

Run the focused test and require all sampler tests to pass.

### Task 2: Actual API usage in myAgent

**Files:**
- Modify: `code/model_backends.py`
- Modify: `code/tqa.py`
- Modify: `tests/test_model_backends.py`

- [x] **Step 1: Write failing usage test**

Extend the fake completion with usage values and assert the returned myAgent callable exposes cumulative `snapshot()` metrics for request, prompt, completion, and total tokens.

- [x] **Step 2: Confirm RED**

Run `python -m unittest discover -s tests -p "test_model_backends.py" -v` and require failure because the callable has no usage snapshot.

- [x] **Step 3: Implement usage-tracking callable**

Replace the closure with a callable object that records provider usage while preserving the existing request shape. In `tqa.py`, calculate before/after deltas per sample and write `api_metrics` alongside estimated `llm_metrics`.

- [x] **Step 4: Confirm GREEN**

Run model-backend tests and a one-record fake entry-point test; require actual usage values in output.

### Task 3: Dataset-aware common evaluator

**Files:**
- Modify: `code/evaluate_results.py`
- Modify: `tests/test_evaluate_results.py`

- [x] **Step 1: Write failing metric tests**

Cover WTQ denotation cardinality and normalization, TabFact true/false canonicalization, CRT numeric/text matching, identical prediction extraction for MACT/myAgent schemas, and API usage precedence over estimates.

- [x] **Step 2: Confirm RED**

Run `python -m unittest discover -s tests -p "test_evaluate_results.py" -v`; require the WTQ multi-answer and dataset-metric assertions to fail.

- [x] **Step 3: Implement dataset metrics**

Add `dataset_accuracy(row)`, Python-3 WTQ normalization/denotation matching, TabFact label mapping, CRT normalized matching, per-dataset summaries, and an explicit `accuracy_metric` field. Retain `exact_match` as a compatibility metric.

- [x] **Step 4: Confirm GREEN**

Run focused and full myAgent tests.

### Task 4: Build shared samples and run primary benchmark

**Files:**
- Create: `datasets_ready/benchmark18/*.jsonl`
- Create: `outputs/benchmark18/*.jsonl`

- [x] **Step 1: Generate full adapted datasets**

Create full WTQ pristine-unseen, TabFact test, and CRT-QA JSONL files with `dataset_adapters.py`; verify expected record counts and no missing tables.

- [x] **Step 2: Create fixed samples**

Run `sample_benchmark.py` with sample size 18 and seed 20260623 for each dataset. Verify 18 records, table diversity, TabFact label coverage, and CRT answer-mode coverage.

- [x] **Step 3: Run myAgent**

Run WTQ, TabFact (`scitab`), and CRT with DeepSeek Flash, thinking disabled, temperature 0, timeout 180, eight retries, and explicit output files.

- [x] **Step 4: Run MACT**

Run the same three JSONL files with the same model settings, plan/code sample 1, and maximum three steps.

- [x] **Step 5: Evaluate all six outputs**

Require 18 output records per run, record count parity, actual API token measurements, and zero missing predictions before comparing accuracy.

### Task 5: Diagnose, optimize, and report

**Files:**
- Modify only evidence-backed code/tests from Tasks 1-3 if required
- Create: `BENCHMARK18_REPORT_2026-06-23.md`

- [x] **Step 1: Categorize every mismatch**

Classify mismatches as evaluator, adapter/gold ambiguity, routing/compression, execution, or model reasoning. Count categories by project and dataset.

- [x] **Step 2: Fix repeated system failures with TDD**

For any repeated non-model failure, write a failing regression test, implement the smallest correction, run focused/full tests, and rerun affected shared samples. Do not tune isolated ambiguous examples.

- [x] **Step 3: Write comparison report**

Report dataset-aware accuracy, actual prompt/completion/total tokens, calls, elapsed time, failures, compression, deltas between projects, optimization evidence, limitations, and the recommended server configuration.

- [x] **Step 4: Final verification**

Run both complete test suites, compile checks, `git diff --check`, result-schema validation, and a repository scan confirming no API-key-shaped value was written.
