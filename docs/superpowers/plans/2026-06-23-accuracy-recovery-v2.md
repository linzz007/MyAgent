# Accuracy Recovery V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover myAgent accuracy with typed answers and selective reasoning while preserving a substantial token advantage over MACT.

**Architecture:** A focused `answer_contracts.py` module infers output shape, closed-label risk, normalization, and validation from the question. `TQASessionState` carries that contract through routing, prompting, execution, validation, and JSONL observability. Only high-risk closed-label questions are escalated to the existing Planner/Calculator/Critic loop.

**Tech Stack:** Python 3, pandas, unittest, DeepSeek OpenAI-compatible API, JSONL

---

### Task 1: Answer contract module

**Files:**
- Create: `code/answer_contracts.py`
- Create: `tests/test_answer_contracts.py`

- [x] **Step 1: Write failing inference tests**

Test that entity-list questions produce a list contract, allowed labels are
derived from each answer mode, numeric/comparison fact checks require
reasoning, and simple existence checks do not.

- [x] **Step 2: Run inference tests and confirm RED**

Run `python -m unittest discover -s tests -p "test_answer_contracts.py" -v`.
Expected: import failure because `answer_contracts` does not exist.

- [x] **Step 3: Implement contract inference**

Create immutable `AnswerContract(kind, allowed_labels, reasoning_required,
instructions)` and `infer_answer_contract(question, answer_mode)` without any
gold-answer input.

- [x] **Step 4: Write failing normalization and validation tests**

Cover boolean-to-label mapping, entity-list validation, comma-delimited list
normalization, scale-number formatting, and stray escaped quotes.

- [x] **Step 5: Implement normalization and validation**

Add `normalize_contract_value(value, contract)` and
`validate_contract_value(value, contract) -> (bool, reason)`.

- [x] **Step 6: Run focused tests and confirm GREEN**

Require all answer-contract tests to pass.

### Task 2: Carry contracts through prompts and state

**Files:**
- Modify: `code/my_agents.py`
- Modify: `tests/test_myagent_pipeline.py`

- [x] **Step 1: Write failing state and prompt tests**

Assert state infers a contract, Planner receives its instructions, and Planner
and Critic prompts no longer claim every table is Chinese governmental data.

- [x] **Step 2: Confirm RED**

Run the pipeline test file and require the new assertions to fail.

- [x] **Step 3: Implement state and prompt integration**

Infer the contract in `TQASessionState`, add `{answer_contract}` to Planner and
Critic prompts, and make the wording domain-neutral.

- [x] **Step 4: Confirm GREEN**

Run `test_myagent_pipeline.py` and preserve all existing behavior.

### Task 3: Selective closed-label reasoning and contract replanning

**Files:**
- Modify: `code/my_agents.py`
- Modify: `tests/test_myagent_pipeline.py`

- [x] **Step 1: Write failing selective-routing test**

Use a fake LLM to prove a numeric true/false statement uses Planner/Calculator
instead of the classifier, while a simple existence statement still uses the
classifier.

- [x] **Step 2: Write failing contract-replan test**

Return an integer for an entity-list question on the first plan and a list on
the second. Assert one replan occurs and the final structured value is a list.

- [x] **Step 3: Confirm RED**

Run the pipeline tests and verify both failures are caused by the missing
selective reasoning and contract validation.

- [x] **Step 4: Implement selective escalation and validation**

Force high-risk closed-label states through the complex path, normalize after
execution, replan invalid outputs, and skip cosmetic LLM formatting for labels
and lists.

- [x] **Step 5: Confirm GREEN**

Run focused and full myAgent unit tests.

### Task 4: Persist observability and benchmark

**Files:**
- Modify: `code/tqa.py`
- Modify: `tests/test_task_modes.py`
- Create: `outputs/benchmark18/myagent_*_accuracy_v2.jsonl`
- Modify: `BENCHMARK18_REPORT_2026-06-23.md`

- [x] **Step 1: Write failing output-field test**

Require JSONL items to include `answer_contract`, `risk_escalated`, and
`contract_validation` without exposing gold to runtime decisions.

- [x] **Step 2: Implement output fields and confirm GREEN**

Persist the three fields and run all unit tests.

- [x] **Step 3: Rerun the three fixed 18-sample myAgent inputs**

Use the same DeepSeek model, temperature, thinking setting, and evaluator as
the existing benchmark. Preserve previous outputs under distinct filenames.

- [x] **Step 4: Compare accuracy and token usage**

Accept the revision only if it meets the design success criteria. Diagnose any
regression by failure category rather than tuning individual sample IDs.

- [x] **Step 5: Final verification and report update**

Run all 46+ myAgent tests, compile checks, result-schema validation,
`git diff --check`, and API-key scan. Update the benchmark report with the v2
result and a clear keep/revert decision.
