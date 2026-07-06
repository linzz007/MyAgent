# Generalization V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dataset-aware contracts and table evidence profiles that improve blind generalization while preserving myAgent's token advantage and preventing gold/ID leakage.

**Architecture:** A new `dataset_profiles.py` module emits gold-independent contract hints and prompt constraints for WTQ, TabFact, and CRT. The existing unified Router/Compressor/Planner/Calculator/Critic pipeline consumes these hints plus full-table column profiles, prioritizes real execution errors during replanning, and retains the same output/evaluation boundary.

**Tech Stack:** Python 3, pandas, unittest, DeepSeek OpenAI-compatible API, JSONL

---

### Task 1: Dataset profile and contract hints

**Files:**
- Create: `code/dataset_profiles.py`
- Create: `tests/test_dataset_profiles.py`

- [ ] **Step 1: Write failing tests for dataset hints**

Cover these gold-independent cases:

```python
def test_wtq_plural_which_question_requests_list():
    hints = infer_dataset_hints("wtq", "Which games had above 2 medals?", df)
    assert hints.kind == "list"

def test_tabfact_numeric_statement_requires_reasoning():
    hints = infer_dataset_hints("tabfact", "3 nations have 3 silver medals", df)
    assert hints.reasoning_required is True

def test_crt_home_or_away_is_not_yes_no():
    hints = infer_dataset_hints("crt", "Was the team better at home or away?", df)
    assert hints.answer_mode == ""

def test_crt_integer_metric_average_uses_zero_decimals():
    hints = infer_dataset_hints("crt", "What is the average Roll?", integer_df)
    assert hints.decimal_places == 0

def test_crt_double_count_requests_tuple():
    hints = infer_dataset_hints("crt", "How many wins by decision and how many by finish?", df)
    assert hints.kind == "tuple"
    assert hints.arity == 2
```

- [ ] **Step 2: Run the profile tests and confirm RED**

Run:

```powershell
python -m unittest discover -s tests -p "test_dataset_profiles.py" -v
```

Expected: import failure because `dataset_profiles` does not exist.

- [ ] **Step 3: Implement `DatasetHints` and `infer_dataset_hints`**

Create an immutable dataclass:

```python
@dataclass(frozen=True)
class DatasetHints:
    dataset: str
    answer_mode: str = ""
    allowed_labels: tuple[str, ...] = ()
    kind: str = ""
    arity: int | None = None
    decimal_places: int | None = None
    reasoning_required: bool | None = None
    prompt_instructions: str = ""
    missing_markers: tuple[str, ...] = ("", "nan", "none", "n/a", "tba", "unknown")
```

Implement only question/schema-derived rules. Do not accept answer, ID, table ID, or evaluator fields.

- [ ] **Step 4: Run profile tests and confirm GREEN**

Run the focused test file and require all tests to pass.

### Task 2: AnswerContract V3

**Files:**
- Modify: `code/answer_contracts.py`
- Modify: `tests/test_answer_contracts.py`

- [ ] **Step 1: Write failing tuple and override tests**

Add tests proving `infer_answer_contract` accepts `allowed_labels`, `kind`, `arity`,
`decimal_places`, `reasoning_required`, and `extra_instructions` overrides. Add tuple
normalization tests for `(3, 12)`, `[3, 12]`, and `"3, 12"`.

- [ ] **Step 2: Confirm RED**

Run `test_answer_contracts.py`; expected failure is the missing override API and tuple kind.

- [ ] **Step 3: Implement override-aware contracts**

Extend the dataclass with `arity`. Keep all defaults backward-compatible. For tuple
contracts, normalize to a list with exactly `arity` concise values and reject explanatory
strings such as `"3 by decision and 12 by finish"`.

- [ ] **Step 4: Confirm GREEN**

Run answer-contract and pipeline tests.

### Task 3: Full-table column profiles

**Files:**
- Modify: `code/my_agents.py`
- Modify: `tests/test_myagent_pipeline.py`

- [ ] **Step 1: Write failing schema-profile tests**

Build a DataFrame where the first eight `Fate` values are `Fell` and later values include
`Pulled Up`, `Brought Down`, and `Refused`. Assert `_build_table_schema` exposes all four
representative values. Add numeric, text, mixed and `TBA` marker assertions.

- [ ] **Step 2: Confirm RED**

Run `test_myagent_pipeline.py`; expected failure is missing `column_profiles` and
`column_profiles_text`.

- [ ] **Step 3: Implement deterministic column profiling**

For each column record:

```python
{
    "dtype": "float64",
    "semantic_type": "numeric",
    "non_null": 12,
    "unique_count": 8,
    "representative_values": ["Fell", "Pulled Up", "Brought Down", "Refused"],
    "missing_marker_count": 1,
}
```

Cap representative values and serialized text to keep prompt growth bounded. Sample values
across the complete compressed DataFrame, not only `head(8)`.

- [ ] **Step 4: Add profiles to Planner and Critic prompts**

Include dataset instructions and column-profile text in both prompts. Replace the Router's
"Chinese question" wording with domain-neutral wording.

- [ ] **Step 5: Confirm GREEN and inspect prompt size**

Run focused tests and assert representative text stays under a fixed bound for wide tables.

### Task 4: Execution recovery and safe runtime

**Files:**
- Modify: `code/my_agents.py`
- Modify: `tests/test_myagent_pipeline.py`

- [ ] **Step 1: Write failing runtime tests**

Add tests for:

- `import re` followed by regex parsing.
- `isinstance(value, str)` in a helper function.
- a forbidden `import os` remaining blocked.
- first planner code failing with a `.str` dtype error and second planner prompt receiving
  the exact execution error rather than only a contract-empty message.

- [ ] **Step 2: Confirm RED**

Run focused pipeline tests and verify the expected import/builtin/feedback failures.

- [ ] **Step 3: Implement minimal safe additions**

Add `re` to approved modules and `isinstance` to allowed builtins. In the complex loop,
branch on `not state.exec_success` before answer normalization:

```python
if not state.exec_success:
    state.critic_verdict = "REPLAN"
    state.critic_feedback = f"Execution failed: {state.exec_error}"
    continue
```

- [ ] **Step 4: Confirm GREEN**

Run focused and full test suites.

### Task 5: Wire profiles through runtime and observability

**Files:**
- Modify: `code/tqa.py`
- Modify: `code/my_agents.py`
- Modify: `tests/test_task_modes.py`
- Modify: `tests/test_myagent_pipeline.py`

- [ ] **Step 1: Write failing state/wiring tests**

Assert that a record's `source_dataset` selects the profile, the generated AnswerContract
contains profile overrides, Planner receives profile instructions, and JSONL observability
persists `dataset_profile` without gold-derived fields.

- [ ] **Step 2: Confirm RED**

Run task-mode and pipeline tests.

- [ ] **Step 3: Implement runtime wiring**

Build the DataFrame before inferring hints. Pass the resulting contract, profile name,
instructions and missing markers into `TQASessionState`. Preserve the current CLI and legacy
call sites by using optional constructor arguments.

- [ ] **Step 4: Confirm GREEN**

Run all myAgent tests and both existing benchmark result-schema checks.

### Task 6: CRT tuple evaluation and regression suite

**Files:**
- Modify: `code/evaluate_results.py`
- Modify: `tests/test_evaluate_results.py`

- [ ] **Step 1: Write failing CRT tuple tests**

Assert prediction `[3, 12]` matches gold `"3, 12"`, order is preserved, and `[12, 3]`
does not match. Keep scalar and existing CRT matching behavior unchanged.

- [ ] **Step 2: Confirm RED**

Run `test_evaluate_results.py` and verify tuple matching fails under the current generic EM.

- [ ] **Step 3: Implement CRT structured matching**

Add a CRT-specific matcher that splits a single comma-delimited gold only when prediction is
a multi-value list/tuple. Do not alter WTQ or TabFact metrics.

- [ ] **Step 4: Confirm GREEN**

Run evaluator tests and recompute benchmark18/blind36 summaries without changing their raw
outputs.

### Task 7: Development-set validation and final gate

**Files:**
- Create: `outputs/generalization_v3_dev/`
- Modify: `docs/superpowers/plans/2026-06-24-generalization-v3.md`

- [ ] **Step 1: Run all unit tests and compile checks**

```powershell
python -m unittest discover -s tests -v
python -m compileall -q code
git diff --check
```

- [ ] **Step 2: Run benchmark18 and blind36 as development regressions**

Use the same `deepseek-v4-flash`, temperature 0 and thinking disabled. These results may guide
debugging but must never be relabeled as test results.

- [ ] **Step 3: Require zero execution failures on blind36**

If failures remain, diagnose by generic error category and return to the relevant TDD task.
Do not add ID/entity-specific rules.

### Task 8: New blind60 paired benchmark

**Files:**
- Create: `datasets_ready/blind_holdout_v3_2026-06-24/`
- Create: `outputs/blind_holdout_v3_2026-06-24/`
- Create: `BLIND_HOLDOUT_V3_REPORT_2026-06-24.md`

- [ ] **Step 1: Freeze unseen inputs**

Exclude every ID and table ID present in historical inputs and outputs. Draw 20 records per
dataset with a cryptographic random seed, preserve TabFact balance and CRT strata, and save
SHA-256 values before any model call.

- [ ] **Step 2: Run all myAgent groups without evaluation**

Do not inspect questions, predictions or accuracy while the three groups run.

- [ ] **Step 3: Run all MACT groups without evaluation**

Use the same model and sample files with MACT single-candidate, three-step settings.

- [ ] **Step 4: Unseal once and evaluate**

Compute per-dataset and overall accuracy, actual token use, failures, latency, Wilson intervals,
paired outcomes and exact McNemar p-value.

- [ ] **Step 5: Apply acceptance criteria honestly**

Mark V3 successful only if all design thresholds pass. If not, report the result and treat the
new holdout as consumed; never tune and rerun on it as a test set.
