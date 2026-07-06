# Task-Aware Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict TabFact true/false and explicit CRT Yes/No execution paths while preserving WTQ and mixed-format CRT behavior.

**Architecture:** Carry an answer mode in session state, derive it from the task and question, and branch after shared routing/compression into a closed-label classifier implemented by FinalAnswerAgent.

**Tech Stack:** Python 3, pandas, unittest, existing myAgent pipeline

---

### Task 1: Closed-label classifier

**Files:**
- Modify: `code/my_agents.py`
- Modify: `tests/test_myagent_pipeline.py`

- [x] **Step 1: Write failing tests**

Add tests constructing `TQASessionState(..., answer_mode="true_false")` and `answer_mode="yes_no"`. Assert JSON and verbose labels normalize to the canonical label, invalid output raises, and no planner/critic fake responses are consumed.

- [x] **Step 2: Confirm RED**

Run: `python -m unittest discover -s tests -p "test_myagent_pipeline.py" -v`

Expected: state does not accept `answer_mode` and no classification branch exists.

- [x] **Step 3: Implement minimal behavior**

Add the state field, `FinalAnswerAgent.classify`, strict label parsing, and a post-compression branch in `TableQAPipeline.run`.

- [x] **Step 4: Confirm GREEN**

Run the same focused test command and require all pipeline tests to pass.

### Task 2: Entry-point task mapping

**Files:**
- Modify: `code/tqa.py`
- Modify: `tests/test_myagent_pipeline.py`

- [x] **Step 1: Write failing mapping tests**

Test that `answer_mode_for_task` maps `scitab` to `true_false`, while `answer_mode_for_sample` maps only explicit CRT Yes/No instructions to `yes_no` and leaves numeric/text CRT and WTQ records in general mode.

- [x] **Step 2: Confirm RED**

Run the focused pipeline tests and require an import failure for the missing helper.

- [x] **Step 3: Implement and wire mapping**

Set `answer_mode=answer_mode_for_sample(args.task, question)` for each TSV, DataBench, and JSONL record.

- [x] **Step 4: Run full tests and real smoke**

Run all unit tests, then rerun DeepSeek TabFact and mixed CRT records. Require non-empty canonical labels for closed-label records and no forced classification for numeric/text CRT records.
