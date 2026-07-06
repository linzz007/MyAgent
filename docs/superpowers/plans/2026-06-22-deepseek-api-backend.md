# DeepSeek API Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parameter-selectable model backend to myAgent and make DeepSeek's official API usable without local vLLM dependencies.

**Architecture:** `code/model_backends.py` owns provider resolution, argument registration, and construction of a `Callable[[str], str]`. Both experiment entry points delegate model setup to it and keep dataset/pipeline responsibilities unchanged.

**Tech Stack:** Python 3, argparse, OpenAI Python SDK, unittest

---

### Task 1: DeepSeek backend contract

**Files:**
- Create: `code/model_backends.py`
- Create: `tests/test_model_backends.py`

- [x] **Step 1: Write failing DeepSeek tests**

Add tests that inject a fake OpenAI client factory and assert the official base URL, environment key, selected model, messages, `thinking` body, temperature, token limit, response text, and clear missing-key error.

- [x] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_model_backends -v`

Expected: FAIL because `model_backends` does not exist.

- [x] **Step 3: Implement the minimal shared backend**

Implement `add_model_backend_args(parser)`, `resolve_model_provider(args)`, and `build_llm_fn(args, openai_client_factory=None)`. Keep provider-specific imports inside builder functions and use `extra_body={"thinking": {"type": args.thinking}}` for DeepSeek.

- [x] **Step 4: Verify backend tests pass**

Run: `python -m unittest tests.test_model_backends -v`

Expected: all backend tests PASS without a network request.

### Task 2: Wire both myAgent entry points

**Files:**
- Modify: `code/run_wtq_myagent.py`
- Modify: `code/tqa.py`
- Test: `tests/test_model_backends.py`

- [x] **Step 1: Add failing entry-point tests**

Run each script with `--help` in a subprocess and assert exit code 0 plus the presence of `--model_provider`, `--api_base`, `--api_key_env`, and `--thinking`.

- [x] **Step 2: Verify entry-point tests fail**

Run: `python -m unittest tests.test_model_backends.EntryPointTests -v`

Expected: FAIL because current imports require local model packages and the new flags do not exist.

- [x] **Step 3: Delegate model setup to the shared backend**

Remove eager OpenAI/vLLM/transformers/agents/llm imports from both scripts, import `add_model_backend_args` and `build_llm_fn`, and use them in `main` and parser construction. Preserve existing dataset and pipeline flags.

- [x] **Step 4: Verify entry points**

Run: `python code/run_wtq_myagent.py --help` and `python code/tqa.py --help`.

Expected: both exit 0 on the local API-only environment.

### Task 3: Regression verification and usage documentation

**Files:**
- Modify: `EXPERIMENT_READINESS.md`

- [x] **Step 1: Document secure DeepSeek commands**

Add PowerShell examples that set `DEEPSEEK_API_KEY` in the process environment and run `deepseek-v4-flash` with provider and thinking parameters. Do not include a real key.

- [x] **Step 2: Run the focused and full test suites**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [x] **Step 3: Compile changed Python files**

Run: `python -m py_compile code/model_backends.py code/run_wtq_myagent.py code/tqa.py`

Expected: exit code 0.

- [x] **Step 4: Review the diff**

Run: `git diff --check` and inspect only the files in this plan. Expected: no whitespace errors and no API key value in the diff.
