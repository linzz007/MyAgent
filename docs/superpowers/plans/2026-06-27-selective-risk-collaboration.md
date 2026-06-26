# Selective Risk Collaboration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add risk-driven selective collaboration so myAgent can spend more tokens on high-risk WTQ/TabFact/CRT samples, preserve a measurable token advantage over MACT, and report accuracy by risk stratum.

**Architecture:** New focused modules compute pre/post risk, build dataset-aware evidence packs, compare candidate answers, and enforce MACT-relative budgets. `TableQAPipeline` remains the orchestration entry point and keeps the legacy path available behind a mode flag while the new selective path reuses the existing planner, calculator, critic, finalizer, and multi-view components.

**Tech Stack:** Python 3.13, pandas, unittest, JSONL, existing DeepSeek-compatible backend, existing myAgent evaluator

---

## File Structure

- Create: `code/risk_control.py`
  - Owns formulas, thresholds, budget tiers, and serializable risk records.
- Create: `tests/test_risk_control.py`
  - Verifies formula weights, threshold labels, hard escalation, and token budget accounting.
- Create: `code/evidence_builder.py`
  - Builds gold-free evidence packs from question, DataFrame, schema profile, dataset name, and answer contract.
- Create: `tests/test_evidence_builder.py`
  - Verifies WTQ full-table candidate retrieval, TabFact logic signals, CRT contract signals, and no gold leakage.
- Create: `code/selective_collaboration.py`
  - Defines candidate answers, deterministic agreement, evidence-gap scoring, and a bounded thinking fallback wrapper.
- Create: `tests/test_selective_collaboration.py`
  - Verifies answer similarity, invalid-candidate handling, disagreement escalation, and fallback JSON parsing.
- Modify: `code/my_agents.py`
  - Adds selective mode state fields and pipeline branch while preserving the existing legacy flow.
- Modify: `tests/test_myagent_pipeline.py`
  - Adds selective integration tests and keeps existing pipeline tests green.
- Modify: `code/tqa.py`
  - Adds CLI flags and JSONL observability for risk, evidence, candidates, agreement, and budget.
- Modify: `code/run_wtq_myagent.py`
  - Adds the same mode and observability fields for the WTQ runner.
- Modify: `code/evaluate_results.py`
  - Adds risk-stratified accuracy/token summaries.
- Modify: `tests/test_evaluate_results.py`
  - Verifies risk summaries without changing existing dataset correctness logic.
- Modify: `code/compare_blind_results.py`
  - Updates acceptance from `0.40` token ratio to `0.75` and adds risk-stratum deltas.
- Modify: `tests/test_compare_blind_results.py`
  - Verifies the new acceptance fields.
- Create: `code/calibrate_risk_policy.py`
  - Reads development outputs and selects stable thresholds under the 0.75 MACT token constraint.
- Create: `tests/test_calibrate_risk_policy.py`
  - Verifies deterministic threshold selection and no blind-set dependency.
- Modify: `docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md`
  - Check off steps as they pass.

## Current Inputs

- Design spec: `docs/superpowers/specs/2026-06-24-selective-risk-collaboration-design.md`
- Existing V4 report: `BLIND_HOLDOUT_V4_REPORT_2026-06-24.md`
- Existing V4 machine summary: `outputs/blind_holdout_v4_2026-06-24/summary.json`
- Frozen local datasets for development smoke tests:
  - `datasets_ready/blind_holdout_v4_2026-06-24/wtq.jsonl`
  - `datasets_ready/blind_holdout_v4_2026-06-24/tabfact.jsonl`
  - `datasets_ready/blind_holdout_v4_2026-06-24/crt.jsonl`
- Full adapted datasets for later server experiments:
  - `datasets_ready/full/wtq_unseen.jsonl`
  - `datasets_ready/full/tabfact_test.jsonl`
  - `datasets_ready/full/crt.jsonl`

## Task 1: Risk Control Core

**Files:**
- Create: `code/risk_control.py`
- Create: `tests/test_risk_control.py`

- [x] **Step 1: Write failing formula tests**

Add this test file:

```python
import unittest

from risk_control import (
    BudgetController,
    BudgetPolicy,
    RiskAssessment,
    RiskPolicy,
    RiskProfiler,
    weighted_score,
)


class RiskControlTests(unittest.TestCase):
    def test_weighted_score_clamps_each_input(self):
        result = weighted_score({"a": 1.5, "b": -1.0}, {"a": 0.25, "b": 0.75})
        self.assertEqual(result, 0.25)

    def test_pre_risk_formula_uses_design_weights(self):
        profiler = RiskProfiler(RiskPolicy())
        assessment = profiler.assess_pre(
            semantic_complexity=0.2,
            structure_signals={"coverage": 1.0, "dispersion": 0.5, "type": 0.0},
            ambiguity_signals={
                "entity": 0.0,
                "column": 1.0,
                "temporal": 0.0,
                "reference": 0.5,
            },
            gap_signals={
                "entity": 0.2,
                "column": 0.4,
                "missing": 0.0,
                "stability": 1.0,
            },
            operation_signals={
                "steps": 0.5,
                "dependency": 1.0,
                "contract": 0.5,
                "unit": 0.0,
                "logic": 0.5,
            },
            hard_triggers=[],
        )
        self.assertAlmostEqual(assessment.difficulty, 0.4075, places=4)
        self.assertAlmostEqual(assessment.ambiguity, 0.35, places=4)
        self.assertAlmostEqual(assessment.evidence_gap, 0.37, places=4)
        self.assertAlmostEqual(assessment.operation_risk, 0.55, places=4)
        self.assertEqual(assessment.level, "medium")

    def test_thresholds_and_hard_triggers(self):
        profiler = RiskProfiler(RiskPolicy())
        low = RiskAssessment(pre_risk=0.24, hard_triggers=())
        medium = RiskAssessment(pre_risk=0.25, hard_triggers=())
        high = RiskAssessment(pre_risk=0.55, hard_triggers=())
        forced = RiskAssessment(pre_risk=0.10, hard_triggers=("contract_failure",))
        self.assertEqual(profiler.level_for(low), "light")
        self.assertEqual(profiler.level_for(medium), "medium")
        self.assertEqual(profiler.level_for(high), "high")
        self.assertEqual(profiler.level_for(forced), "fallback")

    def test_post_risk_formula_and_fallback_boundary(self):
        profiler = RiskProfiler(RiskPolicy())
        post = profiler.assess_post(
            pre_risk=0.45,
            candidate_disagreement=0.5,
            verification_gap=0.5,
            execution_failure=0.0,
            contract_failure=0.0,
            unit_failure=0.0,
            normalization_failure=0.0,
        )
        self.assertAlmostEqual(post.post_risk, 0.70, places=4)
        self.assertTrue(post.requires_fallback)

    def test_budget_controller_tracks_mact_relative_caps(self):
        controller = BudgetController(BudgetPolicy(mact_avg_tokens=8867.0))
        self.assertEqual(controller.cap_for("light"), 2216)
        self.assertEqual(controller.cap_for("medium"), 4876)
        self.assertEqual(controller.cap_for("high"), 7537)
        self.assertEqual(controller.cap_for("fallback"), 9754)
        controller.record("light", 1000)
        controller.record("high", 7000)
        self.assertAlmostEqual(controller.avg_tokens(), 4000.0)
        self.assertTrue(controller.within_average_limit())


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```powershell
python -m unittest tests.test_risk_control -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'risk_control'`.

- [x] **Step 3: Implement `code/risk_control.py`**

Create these dataclasses and functions:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def weighted_score(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return sum(clamp01(values.get(name, 0.0)) * weight for name, weight in weights.items())


@dataclass(frozen=True)
class RiskPolicy:
    light_threshold: float = 0.25
    high_threshold: float = 0.55
    fallback_threshold: float = 0.70
    structure_weights: Mapping[str, float] = field(
        default_factory=lambda: {"coverage": 0.50, "dispersion": 0.30, "type": 0.20}
    )
    ambiguity_weights: Mapping[str, float] = field(
        default_factory=lambda: {"entity": 0.35, "column": 0.25, "temporal": 0.20, "reference": 0.20}
    )
    gap_weights: Mapping[str, float] = field(
        default_factory=lambda: {"entity": 0.35, "column": 0.25, "missing": 0.20, "stability": 0.20}
    )
    operation_weights: Mapping[str, float] = field(
        default_factory=lambda: {"steps": 0.25, "dependency": 0.25, "contract": 0.20, "unit": 0.15, "logic": 0.15}
    )


@dataclass
class RiskAssessment:
    difficulty: float = 0.0
    ambiguity: float = 0.0
    evidence_gap: float = 0.0
    operation_risk: float = 0.0
    pre_risk: float = 0.0
    post_risk: float = 0.0
    level: str = "light"
    hard_triggers: Tuple[str, ...] = ()
    requires_fallback: bool = False
    feature_evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetPolicy:
    mact_avg_tokens: float = 8867.0
    light_ratio: float = 0.25
    medium_ratio: float = 0.55
    high_ratio: float = 0.85
    fallback_ratio: float = 1.10
    average_ratio: float = 0.75


class BudgetController:
    def __init__(self, policy: BudgetPolicy | None = None) -> None:
        self.policy = policy or BudgetPolicy()
        self.records: List[Dict[str, object]] = []

    def cap_for(self, level: str) -> int:
        ratios = {
            "light": self.policy.light_ratio,
            "medium": self.policy.medium_ratio,
            "high": self.policy.high_ratio,
            "fallback": self.policy.fallback_ratio,
        }
        return int(self.policy.mact_avg_tokens * ratios.get(level, self.policy.medium_ratio))

    def record(self, level: str, tokens: int) -> None:
        self.records.append({"level": level, "tokens": int(max(0, tokens)), "cap": self.cap_for(level)})

    def avg_tokens(self) -> float:
        if not self.records:
            return 0.0
        return sum(int(row["tokens"]) for row in self.records) / len(self.records)

    def within_average_limit(self) -> bool:
        return self.avg_tokens() <= self.policy.mact_avg_tokens * self.policy.average_ratio

    def to_dict(self) -> Dict[str, object]:
        return {"policy": asdict(self.policy), "records": list(self.records), "avg_tokens": self.avg_tokens()}


class RiskProfiler:
    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy or RiskPolicy()

    def level_for(self, assessment: RiskAssessment) -> str:
        if assessment.hard_triggers:
            return "fallback"
        if assessment.pre_risk < self.policy.light_threshold:
            return "light"
        if assessment.pre_risk < self.policy.high_threshold:
            return "medium"
        return "high"

    def assess_pre(
        self,
        semantic_complexity: float,
        structure_signals: Mapping[str, float],
        ambiguity_signals: Mapping[str, float],
        gap_signals: Mapping[str, float],
        operation_signals: Mapping[str, float],
        hard_triggers: Iterable[str],
    ) -> RiskAssessment:
        structure = weighted_score(structure_signals, self.policy.structure_weights)
        difficulty = 0.45 * clamp01(semantic_complexity) + 0.55 * structure
        ambiguity = weighted_score(ambiguity_signals, self.policy.ambiguity_weights)
        gap = weighted_score(gap_signals, self.policy.gap_weights)
        operation = weighted_score(operation_signals, self.policy.operation_weights)
        pre = 0.20 * difficulty + 0.30 * ambiguity + 0.30 * gap + 0.20 * operation
        assessment = RiskAssessment(
            difficulty=clamp01(difficulty),
            ambiguity=clamp01(ambiguity),
            evidence_gap=clamp01(gap),
            operation_risk=clamp01(operation),
            pre_risk=clamp01(pre),
            hard_triggers=tuple(hard_triggers),
            feature_evidence={
                "structure": dict(structure_signals),
                "ambiguity": dict(ambiguity_signals),
                "gap": dict(gap_signals),
                "operation": dict(operation_signals),
            },
        )
        assessment.level = self.level_for(assessment)
        return assessment

    def assess_post(
        self,
        pre_risk: float,
        candidate_disagreement: float,
        verification_gap: float,
        execution_failure: float,
        contract_failure: float,
        unit_failure: float,
        normalization_failure: float,
    ) -> RiskAssessment:
        hard_failure = max(
            clamp01(execution_failure),
            clamp01(contract_failure),
            clamp01(unit_failure),
            clamp01(normalization_failure),
        )
        post = min(
            1.0,
            clamp01(pre_risk)
            + 0.30 * clamp01(candidate_disagreement)
            + 0.20 * clamp01(verification_gap)
            + 0.30 * hard_failure,
        )
        assessment = RiskAssessment(pre_risk=clamp01(pre_risk), post_risk=post)
        assessment.requires_fallback = post >= self.policy.fallback_threshold or hard_failure >= 1.0
        assessment.level = "fallback" if assessment.requires_fallback else self.level_for(assessment)
        return assessment
```

- [x] **Step 4: Run focused test and commit Task 1**

Run:

```powershell
python -m unittest tests.test_risk_control -v
```

Expected: `OK`.

Commit:

```powershell
git add code/risk_control.py tests/test_risk_control.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: add selective risk control"
```

## Task 2: Evidence Builder

**Files:**
- Create: `code/evidence_builder.py`
- Create: `tests/test_evidence_builder.py`

- [x] **Step 1: Write failing evidence tests**

Add this test file:

```python
import unittest

import pandas as pd

from answer_contracts import infer_answer_contract
from evidence_builder import EvidenceBuilder


class EvidenceBuilderTests(unittest.TestCase):
    def test_wtq_entity_index_finds_rows_beyond_preview(self):
        df = pd.DataFrame(
            {
                "Horse": ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Falcon"],
                "Result": ["lost", "lost", "lost", "lost", "lost", "won"],
            }
        )
        contract = infer_answer_contract("wtq", "What was the result for Falcon?")
        pack = EvidenceBuilder(max_candidates=10).build(
            question="What was the result for Falcon?",
            df=df,
            schema={"column_profiles": []},
            dataset_name="wtq",
            answer_contract=contract,
        )
        self.assertIn(5, [row.row_index for row in pack.candidate_rows])
        self.assertEqual(pack.gap_signals["entity"], 0.0)

    def test_unmatched_entity_and_ambiguous_column_raise_risk_signals(self):
        df = pd.DataFrame({"City": ["Paris"], "City Rank": [1], "Score": [9]})
        contract = infer_answer_contract("wtq", "What is the ranking for Berlin?")
        pack = EvidenceBuilder().build(
            question="What is the ranking for Berlin?",
            df=df,
            schema={"column_profiles": []},
            dataset_name="wtq",
            answer_contract=contract,
        )
        self.assertGreater(pack.gap_signals["entity"], 0.0)
        self.assertGreaterEqual(pack.ambiguity_signals["column"], 0.25)

    def test_tabfact_statement_sets_logic_and_count_signals(self):
        df = pd.DataFrame({"Nation": ["A", "B", "C"], "Medals": [3, 1, 3]})
        contract = infer_answer_contract("tabfact", "Two nations have 3 medals and no nation has 4.")
        pack = EvidenceBuilder().build(
            question="Two nations have 3 medals and no nation has 4.",
            df=df,
            schema={"column_profiles": []},
            dataset_name="tabfact",
            answer_contract=contract,
        )
        self.assertGreaterEqual(pack.operation_signals["logic"], 0.5)
        self.assertGreaterEqual(pack.operation_signals["steps"], 0.5)

    def test_crt_tuple_and_unit_contract_raise_contract_signals(self):
        df = pd.DataFrame({"Method": ["decision", "finish"], "Wins": [3, 12]})
        contract = infer_answer_contract("crt", "How many wins by decision and by finish?")
        pack = EvidenceBuilder().build(
            question="How many wins by decision and by finish?",
            df=df,
            schema={"column_profiles": []},
            dataset_name="crt",
            answer_contract=contract,
        )
        self.assertGreaterEqual(pack.operation_signals["contract"], 0.5)
        self.assertIn("Method", pack.candidate_columns)

    def test_pack_serialization_excludes_gold_answer(self):
        df = pd.DataFrame({"Name": ["A"], "Value": [1]})
        contract = infer_answer_contract("wtq", "What is the value for A?")
        pack = EvidenceBuilder().build(
            question="What is the value for A?",
            df=df,
            schema={"gold_answer": "1", "column_profiles": []},
            dataset_name="wtq",
            answer_contract=contract,
        )
        payload = pack.to_dict()
        self.assertNotIn("gold_answer", str(payload).lower())


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run focused test and confirm RED**

Run:

```powershell
python -m unittest tests.test_evidence_builder -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evidence_builder'`.

- [x] **Step 3: Implement `code/evidence_builder.py`**

Create:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Dict, List, Mapping, Sequence

import pandas as pd


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class EvidenceRef:
    row_index: int
    column: str
    value: str
    match_type: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EvidencePack:
    dataset_name: str
    candidate_rows: List[EvidenceRef] = field(default_factory=list)
    candidate_columns: List[str] = field(default_factory=list)
    entity_matches: Dict[str, List[EvidenceRef]] = field(default_factory=dict)
    operation_hints: List[str] = field(default_factory=list)
    missing_or_ambiguous_items: List[str] = field(default_factory=list)
    structure_signals: Dict[str, float] = field(default_factory=dict)
    ambiguity_signals: Dict[str, float] = field(default_factory=dict)
    gap_signals: Dict[str, float] = field(default_factory=dict)
    operation_signals: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "candidate_rows": [row.to_dict() for row in self.candidate_rows],
            "candidate_columns": list(self.candidate_columns),
            "entity_matches": {
                key: [ref.to_dict() for ref in refs] for key, refs in self.entity_matches.items()
            },
            "operation_hints": list(self.operation_hints),
            "missing_or_ambiguous_items": list(self.missing_or_ambiguous_items),
            "structure_signals": dict(self.structure_signals),
            "ambiguity_signals": dict(self.ambiguity_signals),
            "gap_signals": dict(self.gap_signals),
            "operation_signals": dict(self.operation_signals),
            "provenance": dict(self.provenance),
        }
```

Implement `EvidenceBuilder.build(...)` using these rules:

- Normalize question tokens with `TOKEN_RE`.
- Build row refs by scanning the full DataFrame, not only preview rows.
- Match cell text by exact lowercase containment and token containment.
- Candidate columns include column names whose normalized text overlaps question tokens.
- `structure_signals["coverage"] = min(1.0, log2(max(1, matched_cells) + 1) / 6.0)`.
- `structure_signals["dispersion"] = 1.0` when candidate rows span more than half the table, `0.0` for one row, and proportional between.
- `structure_signals["type"] = 0.5` when candidate columns include both numeric and text columns.
- `ambiguity_signals["entity"] = 1.0` when no cell text matches any non-stopword question token longer than 2.
- `ambiguity_signals["column"] = min(1.0, max(0, len(candidate_columns) - 1) / 4.0)`.
- `gap_signals["entity"] = 1.0` when a titlecase/alphanumeric token longer than 3 has no evidence ref.
- `gap_signals["missing"]` counts empty, `nan`, `none`, `n/a`, `tba`, and `unknown` values in candidate rows.
- `operation_signals` uses question keywords:
  - count/number/how many -> `steps >= 0.5`
  - maximum/minimum/most/least/highest/lowest/first/last/before/after -> `dependency = 1.0`
  - and/or/not/no/only/either/both -> `logic >= 0.5`
  - average/percent/rate/ratio/unit/miles/kg/% -> `unit >= 0.5`
  - list/which/how many by/tuple contract -> `contract >= 0.5`
- Exclude every schema key containing `gold`, `answer`, `label`, or `correct` from provenance.

- [x] **Step 4: Run focused test and commit Task 2**

Run:

```powershell
python -m unittest tests.test_evidence_builder -v
```

Expected: `OK`.

Commit:

```powershell
git add code/evidence_builder.py tests/test_evidence_builder.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: build selective evidence packs"
```

## Task 3: Candidate Agreement and Thinking Fallback

**Files:**
- Create: `code/selective_collaboration.py`
- Create: `tests/test_selective_collaboration.py`

- [x] **Step 1: Write failing collaboration tests**

Add this test file:

```python
import json
import unittest

from answer_contracts import infer_answer_contract
from selective_collaboration import (
    AgreementJudge,
    CandidateAnswer,
    ThinkingSolver,
    answer_similarity,
)


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, prompt, temperature=0.0):
        self.calls.append(prompt)
        return self.text


class SelectiveCollaborationTests(unittest.TestCase):
    def test_list_similarity_is_order_insensitive_for_wtq(self):
        contract = infer_answer_contract("wtq", "Which teams qualified?")
        left = ["A", "B"]
        right = ["b", "a"]
        self.assertEqual(answer_similarity(left, right, contract), 1.0)

    def test_tuple_similarity_preserves_position_for_crt(self):
        contract = infer_answer_contract("crt", "How many wins by decision and finish?")
        self.assertEqual(answer_similarity(["3", "12"], ["3", "12"], contract), 1.0)
        self.assertEqual(answer_similarity(["3", "12"], ["12", "3"], contract), 0.0)

    def test_numeric_similarity_accepts_small_format_difference(self):
        contract = infer_answer_contract("wtq", "What is the average score?")
        self.assertEqual(answer_similarity("3.0", "3", contract), 1.0)

    def test_invalid_candidate_loses_to_valid_candidate(self):
        contract = infer_answer_contract("tabfact", "A has 3 wins.")
        judge = AgreementJudge(contract)
        valid = CandidateAnswer(name="code", raw_answer="true", normalized_answer="true", is_valid=True)
        invalid = CandidateAnswer(name="react", raw_answer="", normalized_answer="", is_valid=False, failure="exec")
        decision = judge.decide([invalid, valid])
        self.assertEqual(decision.selected.name, "code")
        self.assertFalse(decision.requires_fallback)

    def test_conflicting_valid_candidates_require_fallback(self):
        contract = infer_answer_contract("tabfact", "A has 3 wins.")
        judge = AgreementJudge(contract)
        left = CandidateAnswer(name="code", raw_answer="true", normalized_answer="true", is_valid=True)
        right = CandidateAnswer(name="react", raw_answer="false", normalized_answer="false", is_valid=True)
        decision = judge.decide([left, right])
        self.assertTrue(decision.requires_fallback)
        self.assertGreater(decision.disagreement, 0.0)

    def test_thinking_solver_parses_json_answer(self):
        llm = FakeLLM(json.dumps({"answer": "true", "confidence": 0.7, "reasoning_summary": "checked rows"}))
        solver = ThinkingSolver(llm)
        candidate = solver.solve(
            question="A has 3 wins.",
            evidence={"candidate_rows": []},
            candidates=[],
            answer_contract=infer_answer_contract("tabfact", "A has 3 wins."),
        )
        self.assertEqual(candidate.normalized_answer, "true")
        self.assertTrue(candidate.is_valid)
        self.assertIn("Return JSON", llm.calls[0])


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run focused test and confirm RED**

Run:

```powershell
python -m unittest tests.test_selective_collaboration -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'selective_collaboration'`.

- [x] **Step 3: Implement `code/selective_collaboration.py`**

Create:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class CandidateAnswer:
    name: str
    raw_answer: Any
    normalized_answer: Any
    is_valid: bool
    reasoning_summary: str = ""
    evidence_refs: List[Dict[str, object]] = field(default_factory=list)
    executable_program: str = ""
    execution_result: Any = None
    confidence: float = 0.0
    token_usage: int = 0
    failure: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class AgreementDecision:
    selected: CandidateAnswer | None
    candidates: List[CandidateAnswer]
    agreement: bool
    requires_fallback: bool
    disagreement: float
    verification_gap: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["selected"] = self.selected.to_dict() if self.selected else None
        data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data
```

Implement:

- `normalize_atom(value)` lowercases strings, removes outer punctuation, and keeps numbers comparable.
- `to_sequence(value)` converts list/tuple/set or comma-separated strings to atomic strings.
- `answer_similarity(left, right, answer_contract)`:
  - returns `1.0` for equal normalized scalar strings.
  - returns `1.0` for numeric values with absolute difference under `1e-6`.
  - returns Jaccard for WTQ-style lists.
  - returns positional equality for tuple contracts.
  - returns `0.0` when either side is empty and the other is not.
- `verification_gap(candidate, evidence_pack)`:
  - returns `1.0` for invalid candidates.
  - returns `0.5` when answer atoms do not appear in evidence text and no executable result exists.
  - returns `0.0` when the candidate has execution result or answer atoms are grounded.
- `AgreementJudge.decide(candidates)`:
  - filters valid candidates.
  - selects the only valid candidate when exactly one exists.
  - accepts when top two valid candidates have similarity `1.0`.
  - requires fallback when top two valid candidates are both valid and disagree.
- `ThinkingSolver.solve(...)`:
  - sends a compact prompt containing question, evidence dict, candidate summaries, answer contract summary, and the instruction `Return JSON`.
  - parses JSON object with `answer`, `confidence`, and `reasoning_summary`.
  - returns invalid `CandidateAnswer` with `failure="thinking_parse_error"` when parsing fails.

- [x] **Step 4: Run focused test and commit Task 3**

Run:

```powershell
python -m unittest tests.test_selective_collaboration -v
```

Expected: `OK`.

Commit:

```powershell
git add code/selective_collaboration.py tests/test_selective_collaboration.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: compare selective candidates"
```

## Task 4: Pipeline Integration

**Files:**
- Modify: `code/my_agents.py`
- Modify: `tests/test_myagent_pipeline.py`

- [x] **Step 1: Write failing selective pipeline tests**

Append these tests to `tests/test_myagent_pipeline.py` inside the existing pipeline test class or a new `SelectivePipelineTests` class that reuses `FakePipelineLLM`:

```python
def test_selective_mode_records_risk_and_budget(self):
    fake = FakePipelineLLM()
    pipeline, tracker = self._pipeline(fake)
    pipeline.enable_selective_collaboration = True
    state = TQASessionState(
        question="What is the value for Alpha?",
        table=pd.DataFrame({"Name": ["Alpha"], "Value": [7]}),
        dataset_name="wtq",
    )
    result = pipeline.run(state)
    self.assertIsNotNone(result.risk_assessment)
    self.assertIn(result.risk_level, {"light", "medium", "high", "fallback"})
    self.assertIsNotNone(result.evidence_pack)
    self.assertIn("avg_tokens", result.budget_state)

def test_selective_high_risk_runs_candidate_judge(self):
    fake = FakePipelineLLM()
    pipeline, tracker = self._pipeline(fake, enable_multi_view_validation=True)
    pipeline.enable_selective_collaboration = True
    state = TQASessionState(
        question="Which teams had the highest score and no losses after 2010?",
        table=pd.DataFrame({"Team": ["A", "B"], "Score": [9, 9], "Losses": [0, 1], "Year": [2011, 2012]}),
        dataset_name="wtq",
    )
    result = pipeline.run(state)
    self.assertIsNotNone(result.agreement_decision)
    self.assertIsInstance(result.candidate_answers, list)

def test_legacy_mode_does_not_populate_selective_fields(self):
    fake = FakePipelineLLM()
    pipeline, tracker = self._pipeline(fake)
    pipeline.enable_selective_collaboration = False
    state = TQASessionState(
        question="What is the value for Alpha?",
        table=pd.DataFrame({"Name": ["Alpha"], "Value": [7]}),
        dataset_name="wtq",
    )
    result = pipeline.run(state)
    self.assertIsNone(result.risk_assessment)
    self.assertIsNone(result.evidence_pack)
```

- [x] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
python -m unittest tests.test_myagent_pipeline.SelectivePipelineTests -v
```

Expected: FAIL because `enable_selective_collaboration`, `risk_assessment`, `evidence_pack`, `candidate_answers`, `agreement_decision`, and `budget_state` are missing or unused.

- [x] **Step 3: Add state fields and constructor arguments**

Modify `TQASessionState.__init__` to initialize:

```python
self.risk_assessment = None
self.post_risk_assessment = None
self.risk_level = ""
self.evidence_pack = None
self.candidate_answers = []
self.agreement_decision = None
self.budget_state = {}
```

Modify `TableQAPipeline.__init__` signature to accept:

```python
enable_selective_collaboration: bool = False,
mact_avg_tokens: float = 8867.0,
risk_profiler=None,
evidence_builder=None,
agreement_judge_factory=None,
thinking_solver_factory=None,
```

Store:

```python
self.enable_selective_collaboration = enable_selective_collaboration
self.mact_avg_tokens = mact_avg_tokens
self.risk_profiler = risk_profiler or RiskProfiler()
self.evidence_builder = evidence_builder or EvidenceBuilder()
self.agreement_judge_factory = agreement_judge_factory or (lambda contract: AgreementJudge(contract))
self.thinking_solver_factory = thinking_solver_factory
self.budget_controller = BudgetController(BudgetPolicy(mact_avg_tokens=mact_avg_tokens))
```

- [x] **Step 4: Split legacy run into a helper**

Rename the current body of `TableQAPipeline.run` to `_run_legacy(self, state)` without behavior changes.

Add:

```python
def run(self, state: TQASessionState) -> TQASessionState:
    if not self.enable_selective_collaboration:
        return self._run_legacy(state)
    return self._run_selective(state)
```

- [x] **Step 5: Implement `_run_selective` as a conservative wrapper**

Implement `_run_selective` using this order:

1. Run `RouterAgent` and dataset contract forcing exactly as legacy does.
2. Run `TableCompressor`.
3. Build `state.evidence_pack` from the original DataFrame and existing schema.
4. Use router score plus evidence signals to call `RiskProfiler.assess_pre`.
5. Save `state.risk_assessment` and `state.risk_level`.
6. For `light`, call the existing simple path first; if it returns invalid/empty, call `_run_legacy`.
7. For `medium`, call `_run_legacy`, then wrap the result as one `CandidateAnswer`.
8. For `high`, call `_run_legacy`, then run existing `MultiViewValidator` when enabled, and compare the main and alternative candidate when an alternative answer exists.
9. If `AgreementDecision.requires_fallback` and `thinking_solver_factory` is configured, call `ThinkingSolver`.
10. Record `BudgetController.record(state.risk_level, state.llm_tracker.total_tokens)` and serialize to `state.budget_state`.
11. Preserve all final answer normalization and validation already performed by legacy helpers.

Do not add ID-specific, table-specific, gold-specific, or sample-specific branches.

- [x] **Step 6: Run focused tests and commit Task 4**

Run:

```powershell
python -m unittest tests.test_myagent_pipeline -v
```

Expected: `OK`.

Commit:

```powershell
git add code/my_agents.py tests/test_myagent_pipeline.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: wire selective collaboration pipeline"
```

## Task 5: CLI and JSONL Observability

**Files:**
- Modify: `code/tqa.py`
- Modify: `code/run_wtq_myagent.py`
- Modify: `tests/test_task_modes.py`

- [ ] **Step 1: Write failing observability tests**

Add to `tests/test_task_modes.py`:

```python
def test_state_observability_includes_selective_fields(self):
    from tqa import _state_observability
    from my_agents import TQASessionState

    state = TQASessionState(question="q", table=None, dataset_name="wtq")
    state.risk_level = "high"
    state.risk_assessment = {"pre_risk": 0.8}
    state.post_risk_assessment = {"post_risk": 0.9}
    state.evidence_pack = {"candidate_rows": []}
    state.candidate_answers = [{"name": "code"}]
    state.agreement_decision = {"requires_fallback": True}
    state.budget_state = {"avg_tokens": 1000}
    payload = _state_observability(state)
    self.assertEqual(payload["risk_level"], "high")
    self.assertEqual(payload["risk_assessment"]["pre_risk"], 0.8)
    self.assertIn("budget_state", payload)
```

- [ ] **Step 2: Run focused test and confirm RED**

Run:

```powershell
python -m unittest tests.test_task_modes -v
```

Expected: FAIL because selective fields are not emitted.

- [ ] **Step 3: Add CLI flags**

In both `code/tqa.py` and `code/run_wtq_myagent.py`, add:

```python
parser.add_argument(
    "--collaboration_mode",
    choices=["legacy", "selective", "calibration"],
    default="selective",
)
parser.add_argument("--mact_avg_tokens", type=float, default=8867.0)
parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="Run only the first N records after loading; 0 means all records.",
)
```

Pass into `TableQAPipeline`:

```python
enable_selective_collaboration=args.collaboration_mode != "legacy",
mact_avg_tokens=args.mact_avg_tokens,
```

In `code/tqa.py`, extend task choices so TabFact can be invoked directly:

```python
choices=["wtq", "crt", "tat", "scitab", "tabfact", "databench"]
```

After loading `table_dataset`, apply the limit:

```python
if getattr(args, "limit", 0):
    table_dataset = table_dataset[: args.limit]
```

- [ ] **Step 4: Emit selective observability**

In `_state_observability`, add serializable fields:

```python
"risk_level": state.risk_level,
"risk_assessment": _to_serializable(state.risk_assessment),
"post_risk_assessment": _to_serializable(state.post_risk_assessment),
"evidence_pack": _to_serializable(state.evidence_pack),
"candidate_answers": _to_serializable(state.candidate_answers),
"agreement_decision": _to_serializable(state.agreement_decision),
"budget_state": _to_serializable(state.budget_state),
```

Implement `_to_serializable(value)` if no helper already exists:

```python
def _to_serializable(value):
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    return value
```

- [ ] **Step 5: Run focused tests and commit Task 5**

Run:

```powershell
python -m unittest tests.test_task_modes -v
```

Expected: `OK`.

Commit:

```powershell
git add code/tqa.py code/run_wtq_myagent.py tests/test_task_modes.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: expose selective collaboration mode"
```

## Task 6: Evaluation and Acceptance Metrics

**Files:**
- Modify: `code/evaluate_results.py`
- Modify: `tests/test_evaluate_results.py`
- Modify: `code/compare_blind_results.py`
- Modify: `tests/test_compare_blind_results.py`

- [ ] **Step 1: Write failing evaluator tests**

Add to `tests/test_evaluate_results.py`:

```python
def test_summary_reports_risk_strata_accuracy_and_tokens(self):
    rows = [
        {
            "source_dataset": "wtq",
            "prediction": "a",
            "answer": "a",
            "observability": {"risk_level": "light"},
            "api_metrics": {"total_tokens": 100},
        },
        {
            "source_dataset": "wtq",
            "prediction": "b",
            "answer": "c",
            "observability": {"risk_level": "high"},
            "api_metrics": {"total_tokens": 300},
        },
    ]
    summary, evaluated = summarize_rows(rows)
    self.assertEqual(summary["risk_strata"]["light"]["accuracy"], 1.0)
    self.assertEqual(summary["risk_strata"]["high"]["accuracy"], 0.0)
    self.assertEqual(summary["risk_strata"]["high"]["avg_total_tokens"], 300)
```

Add to `tests/test_compare_blind_results.py`:

```python
def test_acceptance_uses_075_token_ratio_and_risk_deltas(self):
    my_rows = [
        {"id": "1", "prediction": "a", "answer": "a", "observability": {"risk_level": "light"}, "api_metrics": {"total_tokens": 70}},
        {"id": "2", "prediction": "b", "answer": "c", "observability": {"risk_level": "high"}, "api_metrics": {"total_tokens": 80}},
    ]
    mact_rows = [
        {"id": "1", "pred_answer": "a", "answer": "a", "observability": {"risk_level": "light"}, "api_metrics": {"total_tokens": 100}},
        {"id": "2", "pred_answer": "c", "answer": "c", "observability": {"risk_level": "high"}, "api_metrics": {"total_tokens": 100}},
    ]
    result = compare_dataset_rows(my_rows, mact_rows)
    self.assertIn("risk_strata", result["myagent"])
    self.assertEqual(result["myagent"]["avg_total_tokens"], 75)
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
python -m unittest tests.test_evaluate_results tests.test_compare_blind_results -v
```

Expected: FAIL because risk-stratum summaries and 0.75 acceptance fields are missing.

- [ ] **Step 3: Add risk-stratum summaries**

In `summarize_rows`, group evaluated rows by:

```python
risk_level = row.get("observability", {}).get("risk_level") or row.get("risk_level") or "unknown"
```

For each stratum, report:

```python
{
    "count": count,
    "correct": correct,
    "accuracy": correct / count,
    "avg_total_tokens": sum(tokens) / count,
    "failure_count": failure_count,
}
```

- [ ] **Step 4: Update blind comparison acceptance**

In `code/compare_blind_results.py`, replace the old `token_ratio_at_most_0_40` field with:

```python
"token_ratio_at_most_0_75": token_ratio is not None and token_ratio <= 0.75,
```

Add:

```python
"risk_accuracy_deltas": risk_accuracy_deltas,
"acceptance_selective_risk_collaboration": (
    overall_accuracy_at_least_mact
    and at_least_two_datasets_at_least_mact
    and token_ratio is not None
    and token_ratio <= 0.75
),
```

Compute `risk_accuracy_deltas` for shared risk levels using `myagent["risk_strata"]` and `mact["risk_strata"]`.

- [ ] **Step 5: Run focused tests and commit Task 6**

Run:

```powershell
python -m unittest tests.test_evaluate_results tests.test_compare_blind_results -v
```

Expected: `OK`.

Commit:

```powershell
git add code/evaluate_results.py tests/test_evaluate_results.py code/compare_blind_results.py tests/test_compare_blind_results.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: report selective risk metrics"
```

## Task 7: Risk Policy Calibration Utility

**Files:**
- Create: `code/calibrate_risk_policy.py`
- Create: `tests/test_calibrate_risk_policy.py`

- [ ] **Step 1: Write failing calibration tests**

Add this test file:

```python
import unittest

from calibrate_risk_policy import choose_policy


class CalibrateRiskPolicyTests(unittest.TestCase):
    def test_choose_policy_prefers_accuracy_under_token_limit(self):
        rows = [
            {"risk": 0.2, "my_correct": True, "mact_tokens": 1000, "light_tokens": 200, "medium_tokens": 500, "high_tokens": 900},
            {"risk": 0.6, "my_correct": False, "mact_tokens": 1000, "light_tokens": 200, "medium_tokens": 500, "high_tokens": 900},
            {"risk": 0.8, "my_correct": True, "mact_tokens": 1000, "light_tokens": 200, "medium_tokens": 500, "high_tokens": 900},
        ]
        result = choose_policy(rows)
        self.assertLessEqual(result["avg_token_ratio"], 0.75)
        self.assertIn(result["light_threshold"], [0.20, 0.25, 0.30])
        self.assertIn(result["high_threshold"], [0.50, 0.55, 0.60])

    def test_choose_policy_does_not_read_gold_or_ids(self):
        rows = [
            {"id": "known", "gold": "secret", "risk": 0.9, "my_correct": True, "mact_tokens": 1000, "light_tokens": 200, "medium_tokens": 500, "high_tokens": 900}
        ]
        result = choose_policy(rows)
        self.assertNotIn("gold", str(result).lower())
        self.assertNotIn("known", str(result).lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run focused test and confirm RED**

Run:

```powershell
python -m unittest tests.test_calibrate_risk_policy -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'calibrate_risk_policy'`.

- [ ] **Step 3: Implement calibration script**

Create `choose_policy(rows)`:

- Search `light_threshold` in `[0.20, 0.25, 0.30]`.
- Search `high_threshold` in `[0.50, 0.55, 0.60]`.
- Reject configs where `light_threshold >= high_threshold`.
- Estimate selected token per row by risk:
  - risk `< light_threshold`: `light_tokens`
  - risk `< high_threshold`: `medium_tokens`
  - otherwise: `high_tokens`
- Compute `avg_token_ratio = avg_selected_tokens / avg_mact_tokens`.
- Reject configs above `0.75`.
- Maximize `(accuracy, -avg_token_ratio, -high_threshold)` using only fields `risk`, `my_correct`, `mact_tokens`, `light_tokens`, `medium_tokens`, `high_tokens`.
- Return a JSON-serializable dict with thresholds, average token ratio, estimated accuracy, and row count.

Add CLI:

```powershell
python code/calibrate_risk_policy.py --input outputs/dev_selective/evaluated_rows.jsonl --output outputs/dev_selective/risk_policy.json
```

The CLI must read JSONL, write JSON with UTF-8 encoding, and print the selected policy.

- [ ] **Step 4: Run focused test and commit Task 7**

Run:

```powershell
python -m unittest tests.test_calibrate_risk_policy -v
```

Expected: `OK`.

Commit:

```powershell
git add code/calibrate_risk_policy.py tests/test_calibrate_risk_policy.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md
git commit -m "feat: calibrate selective risk policy"
```

## Task 8: Full Verification and Development Benchmark

**Files:**
- Modify: `docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md`
- Create: `outputs/selective_risk_dev_2026-06-27/`

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 2: Run compile and diff checks**

Run:

```powershell
python -m compileall -q code
git diff --check
```

Expected: both commands exit with code `0`.

- [ ] **Step 3: Run small three-dataset development sample**

Use the existing DeepSeek environment variables already configured by the user. Run 5 samples per dataset first:

```powershell
python code/tqa.py --task wtq --dataset_path datasets_ready/wtq_unseen_sample5.jsonl --output_path outputs/selective_risk_dev_2026-06-27/wtq_myagent.jsonl --collaboration_mode selective --mact_avg_tokens 8867 --plan_model_name deepseek-v4-pro --model_provider deepseek --api_key_env ANTHROPIC_AUTH_TOKEN --api_base https://api.deepseek.com --thinking disabled --temperature 0 --max_tokens 2048
python code/tqa.py --task tabfact --dataset_path datasets_ready/tabfact_test_sample5.jsonl --output_path outputs/selective_risk_dev_2026-06-27/tabfact_myagent.jsonl --collaboration_mode selective --mact_avg_tokens 8867 --plan_model_name deepseek-v4-pro --model_provider deepseek --api_key_env ANTHROPIC_AUTH_TOKEN --api_base https://api.deepseek.com --thinking disabled --temperature 0 --max_tokens 2048
python code/tqa.py --task crt --dataset_path datasets_ready/crt_sample5.jsonl --output_path outputs/selective_risk_dev_2026-06-27/crt_myagent.jsonl --collaboration_mode selective --mact_avg_tokens 8867 --plan_model_name deepseek-v4-pro --model_provider deepseek --api_key_env ANTHROPIC_AUTH_TOKEN --api_base https://api.deepseek.com --thinking disabled --temperature 0 --max_tokens 2048
```

- [ ] **Step 4: Evaluate the development sample**

Run:

```powershell
python code/evaluate_results.py outputs/selective_risk_dev_2026-06-27/wtq_myagent.jsonl
python code/evaluate_results.py outputs/selective_risk_dev_2026-06-27/tabfact_myagent.jsonl
python code/evaluate_results.py outputs/selective_risk_dev_2026-06-27/crt_myagent.jsonl
```

Expected:

- No execution failure spike.
- Each output includes `risk_strata`.
- Average token use remains under `0.75 * 8867 = 6650.25`.

- [ ] **Step 5: Run blind-style 20-sample check only after unit tests pass**

Use the frozen V4 local holdout as a development regression, not a final thesis result:

```powershell
python code/tqa.py --task wtq --dataset_path datasets_ready/blind_holdout_v4_2026-06-24/wtq.jsonl --limit 20 --output_path outputs/selective_risk_dev_2026-06-27/wtq_v4_20_myagent.jsonl --collaboration_mode selective --mact_avg_tokens 8867 --plan_model_name deepseek-v4-pro --model_provider deepseek --api_key_env ANTHROPIC_AUTH_TOKEN --api_base https://api.deepseek.com --thinking disabled --temperature 0 --max_tokens 2048
python code/tqa.py --task tabfact --dataset_path datasets_ready/blind_holdout_v4_2026-06-24/tabfact.jsonl --limit 20 --output_path outputs/selective_risk_dev_2026-06-27/tabfact_v4_20_myagent.jsonl --collaboration_mode selective --mact_avg_tokens 8867 --plan_model_name deepseek-v4-pro --model_provider deepseek --api_key_env ANTHROPIC_AUTH_TOKEN --api_base https://api.deepseek.com --thinking disabled --temperature 0 --max_tokens 2048
python code/tqa.py --task crt --dataset_path datasets_ready/blind_holdout_v4_2026-06-24/crt.jsonl --limit 20 --output_path outputs/selective_risk_dev_2026-06-27/crt_v4_20_myagent.jsonl --collaboration_mode selective --mact_avg_tokens 8867 --plan_model_name deepseek-v4-pro --model_provider deepseek --api_key_env ANTHROPIC_AUTH_TOKEN --api_base https://api.deepseek.com --thinking disabled --temperature 0 --max_tokens 2048
```

Evaluate each file with:

```powershell
python code/evaluate_results.py outputs/selective_risk_dev_2026-06-27/wtq_v4_20_myagent.jsonl
python code/evaluate_results.py outputs/selective_risk_dev_2026-06-27/tabfact_v4_20_myagent.jsonl
python code/evaluate_results.py outputs/selective_risk_dev_2026-06-27/crt_v4_20_myagent.jsonl
```

Compare with the existing V4 MACT outputs when present in `outputs/blind_holdout_v4_2026-06-24/`.

- [ ] **Step 6: Produce a concise development report**

Create `outputs/selective_risk_dev_2026-06-27/REPORT.md` with:

- Code commit hash.
- Unit test command results.
- Per-dataset accuracy, risk-stratum accuracy, average tokens, and failure count.
- Whether the run is development or formal blind.
- Any remaining gap versus MACT.
- The next formal blind protocol: at least 100 random unseen samples per dataset, table-ID isolation, code frozen before scoring.

- [ ] **Step 7: Commit final verified state**

Run:

```powershell
git status --short
```

Stage only files created or modified by this plan, then commit:

```powershell
git add code/risk_control.py code/evidence_builder.py code/selective_collaboration.py code/my_agents.py code/tqa.py code/run_wtq_myagent.py code/evaluate_results.py code/compare_blind_results.py code/calibrate_risk_policy.py tests/test_risk_control.py tests/test_evidence_builder.py tests/test_selective_collaboration.py tests/test_myagent_pipeline.py tests/test_task_modes.py tests/test_evaluate_results.py tests/test_compare_blind_results.py tests/test_calibrate_risk_policy.py docs/superpowers/plans/2026-06-27-selective-risk-collaboration.md outputs/selective_risk_dev_2026-06-27/REPORT.md
git commit -m "test: verify selective risk collaboration"
```

Expected: commit succeeds.

## Self-Review

- Spec coverage:
  - Risk formulas and thresholds are covered by Task 1.
  - EvidencePack and dataset-aware evidence signals are covered by Task 2.
  - Candidate comparison, post-risk disagreement, and Thinking fallback are covered by Task 3.
  - `TableQAPipeline` selective orchestration is covered by Task 4.
  - CLI switching and output observability are covered by Task 5.
  - Risk-stratified evaluation and 0.75 token acceptance are covered by Task 6.
  - Threshold calibration under token budget is covered by Task 7.
  - Unit tests plus small WTQ/TabFact/CRT development verification are covered by Task 8.
- Placeholder scan:
  - This plan contains concrete file paths, test code, commands, and expected results.
  - No sample ID, table ID, gold answer, or fixed entity branch is introduced.
- Type consistency:
  - `RiskAssessment.to_dict()`, `EvidencePack.to_dict()`, `CandidateAnswer.to_dict()`, and `AgreementDecision.to_dict()` are the serialization interfaces used by `tqa.py`.
  - `risk_level` values are `light`, `medium`, `high`, `fallback`, and `unknown` only in reports.
