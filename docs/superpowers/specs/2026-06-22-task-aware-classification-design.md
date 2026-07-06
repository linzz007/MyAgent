# Task-Aware Classification Design

## Goal

Make myAgent correctly handle TabFact fact verification and CRT's explicitly constrained yes/no questions without forcing CRT's numeric and text questions into a classifier.

## Architecture

`TQASessionState` carries an optional `answer_mode`. The generic entry point maps `scitab` to `true_false`. CRT is mixed-format: `answer_mode_for_sample` selects `yes_no` only when the question explicitly requires a Yes/No-only answer; other CRT records and WTQ remain in the general pipeline. `TableQAPipeline` still performs Router scoring and question-aware compression for every task. After compression, classification modes use `FinalAnswerAgent.classify`; general questions continue through SIMPLE lookup or COMPLEX Planner/Calculator/Critic.

## Classification Contract

The classifier receives the statement/question and compressed table preview. It requests a JSON object with one label from the mode's closed set. `true_false` accepts only `true` or `false`; `yes_no` accepts only `Yes` or `No`. Parsing prefers JSON and then a strict whole-word fallback. Missing or ambiguous labels raise a runtime error instead of silently choosing a value.

On success, the label is written to `state.final_value` and `state.final_answer`, `exec_success` is true, and classification metadata is added to cost metrics. This makes the existing evaluator use the structured label for exact match.

## Testing

Fake-LLM tests verify per-sample CRT mapping, strict label normalization, invalid-output failure, and that classification runs Router + compression + one classifier call without invoking Planner, Calculator, or Critic. Existing WTQ SIMPLE/COMPLEX tests remain unchanged. Real one-record and five-record TabFact/CRT runs verify canonical labels and mixed CRT execution.
