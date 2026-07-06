# Accuracy Recovery V2 Design

## Goal

Raise myAgent accuracy toward or above MACT while retaining at least a 2x
average API-token advantage. The first gate is the existing deterministic
18-sample benchmark; the next gate is a 100-sample-per-dataset server pilot.

## Evidence

The final benchmark has three recurring failure classes:

- WTQ returns the correct semantic value in the wrong output shape or format.
- TabFact sends numeric comparison statements directly to a one-shot label
  classifier instead of Planner/Calculator.
- CRT closed-label questions mix simple classification with questions that
  require arithmetic or temporal comparison.

The current Planner and Critic prompts also describe every input as a Chinese
government statistics table, although the benchmark contains English sports,
music, awards, and competition tables.

## Architecture

Add an `AnswerContract` inferred only from the question and existing
`answer_mode`. It defines whether the answer must be a scalar, entity list, or
one label from a closed set. It also carries a deterministic
`reasoning_required` decision for closed-label questions containing numeric,
aggregate, or comparison operations.

The pipeline becomes:

1. Route and compress as before.
2. Use direct classification only for low-risk closed-label questions.
3. Send high-risk closed-label, aggregate, and list questions through
   Planner/Calculator/Critic.
4. Normalize the structured value and validate it against the answer contract.
5. Replan on contract mismatch without spending a Critic call on an obviously
   invalid output.
6. Return structured labels/lists directly; use natural-language formatting
   only where it cannot change the evaluated value.

## Answer Contracts

- `scalar`: one number or text value.
- `list`: a Python/JSON list of entity strings; a cardinality is invalid.
- `label`: exactly one allowed label derived from `answer_mode`.

Normalization is deterministic and gold-independent. It may canonicalize
booleans, split a comma-delimited value only when a list is required, remove
stray escaped outer quotes, and remove an insignificant `.0` before common
scale words such as million or billion. It does not read gold answers.

## Risk Policy

Closed-label questions require code reasoning when they contain operations such
as counts, ratios, percentages, totals, averages, thresholds, comparative
language, or explicit numeric quantities beyond a year. Other closed-label
questions retain the cheap classifier.

For code reasoning, an invalid answer type triggers one replan with a concrete
contract error. The existing maximum replan count remains unchanged. Full
multi-view validation stays disabled in the primary condition.

## Prompt Policy

Planner and Critic prompts become domain-neutral and English-first. They state
the exact output contract and require exact DataFrame column names. List
questions must return entities rather than their count. Questions asking for
items in the same category as a named reference exclude the reference unless
the question explicitly says otherwise.

## Observability

Each result records the inferred contract, whether risk escalation occurred,
and the final contract validation result. Existing API usage, compression,
routing, and execution fields remain unchanged.

## Success Criteria

- All unit tests pass.
- Six final benchmark inputs remain schema-compatible.
- myAgent accuracy improves on at least one dataset without regressing the
  others by more than one sample.
- Average token usage remains below 50% of MACT on every dataset.
- No gold answer is used for routing, normalization, or validation.

