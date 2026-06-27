# Selective Risk Collaboration Dev Check - 2026-06-27

Scope: small development sample only, 5 records per dataset. This is not a
formal blind benchmark.

## Current Result

| Dataset | Correct | Accuracy | Avg API tokens | Failed exec |
| --- | ---: | ---: | ---: | ---: |
| WTQ | 4/5 | 80.0% | 2428.0 | 0 |
| TabFact | 4/5 | 80.0% | 2138.0 | 0 |
| CRT | 5/5 | 100.0% | 3631.0 | 0 |
| Combined | 13/15 | 86.7% | 2732.3 | 0 |

The configured MACT average-token reference is 8867.0. The current combined
average is about 30.8% of that reference, leaving a large token margin while
recovering the CRT failures that previously dominated this sample.

## What Changed

- Synced normalized `final_value` back to `final_answer`, so JSONL output and
  evaluator-facing fields stay consistent.
- Added WTQ scalar normalization for datetime-like values, country-code answers,
  and year-unit answers.
- Added planner and validator rules for strict "after YYYY" season boundaries,
  percentage snapshot tables, hard-coded label overrides, and unsupported summary
  row exclusion.
- Added CRT deterministic semantic shortcuts for:
  - duration change questions where `24 hours` and `1 day` are equivalent;
  - event-type association questions that require a systematic mapping;
  - average percentage snapshot questions over `% (year)` columns.

## Remaining Errors

- WTQ `nu-0`: the benchmark gold is `Italy`, while the table allows a plausible
  country-count interpretation that returns `Spain`. This looks like an ambiguous
  denotation/benchmark convention issue, not an execution failure.
- TabFact `tabfact-test-4`: the model interpreted "do not continue on" as
  excluding other top-10 finishes, while the gold treats the statement as true.
  This needs a larger TabFact blind check before adding another rule.

## Verification

- `python -m unittest discover -s tests -q`: 150 tests, OK.
- `python -m compileall -q code`: OK.
- `git diff --check`: OK, only line-ending warnings from Git on Windows.

## Files

- WTQ output: `outputs/selective_risk_dev_2026-06-27/wtq_myagent.jsonl`
- TabFact output: `outputs/selective_risk_dev_2026-06-27/tabfact_myagent.jsonl`
- CRT output: `outputs/selective_risk_dev_2026-06-27/crt_myagent.jsonl`
- Error files: `outputs/selective_risk_dev_2026-06-27/*_errors.jsonl`

## Next Gate

Freeze this logic and run a larger blind sample next. I would use at least 20
records per dataset, keeping the current acceptance target: accuracy no worse
than MACT on the same blind rows, with average tokens below 75% of the MACT
reference.
