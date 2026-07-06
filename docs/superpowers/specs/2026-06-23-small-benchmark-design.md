# Small Cross-Project Benchmark Design

## Goal

Measure MACT and myAgent on the same small but diverse WTQ, TabFact, and CRT-QA samples, using comparable correctness and actual API-token metrics, then optimize only evidence-backed systemic failures.

## Sampling

- Use 18 records per dataset.
- Generate the full adapted JSONL once, then select a deterministic sample with seed 20260623.
- Prefer one record per table before selecting a second record from the same table.
- For TabFact, keep true/false labels as balanced as the source permits.
- For CRT-QA, include explicit Yes/No records and general numeric/text records.
- Store one shared sample JSONL per dataset; both projects consume the exact same files.

Sequential first-N sampling is rejected because the current TabFact and CRT source files group multiple questions from the same table. Unstratified random sampling is acceptable as a fallback but provides weaker coverage guarantees.

## Correctness Metrics

The common evaluator extracts one prediction with the same precedence for each schema: myAgent structured value or deterministic lookup value, otherwise natural-language answer; MACT `pred_answer`.

- WTQ: dataset-aware denotation accuracy. Gold and predicted answers are treated as sets, set cardinality must match, strings use the official evaluator's normalization rules, and numeric values compare numerically.
- TabFact: canonical classification accuracy after mapping `1/true/yes` to true and `0/false/no` to false.
- CRT-QA: normalized exact match for text/closed labels and numeric equality with a small floating-point tolerance.

The evaluator also reports execution failures and missing predictions. Legacy generic EM is retained only for compatibility and is not the primary comparison metric.

## Token and Cost Metrics

Both projects report provider-returned per-sample `prompt_tokens`, `completion_tokens`, `total_tokens`, and request count. myAgent keeps its estimator as a secondary diagnostic, but cross-project conclusions use only API usage. Average latency and compression ratio are reported separately.

## Run Configuration

- Model: `deepseek-v4-flash` for both planning and code generation.
- Provider: DeepSeek official OpenAI-compatible endpoint.
- Thinking: disabled.
- Temperature: 0.
- One plan candidate and one code candidate in MACT.
- Maximum three MACT steps for the small benchmark.
- API timeout 180 seconds and eight retries.
- myAgent multi-view validation disabled in the primary comparison; it remains a separate cost/accuracy ablation.

## Optimization Rule

Run the unchanged benchmark first. Inspect mismatches by category: adapter/gold issue, evaluator issue, routing/compression issue, generated-code execution issue, or model reasoning issue. Change code only for repeated system-level failures, add a failing regression test first, and rerun the same fixed samples. Do not tune prompts to isolated ambiguous gold examples.

## Outputs

- Three shared sample JSONL files.
- Six primary result JSONL files.
- Dataset-aware summaries with actual API tokens.
- A Chinese Markdown comparison report containing accuracy, tokens, calls, latency, failures, compression, mismatch categories, and go/no-go guidance.
