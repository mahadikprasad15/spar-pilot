# Pilot 1 — behavioral baseline

## Question

What is the untuned instruction model's performance under a reproducible
GSM8K/MMLU evaluation protocol, and how do prompt, shot and scoring conventions
affect the reported scores?

## Design and configuration

| Field | Protocol |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`; model/tokenizer revisions frozen in each plan |
| GSM8K | 150 distinct official test items, seed 42; saved ordered IDs |
| MMLU | 20 official test items per subject across 57 subjects: 1,140 items |
| Generation | Greedy; GSM8K cap 1,024 new tokens; MMLU text cap 32 |
| GSM8K scoring | Strict and versioned flexible numeric extraction on the same saved responses |
| MMLU scoring | Generated text and constrained next-token A/B/C/D logits, at 0 and 5 shots |
| Primary MMLU metric | 5-shot constrained-logit accuracy |
| Precision | Separately named unquantized BF16/FP16 variants; inspect actual run config |
| Uncertainty | Wilson intervals for GSM8K; within-subject bootstrap for MMLU |

The GSM8K prompt requests step-by-step reasoning and a final `#### <number>`
line. Five-shot MMLU uses the five dev examples from the same subject, in
published order. Full rendered prompts and candidate tokenization are retained.
Capped and invalid outputs remain in the accuracy denominator. Response length
excludes EOS and padding.

## Decisions

- Historical SPAR numbers lack their original prompts/items/scorers. They are
  reference results, not exact-reproduction targets or acceptance gates.
- The subject-balanced MMLU subset is a pilot sample, not full-test MMLU.
- The recorded chat rendering is authoritative. A stated no-system-message
  intention did not prevent Qwen's template from inserting a default system text.
- Exact-logit ties received a versioned invalid/incorrect-and-continue rule.
- Flexible-v2/v3 are separate rescoring products. Prior responses and scores
  remain preserved; no inference is required for these audits.

## Results and execution status

Baseline reports and rescoring exports exist locally. The canonical run set
and current completion state require reconciliation against Drive before a
single numerical summary is designated as the Pilot 1 result.

The Pilot 2 matched FP32 baseline is a separate measurement and must not be
substituted with an older lower-precision baseline.

See [status](status.md) and [artifact access](../artifacts.md).

## Outputs

Frozen cohorts and prompts; per-item generated text/token IDs; extraction and
correctness decisions; MMLU candidate logits; cap/invalid counts; runtime;
structured summaries and report source inventory.

## Limitations

Sampling uncertainty is conditional on the selected benchmark procedure.
Numeric extraction scores final answers, not reasoning validity. A format-only
score change does not establish a change in problem-solving ability.

## References

- [Specification](../../.scratch/pilot-1-evaluation/spec.md)
- [Frozen evaluation protocol](../adr/0001-frozen-pilot-evaluation-protocol.md)
- [Flexible-v2](../adr/0003-gsm8k-flexible-v2.md) and [flexible-v3](../adr/0005-gsm8k-flexible-v3.md)
- [Notebook](../../notebooks/pilot-1-colab.ipynb)
- Implementation: `protocol.py`, `backend.py`, `scoring.py`, `run.py`, `reporting.py`.
