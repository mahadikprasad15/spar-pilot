# Pilot 4 behavioural breakdown

## Notes

- Supplied 2026-10-22: `source-proposal.docx`; readable `source-proposal.md`;
  provenance `source-record.json`.
- Review and first interview frontier: `design.md`.
- Parent experiment: `.scratch/pilot4-grpo/spec.md`.
- Scorer history: `docs/adr/0005-gsm8k-flexible-v3.md`.

## Decisions so far

- The user requested persistence of the proposal and a design-readiness review
  using grill-with-docs.
- One human reviewer; no inter-reviewer agreement claim (design.md, Q2).
- Answer-level decomposition first; consider a focused reasoning-validity
  audit after the initial breakdown, rather than including exhaustive reasoning
  annotation now (design.md, Q1).
- CPU-only source audit and A/B/D first; review findings and inference budget
  before deciding on C (design.md, Q3).
- Targeted human audit for transitions/extraction errors with a reproducible
  unchanged-case sample; no exact overall hand-corrected accuracy claim (Q4).
- Counts and intervals with compatible explanations; the supplied categorical
  decision thresholds are not adopted (Q5).
- No decision-rule thresholds, new inference or implementation have been
  approved in this interview.

## Fog

- Single-reviewer audit sampling/coverage, blinding and rubric.
- Whether forced read-off is needed, to be decided after CPU findings.
- Transition denominators, finite-draw interpretation and coherent decision rules.
- Exact saved-response availability and SFT/GRPO source compatibility.
- Any glossary additions and ADRs depend on settled interview answers.
