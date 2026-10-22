# Pilot 4 behavioural breakdown

## Notes

- Supplied 2026-10-22: `source-proposal.docx`; readable `source-proposal.md`;
  provenance `source-record.json`.
- Review and first interview frontier: `design.md`.
- Specification: `spec.md`; approved breakdown: `ticket-proposal.md`.
- Individual implementation tickets: `issues/01-verified-source-scoring.md`
  through `issues/05-combined-findings-handoff.md`.
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
- Audit selection: all approved transition/invalid/extractor-disagreement cases
  plus 20 seeded remaining unchanged pairs per method; report packet size (Q6).
- Identify intended answers while gold/scorers and method/checkpoint are hidden;
  freeze labels before the second-stage reveal (Q7).
- Public saved-workflow test boundary approved; CPU-only fixtures and separate
  real-artifact acceptance evidence.
- Five-ticket granularity and dependencies approved; tickets published locally.
  Ticket 01 is implemented and executed against saved real outputs; see
  `execution-ticket01.md`. Tickets 02 and 03 are also implemented and executed; see
  `execution-tickets02-03.md`. Ticket 04 (blinded audit) is the next frontier.
- No categorical decision thresholds or new inference are approved. The user
  authorized Ticket 01 implementation separately after the design interview.

## Fog

- Whether forced read-off is needed, to be decided after CPU findings.
- Actual audit workload; later-ticket real-data execution evidence.
