# 05: Assemble the behavioural findings and operational handoff

**What to build:** One source-backed behavioural report and a finished readable
CPU follow-up notebook, assembling automatic and audited evidence without
overstating completion or attributing the gain to reasoning from answer counts.
The same reporting path supports a clearly labelled partial result while human
review is pending.

**Blocked by:** 02 — paired transitions; 03 — sampled reliability/coverage;
04 — blinded audit software and result contracts.

**Status:** resolved

**Parent:** Behavioural gain breakdown specification.

- [x] Assemble verified original/harmonized score totals, transition flows,
      direct paired comparisons, SFT churn/length descriptions, sampled count
      and coverage changes, and available audited extraction cases.
- [x] Reconcile every aggregate against saved per-item records, preserve cohort
      denominators and source identities, and link inspectable evidence.
- [x] Provide readable tables/plots, structured summary exports and methods/
      limitations alongside the interpretation; do not substitute a headline
      narrative for the underlying numbers or a blob of raw JSON for the report.
- [x] Report continuous counts and intervals with compatible explanations. Do
      not adopt the original numeric outcome categories or manufacture mutually
      exclusive causal shares of the net gain.
- [x] A partial report explicitly marks pending human review and omits unsupported
      audited claims. Complete audit status requires real frozen labels and
      the completed reveal product, not fixture or ticket completion.
- [x] The notebook is one incremental CPU workflow with concise stage purpose,
      named settings, saved-state discovery and a tested fresh-runtime resume
      path. No repair-cell chain or GPU/model dependency is introduced.
- [x] Separate the reviewer packet route from analyst reports and private audit
      mappings. Reviewer exports do not reveal fields hidden in stage one.
- [x] Resume verified completed stages and reject stale parent products without
      repeatedly scanning unrelated Drive/model/activation trees.
- [x] Record software implemented, automatic real-data results ready, human
      audit pending and human audit completed as distinct evidence states.
- [x] End with the evidence needed to decide whether forced read-off or a
      focused reasoning audit is warranted. Launch neither automatically.
- [x] Public-workflow end-to-end fixtures exercise source-to-partial-report,
      source-to-frozen-audit-report, interrupted/fresh-session resume, stale
      source/label rejection and per-item/report reconciliation.
- [x] A separate real-artifact acceptance record states what was actually
      retrieved, verified, executed and reviewed; no new inference is required.


## Execution record

2026-10-22: Implemented and executed. The combined v2 report reconciles original
scores, transitions, sampled reliability and the approved partial confirmed-case
corrections. It does not claim a completed original full-transcription audit.
All 16 targeted flags have human confirmations; 188 automatic proposals remain
excluded from corrections. See `../execution-ticket05.md`.

Validation: start-of-session full baseline 473 passed / 35 skipped / four existing
Torch-import warnings; post-change affected regressions 104 passed. Fresh-process
notebook resume, sealed source/output checks, HTML links/schema and exported
figures were verified. Local browser automation was unavailable, so no browser
visual QA is claimed. Changes are committed locally; not pushed.
