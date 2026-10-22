# Proposed implementation slices

Status: proposed; user approval required before publishing individual tickets

Parent: behavioural gain breakdown specification in this effort.

Every slice uses the approved public saved-workflow boundary, tiny offline CPU
fixtures and the same incremental follow-up notebook. Each slice produces a
usable saved result; no schema-only, parser-only or UI-only tickets are proposed.
No wide refactor or separate prefactoring ticket is currently justified.

## 01 — Verify saved greedy sources and publish a harmonized scoring report

**Blocked by:** None.

**What it delivers:** A user selects existing SFT/GRPO outputs and obtains a
verified CPU source bundle, source-compatibility inventory and endpoint scoring
report under strict, flexible-v3 and the last-number diagnostic. The notebook
can restore and display it after a fresh-session restart. Historical responses
and scores remain unchanged. Source discrepancies and missing provenance are
explicit; no raw-response availability or shared step-0 identity is assumed.

**Verification:** Public-command tiny fixtures exercise supported source shapes,
numeric/cap semantics, gold independence, historical-score preservation,
duplicate/incomplete identities, source changes and idempotent resume.

## 02 — Explain endpoint gains and losses with paired transition reports

**Blocked by:** 01.

**What it delivers:** Saved per-question transitions and strict/flexible-v3 flow
tables for both methods, last-number diagnostics, gross/net gains and losses,
SFT churn and descriptive length/gold-step views. A direct paired GRPO64-minus-
SFT64 interval and within-method intervals use the declared matching cohort.
Tables and plots are inspectable from the notebook and a CPU report.

**Verification:** Hand-computable public-workflow fixtures reconcile all flows
to endpoint totals, retain distinct baselines, expose overlap denominators and
check reproducible paired item resampling without claiming seed uncertainty.

## 03 — Measure observed reliability and coverage of saved GRPO draws

**Blocked by:** 01.

**What it delivers:** A user imports the saved GRPO sampled endpoints and gets
validated eight-draw per-question counts, mean reward, observed coverage,
coverage gains/losses, count-transition views and item-clustered intervals for
both diagnostics. An optional baseline cross-check requires matching cohort,
role and decoding. The CPU notebook restores this product independently of
human audit progress.

**Verification:** Public-command fixtures reject missing/duplicate draws and
incompatible baseline samples, and distinguish observed coverage, mean reward
and probability sharpening without labeling 0/8 as zero ability.

## 04 — Run the targeted blinded audit through frozen labels and reveal

**Blocked by:** 02.

**What it delivers:** Deterministic mandatory selection plus 20 stable control
pairs per method; displayed workload; an opaque reviewer packet with shared
responses deduplicated; resumable single-reviewer intended-answer entry;
complete-label freeze; separate gold/scorer reveal and error-attribution output.
The human workflow is usable through the notebook and saved review artifacts;
software fixtures demonstrate the round trip without substituting for real
review completion.

**Verification:** Public-workflow fixtures check exact inclusion, seeded
controls, actual reviewer-export blinding, question/response binding, draft
resume, pending/duplicate/malformed-label rejection, label immutability and
explicit post-reveal revisions. Audit claims remain selected-case only.

## 05 — Assemble the behavioural findings and operational handoff

**Blocked by:** 02, 03, 04.

**What it delivers:** One source-backed report assembles the automatic and
audited products, with transitions, paired comparisons, sampled reliability,
audited extraction examples and limitations. It also renders a truthful partial
report while real human labels are pending. The finished CPU notebook has a
clear resume path and distinguishes software implemented, automatic results
ready, audit pending and audit completed. It records evidence for deciding
whether forced read-off or a focused reasoning review is warranted; neither
runs automatically.

**Verification:** End-to-end public-workflow fixtures cover fresh-session
restart, stale parent rejection, report/per-item reconciliation and partial
versus completed audit states. A separate real-artifact checklist records what
was actually executed, reviewed and verified without model inference.

## Dependency review

- 01 has a useful score report before transition/audit functionality exists.
- 02 and 03 can proceed independently after source/scorer contracts in 01.
- 04 needs 02's question-level transition product, but does not need 03.
- 05 consumes all three analysis products; it must not mark a real human audit
  completed merely because the software ticket's fixture round trip passed.
- No ticket implements forced read-off, new training or a reasoning-validity audit.

## Questions before publication

1. Is five slices the right granularity?
2. Do the listed blockers represent genuine dependencies?
3. Should any slice be merged or split?

Individual ready-for-agent tickets are not published until this breakdown is
approved. No implementation has started.
