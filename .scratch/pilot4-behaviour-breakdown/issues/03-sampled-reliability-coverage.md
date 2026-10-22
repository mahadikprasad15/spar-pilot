# 03: Measure observed reliability and coverage of saved GRPO draws

**What to build:** A CPU import-and-report path for the saved eight-draw GRPO
endpoint evaluations, showing mean reward, observed coverage and changes in
correct-draw counts. It is runnable independently of transition reporting and
human audit once the source/scorer contracts are available.

**Blocked by:** 01 — verified source and harmonized scoring workflow.

**Status:** claimed

**Parent:** Behavioural gain breakdown specification.

- [ ] Import explicitly selected sampled endpoints with method/checkpoint,
      question and draw identities, decoding metadata and available provenance.
- [ ] Require eight distinct draws per question at both endpoints and a common
      evaluation cohort. Missing or duplicate draws block the fixed-eight
      comparison rather than silently changing its denominator.
- [ ] Apply flexible-v3 and the declared last-number diagnostic without changing
      saved responses or historical scores.
- [ ] Save correct-draw counts per item, mean draw reward, observed eight-draw
      coverage, coverage gains/losses and the full count-transition table.
- [ ] Use 10,000 paired question-level bootstrap resamples with seed 42,
      retaining all eight observed draws for each resampled item. Disclose that
      this conditions on the observed draws and the single trained run.
- [ ] Describe 0/8→positive transitions as observed coverage gains, not newly
      acquired ability. Flat coverage with greater mean reward is compatible
      with sharpening but does not uniquely prove it.
- [ ] An optional untuned-sampling cross-check requires matching question IDs,
      evaluation role and decoding. Reject training-cohort substitutes and do
      not pool additional draws opportunistically.
- [ ] Persist source-bound per-item rows, metrics, tables/plots, status and a
      notebook-readable report that resumes without waiting for human labels.
- [ ] Public-command CPU fixtures distinguish coverage from mean reward and
      exercise draw completeness, incompatible cross-checks, paired resampling,
      fresh-session replay and altered-source rejection.
- [ ] Fixture success and actual real-source execution have separate evidence.

2026-10-22: User authorized the next tickets; CPU saved-output implementation started.
