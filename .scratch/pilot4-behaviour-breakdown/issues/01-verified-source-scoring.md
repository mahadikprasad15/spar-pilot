# 01: Verify saved greedy sources and publish a harmonized scoring report

**What to build:** A CPU workflow that imports the selected existing SFT/GRPO
greedy endpoint outputs and publishes a verified source inventory and scoring
report. It is usable from the shared follow-up notebook, including after a
fresh-session restart, without assuming a shared baseline or changing old data.

**Blocked by:** None (can start immediately).

**Status:** claimed

**Parent:** Behavioural gain breakdown specification.

- [ ] Public commands accept explicit source selections, analysis identity and
      artifact root; no model runtime, download, CUDA probe or inference is needed.
- [ ] Source adapters handle the existing greedy response and paired-trajectory
      exports while treating repeated baseline references as references, not
      additional questions. Preserve raw text, stored scores and source identities.
- [ ] Validate unique complete checkpoint/question identities, question/gold
      alignment, prompt/decoding compatibility and available provenance. Missing
      evidence or mismatches are visible; no original run identity is invented.
- [ ] Preserve method-specific step-0 records unless identical source content
      and inputs/settings justify deduplication. Expose original and common-ID
      cohorts without silently substituting overlap results for the headline.
- [ ] Apply existing strict/flexible-v3 scoring unchanged and the specified
      gold-independent last-number diagnostic. Save extracted candidates,
      exact numeric comparisons, spans, cap status and invalid reasons.
- [ ] The report exposes stored versus harmonized scores, per-file counts,
      extractable fractions, conditional accuracy and capped cases. It does not
      claim that the diagnostic is format-blind or human ground truth.
- [ ] Source snapshots, configuration, implementation identities, structured
      records, receipts and status support idempotent replay and reject changed
      sources under the same identity. Prior reports remain unchanged.
- [ ] The notebook's setup/source/scoring sections restore required saved state
      without hidden globals and display the usable scoring report.
- [ ] Public-workflow CPU fixtures cover supported imports, numeric edge cases,
      gold independence, historical-score preservation, caps, incomplete or
      duplicate identities, mismatches, source changes and fresh-process resume.
- [ ] Real source availability and execution evidence are recorded separately
      from fixture success; do not report real Pilot 4 rescoring as completed
      merely because the command implementation passes its tests.

## Comments

2026-10-22: User requested starting Ticket 01 in parallel with two local HTML
explainers. Implementation is in progress; no new inference is authorized.
