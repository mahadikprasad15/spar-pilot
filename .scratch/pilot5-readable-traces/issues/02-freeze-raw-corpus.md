# 02: Prepare and freeze the raw corpus

**What to build:** Build the shared 200-document raw cohort, exact 128-content-token mapping and immutable input identities through readable preparation/review cells. Corpus selection can complete independently of vocabulary review.

**Blocked by:** 01: Restore a clean session, audit sources and smoke-test Colab.

**Status:** ready-for-agent

- [ ] Public preparation recovers the original 150 document identities where eligible, applies an explicitly reviewed extension procedure and freezes the exact added 50. Selection is outcome-blind and never prefers documents because their previous results were clean or strong.
- [ ] Preserve a legacy-150/new-50 stratum label through inputs, retained evidence and reports; do not assert new documents are unseen without checking/recording prior exposure.
- [ ] Hand-worked token/offset cases and actual tokenizer metadata verify content positions and special-token/BOS behaviour; mismatches are surfaced rather than silently replaced.
- [ ] Reuse verifies token/text/source hashes; identical preparation reuses saved evidence without resampling.
- [ ] No learned lens output or grading is emitted by corpus preparation.
