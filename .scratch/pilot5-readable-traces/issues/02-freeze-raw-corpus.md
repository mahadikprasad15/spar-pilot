# 02: Prepare and freeze the raw corpus

**What to build:** Build the shared 200-document raw cohort, exact 128-content-token mapping and immutable input identities through readable preparation/review cells. Corpus selection can complete independently of vocabulary review.

**Blocked by:** 01: Restore a clean session, audit sources and smoke-test Colab.

**Status:** resolved

- [x] Public preparation recovers the original 150 document identities where eligible, applies an explicitly reviewed extension procedure and freezes the exact added 50. Selection is outcome-blind and never prefers documents because their previous results were clean or strong.
- [x] Preserve a legacy-150/new-50 stratum label through inputs, retained evidence and reports; do not assert new documents are unseen without checking/recording prior exposure.
- [x] Hand-worked token/offset cases and actual tokenizer metadata verify content positions and special-token/BOS behaviour; mismatches are surfaced rather than silently replaced.
- [x] Reuse verifies token/text/source hashes; identical preparation reuses saved evidence without resampling.
- [x] No learned lens output or grading is emitted by corpus preparation.

## Implementation evidence · 2026-10-14

- `traces-corpus-prepare` and `traces-corpus-verify` expose the saved workflow. Cells 7–9 implement review, CPU preparation and verification in the shared notebook.
- User approved the same pinned FineWeb first-2,000 pool, outcome-blind disjoint extension, 128-content-token eligibility, separate local seed 42 and source-order preservation. No silent legacy replacement.
- 17 corpus tests cover independent hand-token cases, actual native tokenizer JSON/BOS handling, public child-process commands, seal corruption, unchanged global RNG, cohort counts, fresh restore and reuse without dataset/tokenizer access. Together with bootstrap and existing workflow tests: 32 passed.
- CPU input preparation verifies the parent input completion seal and every declared input payload, without repeating the already completed training/checkpoint audit. Capture still requires adapter/source qualification.
- At this implementation checkpoint the real pinned-data run was pending; it subsequently passed as recorded below.

## Target verification · 2026-10-15

The user supplied successful public Colab verification: `corpus_complete: true`, 200 examples (150 legacy/50 new), 128 content tokens each, tokenizer 0.22.1 with no added special/BOS token, and child exit code 0. All 50 new source indices and both tokenizer hashes are preserved in `artifacts/validation/pilot5-corpus-2026-10-14/colab-verification-2026-10-15.json`. This is user-reported validator evidence, not an independent fetch of the Drive payloads. Ticket 02 is resolved. No model activations or lens readouts were produced by corpus preparation.
