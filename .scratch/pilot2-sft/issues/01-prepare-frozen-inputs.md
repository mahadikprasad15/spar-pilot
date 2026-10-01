# 01: Prepare frozen training inputs and exploratory plan

Type: task
Status: resolved
Blocked by: None (can start immediately)

**What to build:** A public preparation operation that produces a reproducible
Pilot 2 training plan from the frozen Pilot 1 GSM8K inputs, ready for inspection
before downloading model weights or beginning training.

- [x] Select and persist exactly 512 distinct official training examples using
  seed 42 and the frozen dataset revision; save algorithm, IDs, indices and hashes.
- [x] Reuse the exact held-out 150 evaluation items, verify split provenance and
  reject identical questions shared by the training and evaluation cohorts.
- [x] Preserve original gold annotations and final-answer markers; match actual
  evaluation chat/system behavior and persist rendered training examples.
- [x] Verify completion-only labels, prompt/padding masking and supervised
  end-turn; derive the limit from full sequence lengths and reject truncation.
- [x] Save gold-target and full-sequence length diagnostics with counting rules.
- [x] Save a dated exploratory plan containing the frozen protocol, measurements,
  paired interval procedure/settings and evidence limits, without a directional
  prediction or binary collapse gate.
- [x] Repeated preparation verifies and reuses immutable inputs; mismatches fail
  visibly without overwriting existing artifacts.
- [x] CPU tests exercise the public operation using fake external boundaries,
  with no model/dataset download, following red/green TDD.

## Comments

Approved as ticket 1 of the five-ticket breakdown. Parent: Pilot 2 spec.


## Answer

Frozen preparation and its public CLI are implemented. Five CPU tests pass,
including masking, immutable reuse, corruption, overlap and overflow checks.
