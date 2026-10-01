# 01: Prepare frozen training inputs and exploratory plan

Type: task
Status: ready-for-agent
Blocked by: None (can start immediately)

**What to build:** A public preparation operation that produces a reproducible
Pilot 2 training plan from the frozen Pilot 1 GSM8K inputs, ready for inspection
before downloading model weights or beginning training.

- [ ] Select and persist exactly 512 distinct official training examples using
  seed 42 and the frozen dataset revision; save algorithm, IDs, indices and hashes.
- [ ] Reuse the exact held-out 150 evaluation items, verify split provenance and
  reject identical questions shared by the training and evaluation cohorts.
- [ ] Preserve original gold annotations and final-answer markers; match actual
  evaluation chat/system behavior and persist rendered training examples.
- [ ] Verify completion-only labels, prompt/padding masking and supervised
  end-turn; derive the limit from full sequence lengths and reject truncation.
- [ ] Save gold-target and full-sequence length diagnostics with counting rules.
- [ ] Save a dated exploratory plan containing the frozen protocol, measurements,
  paired interval procedure/settings and evidence limits, without a directional
  prediction or binary collapse gate.
- [ ] Repeated preparation verifies and reuses immutable inputs; mismatches fail
  visibly without overwriting existing artifacts.
- [ ] CPU tests exercise the public operation using fake external boundaries,
  with no model/dataset download, following red/green TDD.

## Comments

Approved as ticket 1 of the five-ticket breakdown. Parent: Pilot 2 spec.

