# 07: Measure GRPO writes on the existing fixed sequences

**What to build:** Use Pilot 3's instrument on the GRPO checkpoint trajectory,
producing comparable block-output changes, direct module contributions and
mean vectors without changing the measurement inputs.

**Blocked by:** 05 — Train and resume the complete 64-step GRPO arm.

**Status:** ready-for-agent

- [ ] Verify GRPO source checkpoint/base evidence behind an explicit source
      contract; do not pretend the GRPO arm has an SFT source configuration.
- [ ] Reuse exact GSM8K/FineWeb token IDs, masks, position rules and hashes,
      preserving complete gold solutions and all three measurement views.
- [ ] Retain primary block/module definitions, denominator diagnostics, token
      and equal-example weighting, mean vectors and undefined coverage.
- [ ] Validate zero writes, disabled-adapter invariance, all 196 rank-1 hooks,
      frozen weights, nonfinite handling and calibrated numerical agreement.
      New precision/runtime evidence cannot inherit old FP32 calibration.
- [ ] Profile the real complete measurement path, freeze passing batch
      membership and save verified resumable shards with final hash sealing.
- [ ] Produce CPU-only checkpoint reports and vectors with source identity,
      intervals and direction-resolution evidence.
- [ ] Tests cover controlled full workflow, source adaptation, mismatched token
      hashes, interrupted shards, numerical failures and unchanged old pilots.
- [ ] Notebook instructions explain forward-only work and provide a verified
      completion boundary before GPU release.

## Comments

- October 7, 2026: approved as ticket 7. Behavioural generation and fixed-token
  measurement are separate products and can run in separate GPU sessions.
