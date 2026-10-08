# 07: Measure GRPO writes on the existing fixed sequences

**What to build:** Use Pilot 3's instrument on the GRPO checkpoint trajectory,
producing comparable block-output changes, direct module contributions and
mean vectors without changing the measurement inputs.

**Blocked by:** 05 — Train and resume the complete 64-step GRPO arm.

**Status:** resolved

- [x] Verify GRPO source checkpoint/base evidence behind an explicit source
      contract; do not pretend the GRPO arm has an SFT source configuration.
- [x] Reuse exact GSM8K/FineWeb token IDs, masks, position rules and hashes,
      preserving complete gold solutions and all three measurement views.
- [x] Retain primary block/module definitions, denominator diagnostics, token
      and equal-example weighting, mean vectors and undefined coverage.
- [x] Validate zero writes, disabled-adapter invariance, all 196 rank-1 hooks,
      frozen weights, nonfinite handling and calibrated numerical agreement.
      New precision/runtime evidence cannot inherit old FP32 calibration.
- [x] Profile the real complete measurement path, freeze passing batch
      membership and save verified resumable shards with final hash sealing.
- [x] Produce CPU-only checkpoint reports and vectors with source identity,
      intervals and direction-resolution evidence.
- [x] Save one random rank-1 control using independent unit input/output
      directions in all 196 modules and an isolated seed-42 RNG. Match each
      module's effective update Frobenius norm to GRPO step 64, including
      alpha/r scaling; zero target norms yield zero updates.
- [x] Measure the control on the identical frozen inputs with the same
      instrument/settings, saving its realized activation magnitudes and
      per-layer signed cosines against SFT/GRPO mean writes for both weightings.
      Reuse verified SFT vectors when compatible; expose unresolved directions.
- [x] Verify control module mapping, unit-direction/norm reconstruction,
      seed reproducibility, zero targets, untouched source adapters and normal
      instrument gates. This is one random-intervention reference, not an
      estimated null distribution or numerical calibration substitute.
- [x] Tests cover controlled full workflow, source adaptation, mismatched token
      hashes, interrupted shards, numerical failures and unchanged old pilots.
- [x] Notebook instructions explain forward-only work and provide a verified
      completion boundary before GPU release.

## Comments

- October 7, 2026: approved as ticket 7. Behavioural generation and fixed-token
  measurement are separate products and can run in separate GPU sessions.
- October 7 amendment: the one random control is additional forward-only work
  in this ticket; it does not add a training arm or a tenth ticket.

## Answer

October 8 implementation: explicit verified GRPO source preparation, exact
Pilot 3 input reuse, one isolated seed-42 norm-matched random adapter, independent
calibration/profile/freeze evidence for both products, verified resumable
measurements and CPU reports/comparison. Control variants are zero and the one
random adapter labelled by its GRPO step-64 matching target, not a new training
trajectory. Reuses the existing instrument and persistence infrastructure.

Guided Colab sections 17–27 include profile/review, checkpoint/batch progress,
CPU verification before GPU release and compatible existing SFT-vector reuse.
The notebook is pinned to implementation commit aa27ef2; a test checks all three
new entry points exist at that exact pin. Profile evidence includes source
verification, loading, warmup, saving and wall-cost scopes.

Verification: full pinned offline CPU suite 279 passed, no skips, 22 warnings,
763.31 seconds; new workflow/notebook suite 18 passed, no skips, 2 warnings.
Later targeted profiling and final notebook-pin checks also passed. Includes
real tiny Qwen/PEFT measurement through all 196 hooks, source/norm/RNG controls,
calibration negative controls, interruption/resume, SFT reuse and corruption
rejection. No model weights were downloaded. Initial ambient environment
reported 239 passed/28 skips due to broken Torch; an isolated pinned environment
was used without modifying that installation. Evidence:
`artifacts/runs/diagnostics/pilot4-ticket07-software-validation/`.

Actual Qwen/CUDA numerical acceptance, capacity and scientific fixed-token
measurements remain pending Colab execution. Ticket 8 coefficients/KL and Ticket
9's overall report/handoff are not claimed complete.
