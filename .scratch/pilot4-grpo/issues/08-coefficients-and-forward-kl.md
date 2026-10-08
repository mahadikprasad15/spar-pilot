# 08: Save per-token coefficients and fixed-context KL for both arms

**What to build:** Extend the fixed-input measurement product to retain the
additional raw quantities needed for rank-1 interpretation and later
distributional matching, for GRPO and the existing SFT checkpoints.

**Blocked by:** 07 — Measure GRPO writes on the existing fixed sequences.

**Status:** resolved

- [x] Freeze explicit signed coefficient/scaling/factor conventions and
      prediction-position/KL reductions before collecting the new product.
- [x] Save per-token input coefficients for every intended adapted module with
      IDs, positions, views, factor/scaling hashes and numerical provenance.
- [x] Calculate full-vocabulary KL(tuned || untuned) at identical fixed
      prediction contexts in the approved common FP32 measurement precision,
      using bounded
      chunked memory and independently checked shift/mask alignment.
- [x] Demonstrate zero KL for identical distributions and correct finite
      positive values on known unequal distributions; independently verify
      KL direction and rank-1 coefficient reconstruction.
- [x] Verify chunk/batch agreement at justified scales and prevent full-logit
      retention from exceeding declared memory constraints.
- [x] Rerun only the additional necessary SFT forward measurements under a
      separately named product; old summaries cannot supply missing raw
      coefficients/KL. No SFT retraining or behavioural reevaluation occurs.
- [x] Preserve existing artifact hashes and numerical policies; save the new
      raw quantities and summaries with atomic completion/resume markers.
- [x] Explain that raw coefficients are factorization-dependent and KL is
      context-dependent rather than an unconditional policy-distance estimate.
- [x] Guided notebook stages display the added workload/cost and distinguish
      supplemental measurement from already-completed Pilot 3 evidence.

## Comments

- October 7, 2026: approved as ticket 8. The dependency is implementation and
  validated input/source infrastructure; supplemental forward measurements
  remain separately identifiable from existing write artifacts.

- October 8, 2026: user approved raw signed A·x on adapted module inputs;
  next-target view membership for KL, excluding target zero/padding/out-of-sequence;
  raw values plus token/equal-example means; existing public-workflow test seam.
  Conventions recorded in docs/adr/0008-pilot4-token-coefficients-and-kl.md.

## Answer

Implemented the supplemental coefficient/KL workflow, six CLI commands, source-hash
guards, resumable per-batch artifacts and guided Colab sections 28–33. See
`docs/pilot4-token-measurements.md` and ADR 0008 for the approved conventions.

Software validation: 290 passed, no skips, 22 warnings (887.27 seconds) in the
pinned offline CPU environment; the additional source/engine regression group
passed 8 tests and notebook pin checks passed 3. Durable evidence is in
`artifacts/runs/diagnostics/pilot4-ticket08-software-validation/`.

This resolves implementation and CPU validation. Real checkpoint/CUDA profile,
freeze and measurement gates still must pass in Colab before scientific use;
no real-model GPU measurement has been claimed. Ticket 9 remains open.
