# 08: Save per-token coefficients and fixed-context KL for both arms

**What to build:** Extend the fixed-input measurement product to retain the
additional raw quantities needed for rank-1 interpretation and later
distributional matching, for GRPO and the existing SFT checkpoints.

**Blocked by:** 07 — Measure GRPO writes on the existing fixed sequences.

**Status:** ready-for-agent

- [ ] Freeze explicit signed coefficient/scaling/factor conventions and
      prediction-position/KL reductions before collecting the new product.
- [ ] Save per-token input coefficients for every intended adapted module with
      IDs, positions, views, factor/scaling hashes and numerical provenance.
- [ ] Calculate full-vocabulary KL(tuned || untuned) at identical fixed
      prediction contexts in common measurement precision, using bounded
      chunked memory and independently checked shift/mask alignment.
- [ ] Demonstrate zero KL for identical distributions and correct finite
      positive values on known unequal distributions; independently verify
      KL direction and rank-1 coefficient reconstruction.
- [ ] Verify chunk/batch agreement at justified scales and prevent full-logit
      retention from exceeding declared memory constraints.
- [ ] Rerun only the additional necessary SFT forward measurements under a
      separately named product; old summaries cannot supply missing raw
      coefficients/KL. No SFT retraining or behavioural reevaluation occurs.
- [ ] Preserve existing artifact hashes and numerical policies; save the new
      raw quantities and summaries with atomic completion/resume markers.
- [ ] Explain that raw coefficients are factorization-dependent and KL is
      context-dependent rather than an unconditional policy-distance estimate.
- [ ] Guided notebook stages display the added workload/cost and distinguish
      supplemental measurement from already-completed Pilot 3 evidence.

## Comments

- October 7, 2026: approved as ticket 8. The dependency is implementation and
  validated input/source infrastructure; supplemental forward measurements
  remain separately identifiable from existing write artifacts.
