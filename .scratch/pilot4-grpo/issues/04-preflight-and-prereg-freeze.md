# 04: Preflight the target GPU and freeze the resolved protocol

**What to build:** Turn the audited plan, baseline evidence and validated
algorithm into a reviewed scientific execution plan with real capacity,
determinism, cost and preregistration evidence.

**Blocked by:** 02 — Save the untuned training-cohort sampling baseline;
03 — Validate the GRPO algorithm on offline tiny models.

**Status:** ready-for-agent

- [ ] Verify approved FP32, 10,000-resample paired item bootstrap (seed 42),
      consecutive pre-clip gradient cosine and the two-seed learning controls;
      require remaining scorer/loss/final-cap/generation-policy/monitor/margin
      resolutions and show provenance for every value.
- [ ] Save two disposable two-step runs on the target GPU with controlled RNG,
      compare their declared reward/rollout/state evidence, and check finite
      loss/gradients, zero initialization and frozen weights.
- [ ] Verify zero-adapter greedy outputs against the fresh matched untuned
      baseline, not against historical accuracy 0.640.
- [ ] Profile final-length generation/backward capacity and total stage costs,
      including hashing, transfer and saving overhead; make memory batch
      settings compatible with the approved algorithmic batch arithmetic.
- [ ] Present measured timings/memory and projected stage workloads without
      treating the superseded BF16 document price/time estimates as FP32
      guarantees; include redo cost of checkpoint-only training recovery.
- [ ] Freeze ordered groups, numerical/runtime settings, final cap, algorithm,
      scorer and a user-authored dated preregistration with source hashes.
- [ ] Reject an unresolved, changed or unreviewed full-run request without
      overwriting previous evidence; CPU tests simulate these gate outcomes.
- [ ] Provide a guided notebook review/freeze stage and preserve disposable
      test state separately from the full training run.

## Comments

- October 7, 2026: approved as ticket 4. This ticket delivers the mechanism for
  recording decisions and GPU evidence; it does not authorize silently picking
  the proposed ten-point non-inferiority margin.
