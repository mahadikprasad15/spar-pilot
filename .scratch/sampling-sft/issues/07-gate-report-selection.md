# 07: Produce the descriptive gate report and explicit training selection

**What to build:** CPU commands restore eligible data and HF score evidence to produce gold/k-arm plain-NLL, length, format, leakage, acceptance and coverage views. The common-cohort paired bootstrap uses approved estimands, reports absent baseline/offset honestly, and creates a source-bound request for training k0/k5. No automatic merge or training launch occurs.

**Blocked by:** 06.

**Status:** claimed

**Type:** task

## Acceptance criteria

- [ ] Restore verified gold/k0/k2/k5 eligible products and emit per-arm plus three-arm-common-cohort descriptive tables/plots with source identities.
- [ ] Collect/reuse verified HF FP32 gate score evidence under the appropriate budgeted model-work route; keep subsequent statistical/report commands CPU-only.
- [ ] Compute stop-inclusive answer NLL then equal-problem means; length median/p90, format, coverage/retry/pass/unfinished rates and acceptance by step/cut decile retain explicit denominators.
- [ ] Use10000 whole-problem paired bootstrap resamples with seed42 and percentile95% intervals: difference of arm medians recomputed per resample for length, difference of means for NLL.
- [ ] Report literal calculator-marker rate and response word4-gram overlap with cleaned gold, with predeclared word normalization and undefined status for fewer than4 words.
- [ ] Omit offset/baseline columns and state relative sharpening not measured. No automatic merge, equivalence or causal mediation claim.
- [ ] Create a source-bound explicit k0/k5 training-selection/approval product with measured budget and coverage flags. k2 stays unselected without separate approval.
- [ ] Hand-computable public-workflow fixtures reconcile rows/plots/intervals and reject stale parents or approvals.

## Testing and evidence

Hand-computable public workflows check stop-inclusive equal-problem NLL, difference-of-medians resampling, undefined short-text leakage, whole-problem pairing, report reconciliation and stale approval rejection. Gate scoring model work is separately budgeted; CPU statistics never require CUDA.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.
