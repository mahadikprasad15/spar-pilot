# 05: Implementation and policy checks

**Status:** needs-triage
**Type:** task
**Blocked by:** 01, 02, 03

**What to build:** check functions used by CPU tests and by GPU qualification (reusing `full_fp64`, `branch_gates`, `neuron_gate`, `finite_difference_check` patterns).

- [ ] **Surrogate finite differences:** F(λ) = Σ w·M(λ) with frozen weights; a central difference over gates for chosen adapters and neurons vs the aggregated dJdλ; FP64 reference model (lessons 06a r3 and diagnostic 4: never an FP32 finite difference near FP32 resolution; `model.double()` alone is not FP64 for Qwen2). Report it as an implementation check, not a check of the estimator.
- [ ] **Cached vs teacher-forced:** for given rows, per-token log p from left-padded prefill plus incremental KV-cache decoding of the saved tokens (the generation path) vs the right-padded full forward (the instrument path); report max and median |Δ log p| and |Δ M| per row.
- [ ] **Replay (report-only):** regenerate one sealed sampled batch with its recorded seed, batch composition and `grpo_gpu._generate` settings; report token identity and the first divergence position.
- [ ] **Batch invariance** of `seqlogp` attributions at the qualified token budget; note the rounding-floor caveat (lesson 2026-11-16: relative to the largest attribution).
- [ ] Tests (CPU): finite differences agree on the tiny model; cached vs forced agree within tolerance on the tiny model; mutations (detached weights not frozen, a weights sign flip) make the finite-difference check fail.

## Comments

## Answer
