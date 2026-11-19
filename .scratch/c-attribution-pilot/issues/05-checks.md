# 05: Implementation, policy and numerical checks

**Status:** needs-triage
**Type:** task
**Blocked by:** 01, 02, 03

**What to build:** check functions used both by CPU tests and by GPU qualification. They reuse the patterns of `full_fp64`, `branch_gates`, `neuron_gate` and `finite_difference_check`.

- [ ] **Logits-processor resolution (CPU).**
  - Load the pinned `generation_config.json` (revision `989aa798…`; downloaded into the bundle in ticket 08; a recorded copy as the test fixture).
  - Build the processor and warper list that transformers 4.57.6 resolves from `_generate`'s explicit sampled arguments.
  - Assert that nothing modifies the logits.
  - When this passes, the provenance item moves from unverified to verified.
  - The catalog's `on-policy` label is provisional while its evidence is inferred. Do not treat
    that label or file-hash verification alone as proof that the original sampling distribution
    matches the score-function metric; report remaining source-run uncertainty explicitly.
- [ ] **Surrogate finite differences.**
  - F(λ) = Σ w·M(λ), with frozen weights.
  - A central difference over gates, for chosen adapters and neurons, compared with the aggregated dJdλ.
  - The reference is always FP64 (`full_fp64`), because of the `seqlogp` scale.
  - The FP32 floor ulp(|M|)/ε is reported.
  - Production-1 tolerances apply. This is an implementation check, not a check of the estimator.
- [ ] **Cached against teacher-forced.**
  - Per-token log p from left-padded prefill plus incremental KV-cache decoding of the saved tokens, compared with the right-padded full forward.
  - Report max, median and p99 of |Δ log p| per token, and |Δ M| per row.
  - Row selection must include short, long and capped rows.
- [ ] **Numerics record** (`numerics.json`):
  - batch-invariance absolute and relative discrepancies;
  - identity errors, gross and net;
  - the per-component gross/net ratio distribution;
  - the finite-difference floors.

  These measurements inform the launch-record tolerances. They never loosen historical ones.
- [ ] **Replay (report-only).**
  - Regenerate one sealed sampled batch with its recorded seed, batch membership and `_generate` settings.
  - Report token identity and the first divergence position.
  - Flag it for review only if it suggests a policy or checkpoint mismatch.
- [ ] **No-op.**
  - The intervention context with an empty removal set gives logits bit-identical to intact, and identical greedy tokens on a subset.
  - The restored adapter hash equals the original after every edit.
- [ ] **Batch invariance** of `seqlogp` attributions at the qualified token budget, at production-1's tolerance, with the absolute floor reported (lesson 2026-11-16).
- [ ] **Tests (CPU):**
  - finite differences agree on the tiny model;
  - cached and forced agree on the tiny model;
  - the processor check fails if `repetition_penalty=1.1` is injected;
  - the no-op check fails if a weight is left modified;
  - mutations (weights not detached, a sign flip of the weights) make the finite-difference check fail.

## Comments

- 2026-11-18 (round 1): processor resolution, numerics floors and the no-op check were added. The cached-against-forced tolerance is no longer pre-set; it is decided after the rehearsal (F4, decision 9).

## Answer
