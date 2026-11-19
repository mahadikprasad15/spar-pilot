# 05: Implementation, policy and numerical checks

**Status:** resolved
**Type:** task
**Blocked by:** 01, 02, 03

**What to build:** check functions used both by CPU tests and by GPU qualification. They reuse the patterns of `full_fp64`, `branch_gates`, `neuron_gate` and `finite_difference_check`.

- [x] **Logits-processor resolution (CPU).**
  - Load the pinned `generation_config.json` (revision `989aa798…`; downloaded into the bundle in ticket 08; a recorded copy as the test fixture).
  - Build the processor and warper list that transformers 4.57.6 resolves from `_generate`'s explicit sampled arguments.
  - Assert that nothing modifies the logits.
  - When this passes, the provenance item moves from unverified to verified.
  - The catalog's `on-policy` label is provisional while its evidence is inferred. Do not treat
    that label or file-hash verification alone as proof that the original sampling distribution
    matches the score-function metric; report remaining source-run uncertainty explicitly.
- [x] **Surrogate finite differences.**
  - F(λ) = Σ w·M(λ), with frozen weights.
  - A central difference over gates, for chosen adapters and neurons, compared with the aggregated dJdλ.
  - The reference is always FP64 (`full_fp64`), because of the `seqlogp` scale.
  - The FP32 floor ulp(|M|)/ε is reported.
  - Production-1 tolerances apply. This is an implementation check, not a check of the estimator.
- [x] **Cached against teacher-forced.**
  - Per-token log p from left-padded prefill plus incremental KV-cache decoding of the saved tokens, compared with the right-padded full forward.
  - Report max, median and p99 of |Δ log p| per token, and |Δ M| per row.
  - Select short and long rows per arm and a capped row where available. Online GRPO has zero
    capped rows: report that absence; include the available offline capped row and a capped CPU
    fixture. Compare reconstructed cached decoding, not unavailable original generation scores.
- [x] **Numerics record** (`numerics.json`):
  - batch-invariance absolute and relative discrepancies;
  - identity errors, gross and net;
  - the per-component gross/net ratio distribution;
  - the finite-difference floors.

  These measurements inform the launch-record tolerances. They never loosen historical ones.
- [x] **Replay (report-only).**
  - Regenerate one sealed sampled batch with its recorded seed, batch membership and `_generate` settings.
  - Report token identity and the first divergence position.
  - Flag it for review only if it suggests a policy or checkpoint mismatch.
- [x] **No-op.**
  - The intervention context with an empty removal set gives logits bit-identical to intact, and identical greedy tokens on a subset.
  - The restored adapter hash equals the original after every edit.
- [x] **Batch invariance** of `seqlogp` attributions at the qualified token budget, at production-1's tolerance, with the absolute floor reported (lesson 2026-11-16).
- [x] **Tests (CPU):**
  - finite differences agree on the tiny model;
  - cached and forced agree on the tiny model;
  - the processor check fails if `repetition_penalty=1.1` is injected;
  - the no-op check fails if a weight is left modified;
  - mutations (weights not detached, a sign flip of the weights) make the finite-difference check fail.

## Comments

- 2026-11-19 (Claude): `pilot_eval/c_attribution_checks.py`. The processor check runs the real `grpo_gpu.GPUPreflight._generate` (with the pinned tokenizer from local artifacts and a tiny model with Qwen's vocabulary) and captures what `generate` resolves, on two bases: the pinned `generation_config.json` (whose defaults are T 0.7, top_k 20, top_p 0.8, repetition 1.1) and the fresh defaults that `backend.load_hf_backend` installs. Both resolve to an empty processor list, with T 1.0, top_k 0, top_p 1.0, repetition 1.0 and no-repeat 0. This verifies the code path on this branch; `grpo_gpu.py` and `backend.py` were not hashed by the source runs, so it does not prove the bytes that ran. Injecting a repetition penalty, temperature or top-k makes it fail. The no-op check takes an intervention context factory; ticket 06 supplies the real one. Replay and real-arm cached-against-forced need the GPU (rehearsal, ticket 11). "Weights not detached" has no direct analogue here: weights are plain floats by construction and the instrument refuses non-finite or misshapen coefficients; the weight mutations tested are a sign flip and a one-row shift.

- 2026-11-18 (round 1): processor resolution, numerics floors and the no-op check were added. The cached-against-forced tolerance is no longer pre-set; it is decided after the rehearsal (F4, decision 9).

## Answer

`pilot_eval/c_attribution_checks.py`; tests in `tests/test_c_attribution_checks.py` (15 pass). On the tiny model: surrogate finite differences (5 adapters, 3 neurons, FP64 reference, ε 1e-3) pass production-1 tolerances with median relative error below 1e-3, and fail for both weight mutations; cached and forced agree to below 1e-5 per token, and below 1e-4 on a 1,024-token capped sequence; `seqlogp` batch invariance passes 1e-4; the no-op check fails when a weight is left modified. The pinned `generation_config.json` is copied as a fixture (sha256 `e558847a…`). The provenance item for resolved logits processors can move to verified for the current code path; the catalog's evidence status stays `inferred` until the user decides, since the source runs did not hash the generation code.
