# Response to review round 2

Disposition of `review-round-2.md` (Codex, reviewing `b7a2eef`). All points adopted; claims verified against the repo before adopting. Spec, plan and tickets are now draft v3. The user approved CPU implementation of tickets 01–05 (2026-11-13), which are marked `ready-for-agent`; no GPU run, value or budget is approved.

## Partly resolved round-1 points

- **5 → see 24.**
- **7 → see 23.**
- **8. Adopt.** Pass 1 also saves per example each adapter's Σ_t |c_t| and token count and each `down_proj` adapter's signed Σ_t c_t, so matching and c̄ need no extra run (spec §Stage A+B, ticket 03).
- **9. Adopt.** The gate is labelled a limited check on selected removals; a pass does not establish the ordering or magnitudes of all 196 effects (spec §Statistics, ticket 10).
- **11, 12 → see 25.**
- **20. Adopt; my claim was wrong.** Verified: `scripts/neuron_audit_adapter_init_check.py` computes raw `cos(a_T, a_0)` and drift with no sign alignment. The spec now quotes it only as a historical, gauge-dependent description (meaningful because both factors share one training parameterization), or not at all.

## New points

- **23. Adopt.** Verified by reasoning: frozen weights and integer inputs give no autograd graph, and a constant 1.0 gate adds none. Spec §Stage A+B now requires `requires_grad_(True)` on the embedding output with all weights frozen, and one `torch.autograd.grad` call per metric returning fresh tensors (no `.grad` accumulation). Ticket 03 tests: the frozen model without that input has no `grad_fn`; each metric matches its own finite differences; metric order does not change results.
- **24. Adopt.** Verified: `pilot_eval/backend.py` passes only `tokenizer.eos_token_id`; the pinned tokenizer has EOS `<|im_end|>` = 151645 and PAD `<|endoftext|>` = 151643. Stop pull uses the stop IDs sealed from the decoder config actually used (pilot: {151645}); a different set is refused unless labelled a decoding change. Ticket 01 tests the single-ID case and the refusal.
- **25. Adopt.** Primary 1 now uses the joint bootstrap of r = |D_L| / |D_A| on the same resampled questions: the reference D_A must have an interval excluding 0 (else inconclusive), and equivalence needs the one-sided 95% upper bound of r < 0.25; contradiction needs its one-sided lower bound ≥ 0.25. Primary 3's tests are the two paired R contrasts; sparse/dense classification is descriptive and uses interval bounds (lower bound ≥ 0.70; upper bound < 0.30 in every arm), never point estimates. Holm runs over primary 1's single conjunctive decision (larger component p-value), primary 2's four differences and primary 3's paired contrasts, per metric. Ticket 10 adds fixtures where a point-estimate rule would claim equivalence or sparsity and the interval rule does not.
