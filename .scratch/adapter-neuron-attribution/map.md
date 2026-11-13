# Adapter and neuron attribution — map

## Notes
- First pass only; deferred items are listed at the end of `spec.md`.
- Reuses the joint audit's inventory-v2, sequences-v2, FP32 policy and Pod/export chain.
- Stage 0 heatmaps = write size; this study = necessity (attribution + real ablation).
- Neuron attribution = removing a neuron's term from a `down_proj` adapter's read, not a neuron ablation.
- Proxies (M_acc, M_len, KL) drive attribution and Stage C; v3 accuracy and length are measured only in Stage D.

## Decisions so far
- 2026-11-13 (user): shrink to a first pass — merged A+B pass, base-correct traces only, top-5 singles + top-10 group in Stage C, one 64-neuron read per arm, generation for gold-sft and online-grpo only, no drift metric.

- 2026-11-13 (user): GPU is underused; measure utilization (busy %, MFU, tokens/s) in every stage and size batches by sweep. Run tiers: Colab bug-finding notebook, then Pod qualification and production.

- 2026-11-13: review round 1 (Codex) answered in `review-response.md`; draft v2 of spec, plan and tickets (rehearsals on calibration rows only; z recomputed in a forward-only pass; trace log-likelihood and stop pull replace M_acc/M_len; per-arm gates, equivalence margin, Holm families; FP32 throughout; tickets renumbered 01–10).

- 2026-11-13: review round 2 (Codex) recommends CPU implementation without another design cycle; three corrections (frozen-model gradient path, exact stop IDs, uncertainty in primary decisions) made in draft v3. User approved CPU implementation of 01–05.

## Fog
- Actual utilization of the joint audit's run (no NVML record exists).
- How many base-correct rows survive.
- Whether a ≈ a₀ in each arm's top `down_proj` module.
- Whether first-order estimates track real removal of whole adapters (gate).
- HF FP32 generation time for 8 × 150 items.
