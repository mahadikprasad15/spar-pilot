# Adapter and neuron attribution — map

## Notes
- Follows the joint neuron and adapter audit; reuses its inventory-v2, sequences-v2, FP32 policy and Pod/export chain.
- Stage 0 heatmaps = write size; this study = necessity (attribution + real ablation).
- Neuron attribution here is neuron-to-adapter only (the term a_j z_j in c_t), not a neuron ablation.

## Decisions so far
- (none approved) See spec "Open decisions".

## Fog
- How many base-correct rows survive in discovery and held-out.
- Whether a ≈ a₀ in every selected `down_proj` module (post hoc Stage 0 output).
- Whether first-order estimates track real ablation for whole adapters (05 decides).
- HF generation time in FP32 for ~40 conditions × 150 items.
