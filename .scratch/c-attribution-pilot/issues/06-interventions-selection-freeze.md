# 06: Interventions, selection and freeze

**Status:** needs-triage
**Type:** task
**Blocked by:** 03, 04

**What to build:** in `attribution_ablation.py` and `c_attribution.py`.

- [ ] `neuron_read_removal(model, {layer: [j, ...]})`: zero a_j in that layer's `down_proj` lora_A (save and restore), plus an equivalent hook form. Equals `neuron_gate(g = 0)` for one neuron, and is exact for several neurons across layers.
- [ ] Adapter removal reuses `removal` / edited checkpoints. An in-memory lora_B zeroing context equals both on logits.
- [ ] Selection per arm: top-10 adapters by dJdλ; matched adapter controls via `matched_controls` (write = ‖s·b‖ · mean|c| over these rows); top-10 and top-64 neuron reads pooled across the 28 `down_proj` reads (open decision 1); matched neuron controls (same layer, term size ±20%, outside the top-64, seeded; open decision 3); the most harmful adapters reported only.
- [ ] `freeze/<arm>.json` sealed and hashed **before** any generation unit exists; generation refuses to start without it (lesson 2026-11-13: freeze before outcomes are visible).
- [ ] Tests: equality of every intervention form on logits (tiny model, prefill and decode with a cache); controls exclude targets, are deterministic, and are unavailable rather than loosened; the freeze refuses existing generation outputs.

## Comments

## Answer
