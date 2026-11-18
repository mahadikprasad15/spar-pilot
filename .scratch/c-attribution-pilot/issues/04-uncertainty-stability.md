# 04: Uncertainty and stability

**Status:** needs-triage
**Type:** task
**Blocked by:** 03

**What to build:** in `c_attribution.py`, descriptive uncertainty from the sealed per-response attributions (CPU only; no recomputation of gradients).

- [ ] Question bootstrap (10,000, seed 1234 **[proposed]**), recomputing leave-one-out weights within each resample: percentile intervals for every adapter's and neuron read's dJdλ, and ranks.
- [ ] Split-half by draws (0–3 vs 4–7, leave-one-out within halves) and by question ID parity: Spearman of the 196-vectors, top-10 overlap, sign agreement in the top 20.
- [ ] Report J̄, informative-question counts, and the top-10 share of Σ|dJdλ|.
- [ ] Tests: deterministic under a seed; a planted-signal fixture gives high split-half agreement; a pure-noise fixture gives low agreement; intervals cover the planted value at the nominal rate in a small simulation.

## Comments

## Answer
