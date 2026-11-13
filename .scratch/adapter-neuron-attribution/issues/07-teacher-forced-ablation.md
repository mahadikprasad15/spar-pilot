# 07: Stage C teacher-forced ablation on confirmation

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** Score every frozen Stage C condition on confirmation rows, per example.

**Blocked by:** 06.

- [ ] Singles, cumulative top 1..10, random controls, bands (necessity and sufficiency), neuron masks and constant c̄, per arm × metric.
- [ ] ΔM_acc, ΔM_len, KL_full per example; Drift on background for band and group conditions.
- [ ] Per-example spot check: real single-example ablation for the top 3 adapters on 10 examples.
- [ ] Resumable units; verified export.

## Comments
