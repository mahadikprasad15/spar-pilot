# 06: Attribution run and selection freeze

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 05b

**What to build:** Run the merged pass for all six arms on discovery, then freeze every Stage C and D condition.

- [ ] One work unit per arm; resume verified by hash; progress record.
- [ ] Optional prediction record saved before results are inspected.
- [ ] Selection freeze (top 10, top 5, matched randoms, top `down_proj` adapter and its neuron rankings, per arm and metric) written and hashed before any confirmation row is read.
- [ ] GPU utilization summary per unit (from `attribution_gpu.py`) saved with the outputs.
- [ ] Verified export.

## Comments
