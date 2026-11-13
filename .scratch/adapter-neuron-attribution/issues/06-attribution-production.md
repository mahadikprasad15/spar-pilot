# 06: Stage A and B production, then selection freeze

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** Run Stage A for every arm × metric × trace set on discovery, freeze Stage B layers, run Stage B, then freeze and seal all Stage C/D conditions.

**Blocked by:** 05.

- [ ] Durable work units per (arm, metric, trace set); resume verified by hash; progress record.
- [ ] Layer freeze written and hashed before Stage B starts.
- [ ] Selection freeze: top 10 per (arm, metric, set), controls, bands, neuron rankings and masked-read conditions, written and hashed before any confirmation row is read.
- [ ] Optional prediction record, saved before Stage A results are inspected.
- [ ] Verified export after each stage.

## Comments
