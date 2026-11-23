# 14: Causal checks (proposed)

**Status:** needs-info
**Type:** task
**Blocked by:** 09

**Needs the researcher's OK (D23) and its own launch record.** Not part of the minimal pilot. Separate from the C-attribution effort's held-out ticket 14.

- [ ] Revert all survivors: validation CE and accuracy, plus cohort150 greedy, should return to untuned within round-off.
- [ ] Single reverts on validation CE for an importance ranking (layer blocks only if more than 512 survivors).
- [ ] Block reverts: per layer and top-k by importance (k = 1, 4, 16, 64); greedy accuracy for 8 blocks.
- [ ] Same-layer random control: move each survivor's change onto a random non-survivor in the same layer, 20 draws.
- [ ] Red/green: spec test 17.
- [ ] Estimate: about 1–1.5 h per arm on one A100-80GB (arithmetic); ticket 07 times one revert.

## Comments

## Answer
