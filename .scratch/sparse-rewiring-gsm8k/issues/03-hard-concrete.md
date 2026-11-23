# 03: Hard-concrete gate, shared mask and survivor export

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 02
**Group:** B

Gate equations, expected-active, deterministic gate, freeze to binary, the D14 per-group `u` cache (decided) (one draw per 4 examples, reused under checkpoint recompute), and the survivor export.

- [ ] Red/green: spec tests 5, 6 and 10a.
- [ ] Export: zero-based `(layer, neuron)`, `logα`, `z_det`, binary state, gauge-invariant norms; three counts per layer.
- [ ] Sealed with `attribution_units.seal_run` / `_encode`.

## Comments

## Answer
