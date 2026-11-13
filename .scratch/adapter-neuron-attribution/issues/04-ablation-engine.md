# 04: Ablation engine and condition builder

**Status:** claimed (2026-11-13) — CPU implementation approved by the user 2026-11-13 (spec draft v3). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 02

**What to build:** `pilot_eval/attribution_ablation.py`.

- [ ] Removal via g = 0 for any set; equal to an edited checkpoint (chosen b zeroed, hashed).
- [ ] Matched controls per target: same module type, write tolerance, without replacement, seeded, excluding both metrics' top-10 sets; group = union of per-target matches; unavailable recorded, never loosened.
- [ ] Natural 64-term read (no rescaling) by a given neuron order; constant read c̄ at every unpadded position.
- [ ] Builder reads only a sealed freeze; refuses confirmation- or cohort-derived inputs.
- [ ] Generation from edited checkpoints with a frozen batch manifest (left padding, per-sequence stop).
- [ ] Tests: all-neuron read == full; empty removal == full; constant read writes s·b·c̄; checkpoint == gate; sampler exclusions, determinism, unavailable path; manifest batch order reproduced.

## Comments
