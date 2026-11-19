# 06: Interventions, selection, matched controls and freeze

**Status:** needs-triage
**Type:** task
**Blocked by:** 03, 04

**What to build:** these go in `attribution_ablation.py` and `c_attribution.py`. The existing `matched_controls` stays callable for production-1 reproducibility, unchanged.

- [ ] **Neuron-read removal.**
  - `neuron_read_removal(model, {layer: [j, ...]})` zeroes a_j in that layer's `down_proj` lora_A, with save, restore and a restored-hash check.
  - It also has an equivalent hook form.
  - It equals `neuron_gate(g = 0)` for one read, and is exact for many reads across layers, at prefill and at decode with a cache.
- [ ] **Adapter removal.** An in-memory lora_B zeroing context that equals both `removal` and the edited checkpoint on logits, with a restored-hash check.
- [ ] **Group-wise matcher.** `match_group(targets, size, strata, *, exclude, tolerance, seed)`:
  1. For each stratum, build the bipartite graph of targets and eligible candidates. A candidate is eligible if it is in the same stratum, not excluded, and |size − target| ≤ tolerance·|target|.
  2. Permute the candidate order with the seed.
  3. Run a deterministic maximum-cardinality matching (augmenting paths in target-rank order).

  The control is unavailable only if the maximum matching is smaller than the group. The output then names:
  - the deficient targets;
  - the eligible-candidate count per target;
  - the largest feasible matching.

  It never loosens eligibility. It reports the match ratios (min, median, max).
- [ ] **Regression fixture** (review round 1, F3):
  - targets 0 and 1 with writes 1.00 and 1.40;
  - candidates 2 and 3 with writes 1.16 and 0.85;
  - same module type; tolerance 0.20.

  The test must fail (red) against the existing sequential `matched_controls` first. The new matcher then returns 0→3, 1→2.

  Also test a truly infeasible group, which reports the deficient target; and determinism under a seed.
- [ ] **Selection per arm** (spec: Selection):
  - `top10-adapters`;
  - `global64-reads` across all 28 × 8,960 reads;
  - `hot64-reads` (64 per hot layer);
  - `layer64-reads` (64 per `down_proj` layer).

  Each group:
  - is **positive dJdλ only**;
  - has ties broken by layer, then module or index;
  - on a shortfall, is frozen at its actual positive size with the control matched to that size and the shortfall recorded; never padded, never dropped [user, 2026-11-19].

  Each group gets its own matched control:
  - adapters: same module type, write ±20%;
  - reads: same layer, read size ±20%;
  - exclusion: all same-kind targets of the arm.

  Infeasible layers in `layer64` never block other groups. If any layer is infeasible, the `layer64-reads` target still runs unchanged and `match-layer64` is recorded as unavailable, naming the deficient layers and targets [user, 2026-11-19]. The most harmful adapters are frozen and reported only.
- [ ] **Measures.** Adapter write is ‖s·b‖·mean|c|, and read size is |a_j|·mean|z_j|. Both come from ticket 01's accumulations over the arm's 1,200 rows, all unpadded positions, token-weighted.
- [ ] **Hot layers computed by code.** `hot_layers(source, sha256, *, context, step, k, module, arms, combine)` reads the effective-write table, refuses a hash mismatch, ranks each arm's modules by write, keeps the layers whose `down_proj` is in the top k, and combines them ("union" here). Test: on the real file with `gsm8k_solution`, step 64, k 10, union, it returns {11, 15, 16, 24}; it also writes the k-sensitivity table (k = 5, 10, 15, 20) for the report.
- [ ] **Hot-layer provenance.** The freeze input binds:
  - the layer IDs;
  - the source file path and sha256;
  - the measurement, context and k.

  It is sealed in the launch record **before** aggregation outputs are readable. Selection refuses a hot-layer list that is missing from the launch record or that was computed from J.
- [ ] **Freeze.**
  - `freeze/<arm>.json` holds the selections, controls, match diagnostics, shortfalls, split IDs, predicted removal effects for groups and controls, and seeds.
  - It is sealed and hashed **before** any generation unit exists. Generation refuses to start without it (lesson 2026-11-13).
- [ ] **Tests:**
  - every intervention form is equal on logits;
  - controls exclude targets and are deterministic;
  - unavailable is reported only when infeasible;
  - the freeze refuses existing generation outputs;
  - the hot-layer binding refuses a late or J-derived list.

## Comments

- 2026-11-18 review: the existing sequential `matched_controls` can return an unavailable group even when a complete valid matching exists. Reproduced on a two-target synthetic fixture; no prior real-study failure is claimed. Use seeded whole-group feasibility/matching without relaxing eligibility, and add the regression case. See [review round 1](../review-round-1.md#f3-confirmed-false-unavailable-outcome-in-the-existing-matcher). Fix not yet implemented or verified.
- 2026-11-18 (response round 1): scope revised to the three top-64 read groups, with no top-10 read ablations [user]. The matcher above is specified. The hot-layer list content is open (spec open decision 1).
- 2026-11-19: hot layers {11, 15, 16, 24} confirmed, derived by `hot_layers` with the union rule. Infeasible `layer64` control: run the target, report the control unavailable.

## Answer
