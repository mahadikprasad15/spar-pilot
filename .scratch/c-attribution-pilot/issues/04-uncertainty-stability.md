# 04: Uncertainty and stability

**Status:** needs-triage
**Type:** task
**Blocked by:** 03

**What to build:** descriptive uncertainty in `c_attribution.py`, computed from the sealed per-question S_q. It runs on CPU only, with no recomputation of gradients. **Nothing here is a gate.**

- [ ] **Question split.**
  - Two halves of 75 question IDs, drawn with a recorded seed (proposed 20261118).
  - The same split for both arms, with the ID lists sealed.
  - From each half's S_q, report:
    - adapters: Spearman of the 196-vectors, top-10 overlap, and sign agreement within each half's top 20;
    - global top-64 reads: overlap, sign agreement, and the layer distribution;
    - hot-layer and per-layer top-64 reads: overlap and sign agreement per layer, plus the equal-layer mean.
- [ ] **Adapter bootstrap.**
  - 10,000 resamples of whole questions, seed 1234 (proposed).
  - The same resampled indices for both arms.
  - Outputs: percentile intervals for dJdλ, rank intervals, and paired intervals for online − offline.
- [ ] **Neuron bootstrap.** Streamed: a count matrix [B, 150] times S chunked over columns. Only percentiles are kept, never the resamples. Work in this order:
  1. every selected read and control (after the freeze);
  2. per-layer sums;
  3. all reads, only if the benchmark shows it fits the cap.
- [ ] **Benchmark mode.** Records the wall time and peak RSS for each of the three tiers on a given S shape, for the rehearsal and for the launch record.
- [ ] **Also reported:** J̄; informative-question counts; the top-10 adapter share of gross Σ|dJdλ|.
- [ ] **Labels.** All intervals are labelled "exploratory, pointwise; selected on this cohort".
- [ ] **Tests:**
  - deterministic under a seed;
  - the streamed result equals the unchunked result;
  - the question bootstrap on S_q equals recomputing the leave-one-out weights inside resamples, on a fixture;
  - a planted-signal fixture gives high split agreement, and a pure-noise fixture gives low agreement;
  - intervals cover the planted value at roughly the nominal rate in a small simulation.

## Comments

- 2026-11-18 (round 1): the user chose a seeded 75/75 question split with no gate. The v1 draw split is dropped, pending the user's confirmation (spec open decision 7). The neuron bootstrap is chunked and benchmarked (F2, F5, decisions 7–8).

## Answer
