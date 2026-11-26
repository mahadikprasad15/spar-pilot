# 04: Uncertainty and stability

**Status:** resolved
**Type:** task
**Blocked by:** 03

**What to build:** descriptive uncertainty in `c_attribution.py`, computed from the sealed per-question S_q. It runs on CPU only, with no recomputation of gradients. **Nothing here is a gate.**

- [x] **Question split.**
  - Two halves of 75 question IDs, drawn with a recorded seed (proposed 20261118).
  - The same split for both arms, with the ID lists sealed.
  - From each half's S_q, report:
    - adapters: Spearman of the 196-vectors, top-10 overlap, and sign agreement within each half's top 20;
    - global top-64 reads: overlap, sign agreement, and the layer distribution;
    - hot-layer and per-layer top-64 reads: overlap and sign agreement per layer, plus the equal-layer mean.
- [x] **Draw split** [user, 2026-11-19]. Draws 0–3 against 4–7, with within-half leave-one-out weights (n = 4). Adapters from the stored per-response rows; reads from ticket 03's half-sums. The same summaries as the question split, labelled "noisier: 4 draws per question per half". No bootstrap.
- [x] **Adapter bootstrap.**
  - 10,000 resamples of whole questions, seed 1234 (proposed).
  - The same resampled indices for both arms.
  - Outputs: percentile intervals for dJdλ, rank intervals, and paired intervals for online − offline.
- [x] **Neuron bootstrap.** Streamed: a count matrix [B, 150] times S chunked over columns. Only percentiles are kept, never the resamples. Work in this order:
  1. every selected read and control (after the freeze);
  2. per-layer sums;
  3. all reads, only if the benchmark shows it fits the cap.
- [x] **Benchmark mode.** Records the wall time and peak RSS for each of the three tiers on a given S shape, for the rehearsal and for the launch record.
- [x] **Also reported:** J̄; informative-question counts; the top-10 adapter share of gross Σ|dJdλ|.
- [x] **Labels.** All intervals are labelled "exploratory, pointwise; selected on this cohort".
- [x] **Tests:**
  - deterministic under a seed;
  - the streamed result equals the unchunked result;
  - the question bootstrap on S_q equals recomputing the leave-one-out weights inside resamples, on a fixture;
  - a planted-signal fixture gives high split agreement, and a pure-noise fixture gives low agreement;
  - intervals cover the planted value at roughly the nominal rate in a small simulation.

## Comments

- 2026-11-19 (Claude): implemented in `pilot_eval/c_attribution_stability.py` (a sibling module, to keep `c_attribution.py` small). Definitions: "top k" ranks by dJdλ, largest first, as selection does; sign agreement uses each half's k largest |dJdλ| and reports the share of their union with matching signs. Spearman uses average ranks (no scipy dependency).

- 2026-11-18 (round 1): the user chose a seeded 75/75 question split with no gate. The v1 draw split is dropped, pending the user's confirmation (spec open decision 7). The neuron bootstrap is chunked and benchmarked (F2, F5, decisions 7–8).
- 2026-11-19: the user kept the draw split in addition to the question split.

## Answer

`pilot_eval/c_attribution_stability.py`; tests in `tests/test_c_attribution_stability.py` (10 pass): seeded split, Spearman with ties, overlap and sign agreement by hand, planted signal against noise for both splits, streamed equal to unchunked, bootstrap-on-S equal to recomputed weights, coverage 0.88–0.99 in a 200-trial simulation, paired arms, tiers with benchmark. Benchmark on this Mac, 10,000 resamples, real width: two layers [150, 2, 8,960] take 6.4 s for all reads (157 MiB traced peak), so all 28 layers are about 90 s; selected 256 reads 0.1 s. The all-reads tier is therefore feasible on CPU.
