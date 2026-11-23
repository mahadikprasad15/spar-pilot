# C-attribution rehearsal: measurements and production proposals (ticket 11 → 12)

Source: clean HF restores of both rehearsal runs, all four exports complete and verified. The restores are in
`artifacts/imports/c-attribution-rehearsal-pod-{1,2}[-final]/`; the structured numbers are in
`artifacts/imports/measurements/c-rehearsal-pod-{1,2}.json`, written by `scripts/c_attribution_measurements.py`.
These are rehearsal numbers (4 questions per arm), not results.

## What the rehearsal established

- **Qualification:** all 12 checks pass and the sign-off says `accept`, in both runs.
- **Interruption and resume:** run 2 was killed with 1 attribution unit sealed. The local root was deleted, the
  new attempt took over (`take-over`, heartbeat age 1,394 s), and the restored `online-grpo/unit-0000` was
  reused: it was sealed at 04:39:48, before the kill, and the other three units were sealed after 05:03:14,
  when attempt-002's attribution started.
- **Reproducibility across the two runs, including the resumed one:**
  - all 56 greedy answers are token-identical;
  - all 4 attribution unit sums are bit-identical (max difference 0.0);
  - all 8 freezes select the same targets.
- **Backups:** volume copy and HF export verified and complete. The export took 9.3 s for 53 MB, in one attempt.
- **One failure,** fixed in `69554a0`: on a real stop, the closing records were never written (lesson
  2026-11-22).

## Floors against the tolerances

| Quantity | Online | Offline | Tolerance (production-1) | Margin |
|---|---|---|---|---|
| FD median relative error | 5.9e-6 | 4.3e-6 | 0.05 | ×8,500 |
| FD max relative error | 2.4e-5 | 6.2e-6 | 0.20 | ×8,300 |
| Neuron-sum identity (relative to gross) | 1.7e-7 | 1.0e-7 | 1e-4 | ×570 |
| Batch invariance (fraction of max \|attr\|) | 2.2e-5 | **4.2e-5** | 1e-4 | **×2.4** |
| Peak memory fraction (token budget 8192) | 0.534 | 0.508 | 0.80 | 45 GB of 85 GB |
| Cached vs forced, per-token abs, max | 1.21e-4 | 8.1e-5 | measure-only | |
| Cached vs forced, per-token abs, p99 | 3.3e-5 | 3.6e-5 | measure-only | |
| Cached vs forced, per-row abs M, max | 2.82e-4 | 2.05e-4 | measure-only | |

Gross/net ratio per adapter, online and offline: median 0.165 and 0.195, p10 0.023 and 0.041, p01 0.0025 and
0.0023, max 0.61 and 0.63.

- **Batch invariance is the only gate with a small margin** (×2.4 offline). Production qualifies the same 16
  rows per arm with the same budget, and the two runs reproduced bit for bit, so the measurement should repeat.
  Per AGENTS.md the tolerance stays as it is. If production fails here, that is a finding.
- **Finite difference** is about 10,000 times inside its gate: the FP32 qualification problem from earlier
  projects does not appear here.

## Decided for the production record (user, 2026-11-22)

- **Cached-against-forced tolerances (provisional):** `per_token_abs_max: 1.0e-3` and `per_row_abs_M_max: 3.0e-3`,
  about 10× the measured maxima of 1.21e-4 and 2.82e-4. The user asked to see how they are computed before the
  record is finalised (next section).
- **Untuned baseline:** included (+150 answers, 3,150 in total).
- **Run ID:** `c-production-1`, label `production`. It is unused in the catalog, the code and the HF export repo.
  The older pipeline's `production-1` is a different ID under another volume subtree. Precheck on the Pod
  refuses any existing run under this ID with a different record.
- **Everything else as in the rehearsal:**
  - the other tolerances, which are production-1's;
  - the seeds: selection 20261121–24, split 20261118, bootstrap 1234, 10,000 resamples;
  - the hot layers [11, 15, 16, 24];
  - batch 16, cap 1024, token budget 8192, 2 questions per unit;
  - timeouts and the 900 s stale marker.

## Cached against forced: what is computed (`c_attribution_checks.cached_vs_forced`)

- **A row** is one saved sampled response: the prompt tokens followed by the generated tokens of one draw (one of
  the 8) for one question, as tokenized from the behaviour record. `M` is that row's sequence log-probability:
  the sum, over every generated token including EOS or the last capped token, of log p(token | prefix), in nats.
  This is the quantity attribution differentiates.
- **Which rows:** 16 per arm, spread evenly over length across all 1,200 rows of the arm, always including every
  capped row (offline has one, 1,024 tokens; online has none). The choice is deterministic, so production uses
  the same 16 rows as the rehearsal. Generated lengths were 98–1,024 tokens (offline) and 105–876 (online).
- **Cached path:** the prompts are left-padded and prefilled with a KV cache, then the saved tokens are fed one at
  a time through the cache. The log-softmax of the FP32 logits at each step gives log p of the saved token. This
  reconstructs how the tokens were generated. The original generation scores were not saved.
- **Forced path:** one right-padded, teacher-forced forward pass over the full row, the path attribution uses,
  with log-softmax at each target position.
- **Per-token difference:** |log p_forced − log p_cached| for every generated token of the 16 rows, pooled per
  arm. The record keeps the max, median and p99.
- **Per-row difference:** |M_forced − M_cached|, the sums taken in FP64. The record keeps the max and median.
- **Scale:**
  - Per-token differences: median exactly 0 (most tokens agree bit for bit), p99 about 3.5e-5 nats, max
    1.2e-4 nats.
  - Per-row differences: max 2.8e-4 nats. That is consistent with FP32 rounding differences between the two
    computation orders, accumulated over up to 1,024 tokens.
  - |M| itself is not saved for these 16 rows. The finite-difference rows (the 4 shortest mixed rows) reach
    |M| = 44.5 nats, so a per-row difference of 2.8e-4 is of order 1e-5 relative or smaller.
- **Gate in production:**
  - With numeric tolerances, qualification adds an `<arm>:cached_vs_forced` check. It passes only if the arm's
    per-token max ≤ 1e-3 **and** its per-row max ≤ 3e-3.
  - A failed check makes the automatic sign-off refuse (`StopBeforeProduction`), so the chain stops before rows
    and attribution. It still runs the teardown (volume copy, HF export, summary) and stops the Pod once a
    backup holds the work.
  - `measure-only` is accepted only in a rehearsal; a production record must carry numbers.
- **What is recorded:** the summary statistics above, the 16 row IDs and their generated lengths, in
  `qualification.json` (details) and `numerics.json`, and shown in the report with the rule.
  **Added 2026-11-22 (user):** one entry per row, with ID, generated length, M_forced, M_cached, |ΔM| and the
  row's largest per-token |Δ|. It is recorded in `qualification.json`, `numerics.json` (per arm) and the report,
  which has a table per arm. Tested red/green; the summary maxima equal the table's.

## Bootstrap benchmark at 150 questions

`scripts/c_attribution_bootstrap_benchmark.py`, synthetic sums at production shape (150 × 28 × 8,960 reads,
196 adapters, 512 selected reads, 10,000 resamples), on this Mac (arm64, 8 CPUs, numpy 2.5.4), calling the same
functions as the aggregate stage. Results are in `artifacts/imports/measurements/bootstrap-benchmark-q150.json`.

| Step | Seconds | Peak RSS |
|---|---|---|
| Resample counts | 0.02 | |
| Neurons, per-layer tier (per arm) | 0.05 | 0.37 GB |
| Neurons, selected reads (per arm) | 0.19 | 0.45 GB |
| Adapters, 2 arms + paired difference and ranks | 0.62 | 0.62 GB |
| *Neurons, every read (not run in production)* | *92* | *0.62 GB* |

Production's bootstrap is about 1.2 s in total here. Allowing 10× for a slower Pod CPU, it is still under 15 s
and well under 1 GB. The aggregate stage is therefore bounded by loading the 150 unit sums (about 1.5 GB) and by
selection, not by the bootstrap: allow a few minutes.

## Production time budget (no time caps; planning only)

| Stage | Basis | Estimate |
|---|---|---|
| Preflight and qualification | the same 16 rows per arm (rehearsal: 8 s + 321 s) | about 6 min |
| Attribution, 2 × 1,200 rows | 826,549 tokens at 880–1,050 tokens/s (steady units) | 15–20 min |
| Aggregate, bootstrap, freeze | bootstrap about 1 s at 150 questions (benchmark); loading and selection | under 5 min |
| Greedy, 3,150 answers | 21 condition runs × 10 batches of 16; rehearsal batches took 33–90 s | 2–5.5 h |
| Report, volume copy, export | about 1.5 GB of unit sums (53 MB took 9 s) | about 5–10 min |
| **Total** | | **about 2.5–6.5 h**, mostly greedy |
