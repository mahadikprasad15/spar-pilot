# 03: Leave-one-out weights, aggregation and sign convention

**Status:** resolved
**Type:** task
**Blocked by:** 01, 02

**What to build:** `pilot_eval/c_attribution.py`. It is pure numpy, with no torch import at module level.

- [x] **Weights.** `loo_weights(groups)` computes w_qk = (R_qk − b_qk)/(Q·n_q) with b_qk = (Σ R − R_qk)/(n_q − 1).
  - n_q < 2 is refused.
  - All-same groups give exactly 0.
  - Σ_k w_qk = 0.
  - It also returns the per-question coefficients (R_qk − b_qk)/n_q used by ticket 01's pre-aggregation.
- [x] **Draw-half sums.** The pass also accumulates, per arm, Σ_q S_q^A and Σ_q S_q^B for the reads ([2, 28, 8,960], FP64), with leave-one-out weights recomputed within draws 0–3 and draws 4–7 (n = 4). Test: on a fixture, these equal the sums recomputed from per-response attributions.
- [x] **Aggregation.** `aggregate(S, Q)` computes removal_J = (1/Q) Σ_q S_q and dJdλ = −removal_J. It does this for the 196 adapters and for every `down_proj` layer's 8,960 reads, reading the sealed units.
- [x] **Totals and group predictions.**
  - The gross total Σ_c |dJdλ_c| and the net total |Σ_c dJdλ_c|, per scope.
  - Shares always use the gross total.
  - `group_prediction(selection)` = Σ removal_J over the group.
- [x] **Sign convention.** The table from the spec appears verbatim in the module docstring and in every output's `sign_convention` field.
- [x] **Aggregated neuron identity.** Σ_j dJdλ_j = dJdλ_i per `down_proj` layer, under production-1's rule (relative to Σ_j |dJdλ_j|), with the net error also recorded. Written to `identity.json`.
- [x] **Tests:**
  - hand-computed weights;
  - pre-aggregated and direct aggregation are equal;
  - an exact-enumeration fixture (2-token vocabulary, length-2 responses) where the exact ∇J equals the estimator's expectation over all responses;
  - a synthetic helpful adapter gives dJdλ > 0 and removal_J < 0;
  - a cancellation fixture whose net total is about 0 while its gross total is large: shares stay finite and correct;
  - mutations fail: a sign flip, GRPO std-normalized advantages, a token-mean metric.

## Comments

- 2026-11-19 (Claude): `pilot_eval/c_attribution.py`. `reduction_request(rows, rewards)` builds the instrument's detached per-row coefficients and half coefficients from whole-question rewards, so packed micro-batches may split a question. `QuestionSums` accumulates adapter S_q (from per-response attributions), read S_q (from the instrument's `by_question`), and both draw-half sums, refusing a row added twice and checking the row count. Reading sealed units from disk belongs to the runner (ticket 09), which feeds the same accumulator.

- 2026-11-18 (round 1): the estimator and signs were confirmed sound under the stated assumptions. Pre-aggregation, the gross and net denominators and group predictions were added (F2, F6, decisions 9–11).

## Answer

`pilot_eval/c_attribution.py`; tests in `tests/test_c_attribution_aggregation.py` (18 pass). Exact enumeration (2-token vocabulary, length-2 responses, 2 questions) gives the estimator's expectation equal to the exact gradient to 1e-12 for n = 2, 3 and 4 draws; the sign-flip, std-normalized and token-mean mutants all miss it. An end-to-end tiny-model test runs the instrument's reduced path on micro-batches that split questions, and matches direct weighted per-response aggregation to 1e-10 for adapters and reads, with the aggregated identity passing.
