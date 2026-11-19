# 01: Sequence log-probability metric

**Status:** resolved
**Type:** task
**Blocked by:** —

**What to build:** a third metric, `seqlogp`, in `attribution_metrics.py` and `attribution_instrument.py`. It sits alongside the existing `trace` and `stop` metrics, which stay unchanged.

- [x] **The mask.** `seq_targets` covers every generated position:
  - content tokens, plus the sampled EOS on EOS rows;
  - on capped rows, all 1,024 tokens and no EOS.
- [x] **The value** is the **sum** of log p over the targets, accumulated in FP64, via the existing `position_terms`.
- [x] **Metric selection.** `Instrument` accepts any non-empty subset of the known metrics. The default stays `('trace', 'stop')`, so every existing caller and test is unchanged.
- [x] **Contraction.** Attribution still contracts over **all unpadded positions**, prompt prefill included. Pass 2 and the neuron identity both work for `seqlogp`.
- [x] **Neuron-read accumulation.** With leave-one-out coefficients for the batch supplied as detached floats, the neuron-read pass accumulates per-question S_q[28, 8,960] in FP64. It also accumulates Σ|z_j| and Σ|c_t| with token counts. Per-response neuron arrays are not retained (spec: Storage).
- [x] **Identity.** Each response records its identity error with the gross denominator Σ_j |attr_j|, and its net value.
- [x] **Rows.** A row may carry only `seq_targets`, with no trace or stop masks, when only `seqlogp` is requested.
- [x] **Tests** (tiny Qwen2 with PEFT 0.18.0):
  - hand-computed sums with distinct values at each position;
  - EOS included; a capped row; an EOS-only response with one target;
  - mutations fail: mean instead of sum, EOS dropped, response-only contraction;
  - `seqlogp` attributions match their own finite differences against an FP64 reference (`full_fp64`), and match the branch oracle;
  - per-question S_q equals the direct weighted sum of per-response reads;
  - `trace` and `stop` outputs are bit-identical with and without `seqlogp` requested;
  - every existing instrument and metrics test passes unmodified.

## Comments

- 2026-11-19: Instrument supports selected metric subsets and the sequence metric. Tiny CPU adapter/neuron finite differences and the legacy-metric regression checks pass. Question-level neuron accumulation, per-response identity records and remaining ticket acceptance tests are still open.

- 2026-11-19: Implementation started. The standalone sequence-log-probability metric and its initial CPU tests are the first slice; instrument wiring and the remaining acceptance tests are still open.

- 2026-11-18 (round 1): FP64 accumulation, per-question pre-aggregation and the gross-denominator identity were added for review findings F2, F4, F5 and decision 9.

## Answer

Implemented the FP64-accumulated sequence log-probability metric, selectable instrument metrics,
and a streaming neuron-reduction path with per-question sums, draw-half sums, activation totals,
and per-response identity diagnostics. The production caller must supply the validated target masks
and detached question/half coefficients; tickets 02 and 03 own those inputs and downstream aggregation.
Focused CPU validation: 50 tests passed across the new metric, new instrument/reduction tests, and
unchanged legacy metric/instrument tests. Tiny-model checks include FP64 finite differences, an
independent branch-gate oracle, prompt-prefill contributions, and a 1,024-token capped sequence.
No real-model or GPU qualification is claimed.
