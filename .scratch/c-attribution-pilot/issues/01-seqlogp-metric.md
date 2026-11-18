# 01: Sequence log-probability metric

**Status:** needs-triage
**Type:** task
**Blocked by:** —

**What to build:** a third metric `seqlogp` in `attribution_metrics.py` and `attribution_instrument.py`, alongside the existing `trace` and `stop`, which stay unchanged.

- [ ] `seq_targets` mask = every generated position: content tokens plus the sampled EOS on EOS rows; all 1,024 tokens and no EOS on capped rows. Value = **sum** of log p over the targets (FP64 accumulation), via the existing `position_terms`.
- [ ] `Instrument` accepts any non-empty subset of the known metrics (`metric_order`), default `('trace', 'stop')`, so every existing caller and test is unchanged. Remove the hard-coded "must name both metrics" check only for subsets of known names.
- [ ] Attribution still contracts over **all unpadded positions** (prompt prefill included); pass 2 and the neuron identity work for `seqlogp`.
- [ ] Rows may carry only `seq_targets` (no trace or stop masks) when only `seqlogp` is requested.
- [ ] Tests (tiny Qwen2 + PEFT 0.18.0): hand-computed sums with distinct per-position values; EOS included; capped row; EOS-only response (one target); mutations fail (mean-not-sum, EOS dropped, response-only contraction); `seqlogp` attributions match their own finite differences and the branch oracle; `trace` and `stop` outputs bit-identical with and without `seqlogp` requested; every existing instrument and metrics test passes unmodified.

## Comments

## Answer
