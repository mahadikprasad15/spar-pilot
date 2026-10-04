# 02: Validate and measure one batch

**What to build:** Run a bounded diagnostic batch through the real activation instrument, validate its identities, and save interpretable block/module summaries across the five checkpoints.

**Blocked by:** 01 — Freeze and audit measurement inputs.

**Status:** ready-for-agent

- [ ] Measure all intended decoder blocks and adapted projections in FP32 on frozen inputs; distinguish block changes from same-input direct contributions.
- [ ] Pass exact-zero, frozen-base/source identity, rank-1 and disabled-adapter invariance gates with saved diagnostics under the spec's thresholds.
- [ ] Save per-example sufficient summaries, coverage and diagnostic completion evidence; no full-cohort completion claim from a diagnostic batch.
- [ ] Independently calculated CPU cases verify weighting, cancellation, undefined denominators and negative controls.
- [ ] Tiny locally constructed Qwen/PEFT tests exercise real hooks, padded inputs and switching without model downloads.
- [ ] Cleanup removes hooks/restores state after errors; failed gates block scientific completion.
