# 02: Validate and measure one batch

**What to build:** Run a bounded diagnostic batch through the real activation instrument, validate its identities, and save interpretable block/module summaries across the five checkpoints.

**Blocked by:** 01 — Freeze and audit measurement inputs.

**Status:** resolved

- [x] Measure all intended decoder blocks and adapted projections in FP32 on frozen inputs; distinguish block changes from same-input direct contributions.
- [x] Pass exact-zero, frozen-base/source identity, rank-1 and disabled-adapter invariance gates with saved diagnostics under the spec's thresholds.
- [x] Save per-example sufficient summaries, coverage and diagnostic completion evidence; no full-cohort completion claim from a diagnostic batch.
- [x] Independently calculated CPU cases verify weighting, cancellation, undefined denominators and negative controls.
- [x] Tiny locally constructed Qwen/PEFT tests exercise real hooks, padded inputs and switching without model downloads.
- [x] Cleanup removes hooks/restores state after errors; failed gates block scientific completion.

## Answer

Implemented the FP32 Qwen/PEFT activation engine, FP64 summary calculations and
public one-batch diagnostic command. NPZ summaries, JSON diagnostics, source/
runtime identities, failure logs and verified completion markers are durable.

CPU tests cover two-block and 28-block tiny models, all 196 projections at the
production layer count, unequal padded inputs, body-versus-causal-LM hook
agreement, zero write, checkpoint switching, independent numerical examples,
wrong sign/scale/token/module controls, strict below-tolerance nonzero rejection,
state restoration and immutable diagnostic reuse/corruption checks.

2026-10-05 verification: full suite 152 passed with 13 warnings. After the final
local-checkpoint Hub-query regression fix, all 21 activation-related tests passed
again. Expected PEFT tiny-model warnings and a PyTorch TF32 API deprecation remain.
Confirmed red/green cycles; scientific tolerances were unchanged.

This is CPU implementation evidence. Actual Drive source checkpoints, a real
Qwen GPU run and production memory/throughput are still unverified. Those remain
runtime gates. Interrupted diagnostics currently repeat their bounded batch;
production shard-level resume belongs to ticket 04.
