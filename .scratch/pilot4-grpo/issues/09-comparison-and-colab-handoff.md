# 09: Publish the paired-arm comparison and complete Colab handoff

**What to build:** Join verified Pilot 2/3 and Pilot 4 evidence into an
understandable, precision-aware report with a complete guided execution and
artifact-download workflow.

**Blocked by:** 06 — Evaluate greedy and sampled behaviour with the paired gate;
07 — Measure GRPO writes on the existing fixed sequences;
08 — Save per-token coefficients and fixed-context KL for both arms.

**Status:** resolved

- [x] Publish paired behavioural/write/KL checkpoint trajectories and signed
      SFT-versus-GRPO vector cosines with exact source/checkpoint/scorer/input
      hashes and token/equal-example weighting.
- [x] Report matching differences explicitly, including training precision,
      data-source and completion-count differences; do not call an unmatched
      pair an objective-only causal comparison.
- [x] Flag unresolved directions, missing SFT consistency data and unavailable
      diagnostics rather than fabricating values or accepting isotropic cosine
      0.05 as an empirical model-write noise floor.
- [x] Report the one random adapter's per-module weight-norm matching, realized
      activation magnitudes and signed write cosines as a control reference.
      Do not infer a null percentile/p-value from one realization or claim
      weight-norm matching guarantees activation-norm matching.
- [x] Distinguish equal steps, equal write norm and equal KL; contextualize
      historical scores and a preview based on early SFT checkpoints.
- [x] Explain each plot's formula, denominator, population, uncertainty and
      limits, with a concrete worked example and no unsupported mechanism claims.
- [x] Validate completion, required raw artifacts and download inventory before
      showing a complete report; reports require no model loading.
- [x] Finish the single guided notebook with setup, audit, controls, preflight,
      baseline, freeze, training, evaluation, measurement, verification and CPU
      readout stages; do not force completed earlier stages to rerun.
- [x] Explain checkpoint-only training recovery, incomplete-attempt exclusion,
      the maximum 32-step redo gap, and reuse of existing persistence/exit
      helpers; distinguish it from item/draw-level evaluation resume.
- [x] Exercise a complete controlled workflow through interruption/resume and
      final reporting, plus notebook source-pin/child-exit/order contracts.
- [x] Provide explicit GPU-disconnection and report/raw-artifact download
      instructions; preserve all old run/report directories and source configs.

## Comments

- October 7, 2026: approved as ticket 9. Scientific outcomes are not prerequisites
  for a complete handoff: null, inconclusive or flagged results are reported
  accurately when the instrument and artifact contracts are satisfied.
- October 7 amendment: incorporate the approved simplifications and concrete
  statistics/control rules without recreating tickets or changing blocking edges.

## Answer

Implemented October 9, 2026. CPU-only `grpo-final-report`,
`grpo-final-verify` and `grpo-final-export` join verified original evidence,
publish checkpoint trajectories and matching limitations, and export a sealed
raw-artifact bundle. The notebook handoff is in sections 34–36.

Validation: full offline pinned CPU suite **297 passed, 22 warnings**,
1157.87 seconds. The controlled public workflow exercises interruption/resume,
source compatibility, raw-shard loss, report verification and archive reuse.
Notebook source-pin, ordering and child-exit contracts passed. Red/green tests
and the corrected prerequisite inventory are retained in the local diagnostics
artifact tree. No acceptance numbers or scientific thresholds were changed.

Actual Qwen/source/CUDA execution and scientific measurements remain Colab
work; CPU software completion does not establish an experimental outcome.

Guide: `docs/pilot4-final-handoff.md`. Implementation: `pilot_eval/grpo_handoff.py`
and `pilot_eval/grpo_handoff_plots.py`.
