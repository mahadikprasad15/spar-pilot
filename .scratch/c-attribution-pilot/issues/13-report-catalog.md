# 13: Report and catalog registration

**Status:** needs-triage
**Type:** task
**Blocked by:** 07, 04

- [ ] **`report.md` and `report.json`.** Contents:
  - **Header.** The "How to read the results" statements and the sign-convention table.
  - **Attribution summary.**
    - J̄ per arm, and informative-question counts.
    - Adapter dJdλ tables with exploratory pointwise intervals.
    - Gross and net totals; concentration as a share of gross.
    - Neuron-read summaries per layer.
    - Identity, gross and net.
  - **Stability.** The question-split summaries for all four rankings, labelled descriptive.
  - **Qualification and numerics.**
    - Finite differences against FP64, and the floors.
    - Cached against forced, per token and per sequence.
    - Replay (report-only).
    - Batch invariance and no-op.
    - Provenance status: recorded, inferred or verified.
  - **Selection.** Per group: size, any shortfall, hot-layer provenance, and the layer and module distribution. Per control: match ratios, feasibility or unavailability with the deficient targets, and the control's own predicted effect.
  - **Ablation table.** Per condition:
    - raw accuracy;
    - paired Δpp with its interval;
    - % of intact;
    - target − control, with its interval;
    - gains and losses, McNemar;
    - invalid and capped counts, and lengths;
    - group and control first-order sampled-J predicted removal effects, labelled.

    Also the untuned row, and the intact-repeat flip floor.
  - **Comparison.**
    - Online against offline signed scores, with paired intervals.
    - Top-set overlaps, global and per-layer read overlaps, layer and module distributions.
    - The one-seed caveat.
  - **Labels** wherever they apply: exploratory; greedy ≠ sampled J; predicted effects are local.
- [ ] **Figures** (reuse the `artifacts/analysis/attribution-production-1/build_figures.py` conventions and validated palette):
  - layer × module dJdλ heatmaps per arm;
  - online against offline scatter;
  - per-layer read-score distributions;
  - ablation bars with paired intervals and the control beside each target.
- [ ] **Catalog registration:**
  - every greedy run as a response set (`kind: evaluation`, `c_suitability: greedy-only`);
  - the freeze;
  - new `prior_uses` of `gsm8k-test-cohort150`, including the rehearsal IDs;
  - `python -m pilot_eval.catalog verify` reports OK.

## Comments

- 2026-11-18 (round 1): predicted effects, paired deltas, control feasibility, cross-arm comparison, stability and labels were added (F1, F2, F6, decision 10).

## Answer
