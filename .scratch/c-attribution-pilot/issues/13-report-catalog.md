# 13: Report and catalog registration

**Status:** done for `c-production-1` (2026-11-22)
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

- 2026-11-20: implemented in `pilot_eval/c_attribution_report.py`, tested in `tests/test_c_attribution_report.py`
  on a real CPU chain run. Every number in the report is checked against the sealed artifact it comes from.
- **Where each part runs.**
  - `report.json` (schema `c-attribution-report-v2`) and `report.md` are built by the chain's report stage, on the
    Pod, from sealed artifacts only.
  - Figures (`figures`) and catalog registration (`register`) run locally after the export is restored. The Pod
    needs no matplotlib and cannot commit the catalog.
- **Report sections.**
  - Header: how to read the results, the sign convention, labels.
  - Attribution per arm: J̄, informative questions, the adapter table with exploratory intervals and rank
    intervals, gross and net totals, the top-10 share of gross, per-layer read sums with intervals and
    distributions, the identity, harmful adapters (reported only).
  - Stability: question split and draw split.
  - Qualification and numerics: every check, FD against FP64 with its floors, cached vs forced, batch
    invariance, identity, no-op, processors, memory, replay (not run).
  - Provenance per arm: verified, recorded, inferred.
  - Selection: per group and its control, hot-layer provenance.
  - Ablation: every condition, the untuned row, the nondeterminism floor.
  - Selected-read and matched-control intervals.
  - Online vs offline: paired adapter intervals, top-10 and read overlaps, distributions, stability side by side,
    the one-seed caveat.
  - The three labels head every results section.
- **Figures** (palette and conventions from `attribution-production-1/build_figures.py`), from `report.json` only:
  - a layer × module dJdλ heatmap per arm;
  - an online vs offline scatter;
  - per-layer read-score distributions;
  - ablation Δpp bars per arm, with each matched control beside its target.
- **Catalog registration.**
  - One `greedy-only` evaluation entry per arm and condition, plus untuned once. Each lists one file per fixed
    batch with its sha256 and row count, with `cohort_coverage: subset`, because `verify` checks each file on its
    own.
  - Two new `prior_uses` of cohort150: attribution and selection with the freeze hashes, and the greedy ablation,
    each with the exact question IDs.
  - It refuses to overwrite an existing entry. Run `python -m pilot_eval.catalog verify` and commit only when it
    reports OK.
- **Dependency.** The figures need matplotlib (the project's `report` extra).
- 2026-11-22, production (`c-production-1`):
  - **Report:** `report.md` and `report.json`, built on the Pod and restored from HF.
  - **Figures:** in `artifacts/imports/c-attribution-production/figures/` (6 PNGs and `figures.json`).
    - Layout fixed after looking at the real output: the labels note is wrapped to the plot width (it had widened
      the heatmap and scatter images), and the read-score scale moved into the axis label (the floating `1e-6`
      overlapped the note).
    - Skipped controls read "no matched control", not "n/a".
  - **Catalog:**
    - 15 `greedy-only` response sets: 7 conditions × 2 arms, plus untuned.
    - 3 `prior_uses` of cohort150: qualification (28 questions), attribution and selection with the freeze hashes
      (150), greedy ablation (150).
    - `catalog verify`: all 150 production batch files pass. The 45 failures are the pre-existing ones (files of
      older runs absent from this worktree), unchanged.
  - **Tests:** the `tests/test_c_attribution*` and `tests/test_catalog*` suites pass (257) on HEAD with the
    figure change. With the catalog uncommitted, 7 bundle tests refuse by design ("commit it before building the
    bundle").
