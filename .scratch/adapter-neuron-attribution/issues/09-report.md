# 09: Report

**Status:** needs-triage — draft for adversarial review
**Type:** task
**Blocked by:** 06, 07, 08 for real data; fixture-implementable after 04

**What to build:** `pilot_eval/attribution_report.py`: the validity gate, then the three primaries with their frozen rules, then labelled exploratory items.

- [ ] Gate: pooled predicted-vs-real Spearman per metric; top-10 group sign agreement per arm. A failed gate changes which evidence the primaries use, never a threshold.
- [ ] Primaries with paired bootstrap CIs (shared question resampling).
- [ ] Coverage: arms present, missing units, modules where a moved.
- [ ] Figures: per-arm layer × module attribution maps (both metrics); predicted vs real; neuron concentration curves with shuffled baseline; Stage D accuracy and length per condition.
- [ ] Fixture tests: planted separable and non-separable cases recovered; gate trips on a planted bad estimator.

## Comments
