# 09: Report: validity gates, primaries, secondaries

**Status:** needs-triage — draft for adversarial review
**Type:** task

**What to build:** `pilot_eval/attribution_report.py` reading sealed 06/07/08 outputs: validity gates first, then the three primaries with their frozen rules, then the labelled secondary family.

**Blocked by:** 06, 07, 08 for real data; implementable on fixtures after 04.

- [ ] Gates: predicted-vs-real Spearman per arm × metric; bootstrap top-10 Jaccard; C/D sign agreement. A failed gate changes which evidence the primaries use, as the spec states; it never changes a threshold.
- [ ] Primaries with paired bootstrap CIs (question resampling shared across arms).
- [ ] Secondaries with Holm correction within family, labelled exploratory.
- [ ] Coverage table: arms present, units missing, unresolved modules (a moved, scale unresolved).
- [ ] Figures: per-arm layer × module attribution maps (both metrics), predicted-vs-real scatter, neuron masking curves with controls.
- [ ] Fixture tests: planted separable / non-separable cases recovered; gates trip on a planted bad estimator.

## Comments
