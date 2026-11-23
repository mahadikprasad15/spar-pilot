# 10: Comparison report

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03

CPU comparison (D19, decided) on fixtures now, real inputs after 01 and 09. GRPO arms first: C attribution (`c-production-1`, full universe) and production-1 adapter attribution (one layer each). Per-layer counts, overlaps with C frozen sets and top-K, layer-matched null, sign strata, coverage stated.

- [ ] Attribution source manifest: one entry per (source, arm) with universe mask, sign convention and frozen sets; other arms' C attribution and joint-audit lists are later entries, not code. Spec test 15.

- [ ] Rank 1 vs rank 0 (D21): survivor Jaccard and per-layer overlap with a layer-matched null, unique neurons, overlap of each arm and of the intersection with the C sets, paired accuracy, validation CE, counts and hardening change.
- [ ] Red/green: spec test 13.
- [ ] Report labels: exploratory; selected object ≠ attribution object; one seed; circular when C overlap is combined with cohort150 accuracy (D12).

## Comments

## Answer
