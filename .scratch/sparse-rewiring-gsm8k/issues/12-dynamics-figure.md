# 12: Training-dynamics figure

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03, 04, 06

Requested by the researcher on 2026-11-23. A CPU figure builder for `spec.md` §Training dynamics. Build it on fixture logs now, and run it on the real run's logs after ticket 09.

- [ ] Four panels on a shared step axis. Panel 1: expected-active, nonzero `z_det` and binary survivor counts on a log scale, plus a layer × step heatmap. Panel 2: gauge-invariant read-change norms (gate and up), over all neurons and over survivors only. Panel 3: dense validation CE, validation greedy accuracy at the checkpoints, cohort150 at 64 and final only, and untuned and gold-sft step-64 reference lines. Panel 4: a freeze marker, stop-check markers with pass/fail, plus churn (Jaccard overlap, counts switching on and off) between checkpoints.
- [ ] Red/green: spec test 14.
- [ ] Gold-sft checkpoints 0 and 64 get evaluated on the validation set (CE and one greedy pass) inside ticket 09's run, and that use is registered in the catalog.
- [ ] One figure per arm, plus a side-by-side rank-1 vs rank-0 version on shared y axes.
- [ ] Outputs go to `dynamics/`: PNG and SVG, the CSV that is plotted, and a provenance JSON (input hashes, run ID, code commit).
- [ ] Caption: descriptive co-movement in one run, not a causal link between gate shutting and accuracy.

Cadence decided (D20): cheap metrics every step, validation CE every 16 steps.

## Comments

## Answer
