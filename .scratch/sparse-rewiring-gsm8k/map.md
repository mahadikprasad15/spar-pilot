# Sparse rewiring on GSM8K — map

## Notes
- Design input: `design-input.md` (researcher, 2026-11-22, unchanged). Filled design: `design.md`. Options: `decision-log.md`. Draft spec: `spec.md`. Tickets: `issues/01`–`12`.
- Docs-only effort so far. No code, GPU work or artifact change. The parallel C-attribution run (`c-production-1`) finished before this effort started and is read-only input here.
- Evidence came from Git (`main@b1a541b`, `c-attribution-pilot@7667739`, `max@a6dc314`) and the catalog. Saved run files were not reachable; ticket 01 reads them.
- Recomputed on CPU from code: the 512 training-ID digest and the cohort150 digest both match the catalog.

## Decisions so far
- 2026-11-22 (researcher, design input): gold-SFT data and settings where they improve comparability; Max's defining method; rank 1 first; gate+up parts; no training or GPU spend during design.
- 2026-11-22 (Claude, proposed only): D1–D19 in `decision-log.md`.
- 2026-11-23 (researcher): add a training-dynamics figure (gates, read-change norms, performance vs step). Spec §Training dynamics, D20, ticket 12.
- 2026-11-23 (researcher): D13 decided — run both rank 1 and rank 0, everything else identical. Proposed rank-0 LR 3e-4; comparison D21.
- 2026-11-23: D14 revised (proposed, not signed off): microbatch 1 with one mask draw shared per 4 examples, matching Max's 2 draws per step.

## Fog
- Whether gold's saved order, resolved trainer arguments and checkpoint hashes match the code reconstruction.
- `sequences-v2` IDs (needed to exclude them from validation).
- Mask dynamics at rank 1 on GSM8K: survivor count, and whether λ 6e-4 is degenerate.
- Actual GPU time and memory.
- Whether Max will license the rewiring code.
