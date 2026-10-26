# 02: Verified spectral cache and endpoint projection report

**What to build:** Run CPU SVD once per selected base module, resume partial cache creation, and produce endpoint projection tables/plots against initialization, chance and random references. Save bases sufficient for principal edits and floored intruder subspaces, not just the top-20.

**Blocked by:** 01.

**Status:** resolved

**Track:** Primary.

- [x] SVD reconstruction/orthogonality and payload identity gates pass or preserve failure evidence. k30/k50/k70 and dimensions reconcile. Zero factors are undefined where appropriate. Tiny explicit matrices provide independent workflow oracles. A rerun reuses verified module units. Cache preparation does not automatically reveal trained overlaps; report reveal respects the preregistration boundary, while the primary edit path can consume bases without waiting for optional gradient predictions.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Answer

Implemented public spectral prepare/run/verify/report operations, exact CPU
float64 SVD with retained-basis checks, per-module resumable cache, local base
hash verification, and guarded projection reporting with JSONL/CSV, SVG/PNG and
Markdown outputs. CPU notebook sections 5-9 expose the complete saved workflow.
The GPU notebook remains a CPU-safe status entrypoint.

Real Qwen source/weight execution is pending. Software completion does not claim
that the production cache or trained adapter geometry has been measured.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
- 2026-10-26: Implementation started; no GPU execution. Ticket 01 and its
  existing workflow regression checks are green (68 passed).
- 2026-10-26: Resolved after 77 affected/regression tests passed, including nine
  spectral tests. Real file reader exercised on an independent synthetic weight
  snapshot; no model download or GPU use. Code pin:
  6664bc55df4b8e16136fac29749b609b6cd0d352. See execution-ticket02.md.
