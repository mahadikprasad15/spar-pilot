# 01: Source audit, fresh-runtime restoration and budget status

**What to build:** Select source runs in the CPU notebook, audit actual source metadata/tensors and available products, save a parameterized Stage 0 plan, then restore it in a new process and inspect source readiness and cumulative budget. No GPU call occurs. Missing optional sources do not block endpoints.

**Blocked by:** None (can start immediately).

**Status:** resolved

**Track:** Primary.

- [x] Source evidence is distinguished from assumptions; changed source payloads invalidate identity; actual A0 comparisons are reported when available, not inferred from seeds. Persist status, exact source references and credit observations. Restore retains the total 20-unit ceiling and never interprets missing observations as zero expenditure. Existing behavior and release paths remain intact.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Answer

Implemented public prepare/audit/status/budget-record/recover-writer operations,
CPU tensor/source verification, sealed per-source resume, cumulative hash-chained
budget observations with rollback detection, and both shared notebook entrypoints.
The notebook release pin is the tested implementation commit, not an unimplemented
bootstrap. See the Ticket 01 execution record for exact evidence and limitations.

Software completion only: real Drive audit remains pending and no GPU qualification
or scientific measurement is claimed. Budget recording does not authorize GPU jobs;
cost reservations and production authorization belong to Ticket 04.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
- 2026-10-26: Implementation started. Session baseline: 485 passed, 35 skipped,
  4 warnings; model-backed skips include the local Torch library failure.
- 2026-10-26: Resolved after 68 affected/regression tests passed, including 15
  Stage 0 workflow tests. No model download or GPU work. Implementation pin
  76c20655ae5acb1a26ed6dfa06d39e0831b3e5f8; notebooks require publication before
  Colab can fetch it.
