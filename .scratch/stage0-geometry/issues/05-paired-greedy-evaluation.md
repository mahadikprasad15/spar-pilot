# 05: Resumable paired greedy intervention evaluation

**What to build:** Execute the frozen principal variants and fresh SFT/GRPO references on the 150 Pilot 4 questions, resume sealed batches after an interruption, and produce a CPU-readable behavioral result with all responses.

**Blocked by:** 04.

**Status:** resolved

**Execution:** real-GPU collection pending actual Ticket 04 qualification and approvals.

**Track:** Primary.

- [x] Exact prompt/template/cap, fixed batches and scorer contracts; JSONL response/token/length/cap records; completed-unit counters; source and runtime mismatch rejection; unit durability and one-writer safety. References belong to the same comparison runtime. Restart cannot reset the credit ledger or silently combine incompatible sessions. Tiny backends demonstrate the whole collect/interrupt/verify/inspect path.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [x] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Answer

Public prepare/run/status/verify/recover-writer commands, real local-only FP32
PEFT generation, CPU response reconstruction and both notebook handoffs are
implemented. The source/runtime/budget and interrupted-resume behavior is covered
through the shared command boundary without a model download. See
execution-ticket05.md and docs/stage0-geometry.md for evidence and operation.

Real model collection remains pending; this ticket does not include Ticket 06
KL/effective writes or Ticket 07's paired inference and human-review overlays.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
- 2026-10-26: Implemented public prepare/run/status/verify/recover-writer commands, a lazy local-only FP32 PEFT runner, raw response reconstruction and both shared notebook flows. The affected suite passed 100 tests; an additional common-input identity check passed separately. Real qualification, production approvals and collection remain pending. See execution-ticket05.md.
