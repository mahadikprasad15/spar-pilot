# 04: Bounded GPU qualification and measured production freeze

**What to build:** The GPU notebook restores the selected catalogue, verifies the pinned runtime/durable storage, runs bounded instrument and mixed-loader diagnostics, reports costs and approval requests, then produces a frozen executable plan only with valid qualification and budget approvals.

**Blocked by:** 03.

**Status:** ready-for-agent

**Track:** Primary.

- [ ] Sixteen-sequence common-prefix diagnostics, negative control, hard 1e-3 stop, proposed 3E gate, >=1000x sensitivity, near-tie exceptions and separate approval receipts follow the spec. Un-halved partial merges never double-apply the adapter. Credit reservations include diagnostics and total prior usage; unsafe jobs are refused before spending beyond the ceiling. Real-Colab evidence is distinct from mocked CPU checks. Test fixtures exercise all gates without a model download.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [ ] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
