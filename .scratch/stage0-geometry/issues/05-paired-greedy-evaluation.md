# 05: Resumable paired greedy intervention evaluation

**What to build:** Execute the frozen principal variants and fresh SFT/GRPO references on the 150 Pilot 4 questions, resume sealed batches after an interruption, and produce a CPU-readable behavioral result with all responses.

**Blocked by:** 04.

**Status:** ready-for-agent

**Track:** Primary.

- [ ] Exact prompt/template/cap, fixed batches and scorer contracts; JSONL response/token/length/cap records; completed-unit counters; source and runtime mismatch rejection; unit durability and one-writer safety. References belong to the same comparison runtime. Restart cannot reset the credit ledger or silently combine incompatible sessions. Tiny backends demonstrate the whole collect/interrupt/verify/inspect path.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [ ] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
