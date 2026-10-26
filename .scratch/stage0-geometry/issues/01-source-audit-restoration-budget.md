# 01: Source audit, fresh-runtime restoration and budget status

**What to build:** Select source runs in the CPU notebook, audit actual source metadata/tensors and available products, save a parameterized Stage 0 plan, then restore it in a new process and inspect source readiness and cumulative budget. No GPU call occurs. Missing optional sources do not block endpoints.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

**Track:** Primary.

- [ ] Source evidence is distinguished from assumptions; changed source payloads invalidate identity; actual A0 comparisons are reported when available, not inferred from seeds. Persist status, exact source references and credit observations. Restore retains the total 20-unit ceiling and never interprets missing observations as zero expenditure. Existing behavior and release paths remain intact.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
