# 06: Paired fixed-context KL and effective-write results

**What to build:** Collect and verify per-position KL and same-runtime effective writes for frozen variants on the fixed GSM8K/FineWeb sequences, then inspect document/module summaries on CPU without collecting full-vocabulary arrays.

**Blocked by:** 04.

**Status:** ready-for-agent

**Track:** Primary.

- [ ] Counted-position and view mappings are preserved; KL direction and accumulation are correct; direct matrix oracles reconstruct write measurements; switching/integrity/zero controls retain declared gates. Sealed batches resume safely. Saved historical values serve only compatible checks. Joined behavioral/KL claims require matching comparison provenance.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [ ] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
