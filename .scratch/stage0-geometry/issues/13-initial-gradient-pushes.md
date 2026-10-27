# 13: Initial gradient pushes and training-alignment report

**What to build:** Run frozen matched gold/own batches and historical GRPO loss diagnostics at saved initialization, retain per-batch gradients/pushes, and report their size, consistency and geometric/endpoint alignments on CPU.

**Blocked by:** 02, 11.

**Status:** claimed

**Track:** Optional supporting branch.

- [ ] No optimizer update; proper masked CE and DAPO reductions across microbatches; detached historical advantages; dead-group zero pushes; gradient sign and factor-gauge conventions correct. Independent tiny-model reconstruction/finite differences retain approved gates. Differing A0 and selected-cohort sizes are explicit. Conditional predictions are frozen before revealing trained-overlap results, without blocking the primary path. If overlaps were already inspected, label subsequent interpretation exploratory rather than claiming a prospective prediction.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.
- [ ] Model-backed execution requires the applicable recorded approvals and cumulative budget checks; total Stage 0 usage cannot exceed 20 Colab compute units without further user approval.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
