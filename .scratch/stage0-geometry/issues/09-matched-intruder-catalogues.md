# 09: Matched merged-weight intruder catalogues

**What to build:** Build full V6, matched V6/V6c and ratio-restricted variants for SFT and/or GRPO where eligible flags exist; otherwise publish a verified not-applicable result. Catalogue integrates with the existing variant runner.

**Blocked by:** 03, 08.

**Status:** claimed

**Track:** Conditional intruder extension.

- [ ] Halve only eligible primary flagged singular values. Controls are distinct non-flagged top-40 directions, assigned by minimum total absolute singular-value difference with index tie-break. Preserve pair ratios, module exclusions and [0.8,1.25] sensitivity masks without rematching. Full matrices and per-module adapter disable masks reconstruct the intended operator; original sources and full V6 remain separate from matched variants.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
