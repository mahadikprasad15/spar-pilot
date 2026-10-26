# 02: Verified spectral cache and endpoint projection report

**What to build:** Run CPU SVD once per selected base module, resume partial cache creation, and produce endpoint projection tables/plots against initialization, chance and random references. Save bases sufficient for principal edits and floored intruder subspaces, not just the top-20.

**Blocked by:** 01.

**Status:** claimed

**Track:** Primary.

- [ ] SVD reconstruction/orthogonality and payload identity gates pass or preserve failure evidence. k30/k50/k70 and dimensions reconcile. Zero factors are undefined where appropriate. Tiny explicit matrices provide independent workflow oracles. A rerun reuses verified module units. Cache preparation does not automatically reveal trained overlaps; report reveal respects the preregistration boundary, while the primary edit path can consume bases without waiting for optional gradient predictions.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
- 2026-10-26: Implementation started; no GPU execution. Ticket 01 and its
  existing workflow regression checks are green (68 passed).
