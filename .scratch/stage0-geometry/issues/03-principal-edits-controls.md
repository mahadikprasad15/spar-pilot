# 03: Principal edits and per-module magnitude/random controls

**What to build:** The CPU notebook builds and verifies V0a/V0b, V1/V2, three V3 draws, V4, the analogous V2 size control and V5; users inspect an immutable variant catalogue with hashes and norms before evaluation.

**Blocked by:** 02.

**Status:** claimed

**Track:** Primary.

- [ ] k=0 identity is byte-preserving; projected components satisfy declared numerical gates; per-module effective-update sizes match their controls and directions remain correct. Seeds, construction, partial progress and dense independent matrix checks are retained. Existing adapters unchanged.
- [ ] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [ ] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [ ] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
- 2026-10-26: Implementation started. Session baseline: 485 passed, 35 skipped;
  latest affected workflow regression: 77 passed. No GPU or download required.
