# 03: Principal edits and per-module magnitude/random controls

**What to build:** The CPU notebook builds and verifies V0a/V0b, V1/V2, three V3 draws, V4, the analogous V2 size control and V5; users inspect an immutable variant catalogue with hashes and norms before evaluation.

**Blocked by:** 02.

**Status:** resolved

**Track:** Primary.

- [x] k=0 identity is byte-preserving; projected components satisfy declared numerical gates; per-module effective-update sizes match their controls and directions remain correct. Seeds, construction, partial progress and dense independent matrix checks are retained. Existing adapters unchanged.
- [x] The behavior is reachable through the public saved-workflow commands used by the applicable shared notebook, with visible inputs, outputs and resource requirements.
- [x] CPU-only public-boundary tests use tiny fixtures or replaceable backends, include an independent oracle where applicable, and cover interrupted resume and changed-source rejection without a model download.
- [x] Persist configuration, source identities, structured results, status and completion evidence; keep real execution/qualification status separate from software implementation status.

## Answer

Implemented public edit prepare/run/verify commands and CPU notebook sections
10-12. Ten variants cover references, principal removals, three random draws,
per-module V1/V2 magnitude controls and the GRPO output removal. Sealed module
units retain factors, checks and random bases; final PEFT payloads are read back
and verified. Existing sources remain unchanged. See execution-ticket03.md.

Software complete only: actual source/cache execution and real adapter
construction remain pending, and no GPU qualification/evaluation is claimed.

## Comments

- 2026-10-26: User approved this slice, its blocking edges and the shared public workflow test boundary. No real GPU completion is implied.
- 2026-10-26: Implementation started. Session baseline: 485 passed, 35 skipped;
  latest affected workflow regression: 77 passed. No GPU or download required.
- 2026-10-26: Resolved after 85 affected/regression tests passed, including eight
  edit tests. Fresh-process restoration additionally passed. Tested implementation
  pin: a69768afd39fcc81b99c5a5fb07f1997d4e416d6. No model download/GPU use.
