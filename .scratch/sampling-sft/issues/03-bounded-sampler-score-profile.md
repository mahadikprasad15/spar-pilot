# 03: Execute a bounded likelihood-search trace and qualify engine scores

**What to build:** A public command runs the declared k recipe on bounded inputs and saves raw proposals, stored token IDs, scores, acceptance decisions and terminal states. It handles gold-guided prompts, greedy whole-trace regrowth, J64/cap1024, engine replay and exact slicing. Independent HF comparison emits the20-trace score receipt with both approved thresholds. Bounded throughput candidates select and freeze one L40S runtime without automatic retuning later.

**Blocked by:** 01, 02.

**Status:** ready-for-agent

**Type:** task

## Acceptance criteria

- [ ] Run bounded full-trace sampling via the public workflow, including gold cleaning, intentional prefix duplication, distinct proposer/scorer prompts and retained raw IDs.
- [ ] Use temperature0.6/repetition penalty1.1/top-p1 with explicitly pinned remaining defaults, J64/cap1024, strict mean-score improvement and whole-trace cuts/regrowth; k0 has no refinement decisions.
- [ ] Implement the length-based loop and2×ceiling(cap/J) safety bound; cover finished/capped/exhausted/error states, stop-only proposals, shorter regrowth and validated first-stop/cap accounting.
- [ ] Handle empty generation and malformed scoring as protocol errors; replay the same logical request once before stopping, without consuming a model retry.
- [ ] Emit T3 evidence for20 predeclared category-complete traces with exact IDs/position alignment, every mean error≤0.001 nats/token and every maximum per-token error≤0.01 nats; include actual stops.
- [ ] Retain each trace’s errors, worst position and aligned scores; failed gates block production with diagnostics and no numerical relaxation.
- [ ] Profile only bounded reserved candidates, verify actual worker FP32/TF32off and choose one L40S runtime configuration. Record batch dependence without promising seed-only invariance.
- [ ] Independent CPU fake-engine/math tests exercise the same workflow; real engine/category/profile checks remain explicitly pending until separately authorized and executed.

## Testing and evidence

Tiny offline engines/math fixtures exercise the complete trace path, strict improvement and all terminal/error cases; a separate real qualification checklist requires worker flags, token alignment and category-complete score fixtures. All real profile/check work requires reservations and first-spend approval.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.
