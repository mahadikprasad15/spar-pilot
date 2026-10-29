# 03: Execute a bounded likelihood-search trace and qualify engine scores

**What to build:** A public command runs the declared k recipe on bounded inputs and saves raw proposals, stored token IDs, scores, acceptance decisions and terminal states. It handles gold-guided prompts, greedy whole-trace regrowth, J64/cap1024, engine replay and exact slicing. Independent HF comparison emits the20-trace score receipt with both approved thresholds. Bounded throughput candidates select and freeze one L40S runtime without automatic retuning later.

**Blocked by:** 01, 02.

**Status:** resolved

**Type:** task

## Acceptance criteria

- [x] Run bounded full-trace sampling via the public workflow, including gold cleaning, intentional prefix duplication, distinct proposer/scorer prompts and retained raw IDs.
- [x] Use temperature0.6/repetition penalty1.1/top-p1 with explicitly pinned remaining defaults, J64/cap1024, strict mean-score improvement and whole-trace cuts/regrowth; k0 has no refinement decisions.
- [x] Implement the length-based loop and2×ceiling(cap/J) safety bound; cover finished/capped/exhausted/error states, stop-only proposals, shorter regrowth and validated first-stop/cap accounting.
- [x] Handle empty generation and malformed scoring as protocol errors; replay the same logical request once before stopping, without consuming a model retry.
- [x] Emit T3 evidence for20 predeclared category-complete traces with exact IDs/position alignment, every mean error≤0.001 nats/token and every maximum per-token error≤0.01 nats; include actual stops.
- [x] Retain each trace’s errors, worst position and aligned scores; failed gates block production with diagnostics and no numerical relaxation.
- [x] Profile only bounded reserved candidates, verify actual worker FP32/TF32off and choose one L40S runtime configuration. Record batch dependence without promising seed-only invariance.
- [x] Independent CPU fake-engine/math tests exercise the same workflow; real engine/category/profile checks remain explicitly pending until separately authorized and executed.

## Testing and evidence

Tiny offline engines/math fixtures exercise the complete trace path, strict improvement and all terminal/error cases; a separate real qualification checklist requires worker flags, token alignment and category-complete score fixtures. All real profile/check work requires reservations and first-spend approval.

Use red/green TDD at the approved public-workflow boundary. CPU tests run offline without model downloads; record skipped model-backed checks honestly. Real hardware/paid execution evidence is separate. Do not change approved thresholds to make a test pass.

## Scope and authorization

Parent: confirmed Sampling SFT specification. Read the confirmed design amendments before implementation. All work uses canonical durable artifacts and fresh-session restoration. This ticket is software work, not permission to rent a GPU, train, delete storage, publish or push. Preserve required stage-specific approvals.

## Comments

2026-10-28: User approved the ten slices and all declared dependency edges. Published locally; not claimed or implemented.

## Answer

Implemented the engine-independent search, versioned vLLM/HF adapters, explicit runtime/source plans, guarded paid-job child entrypoint, raw request/response and block/RNG evidence, T3 comparison receipts and bounded throughput selection. Shorter/EOS proposals retain upstream ordering; strict score improvement, stop-only legality, 1024-token cap, doubled block bound and exact response slicing are covered by independent CPU cases. Malformed requests replay once and fail with retained diagnostics. No hidden model download or precision fallback. CPU-injected dependencies cannot qualify a GPU.

The optional notebook remains a thin CPU plan client; CLI/RunPod run script is primary. See docs/operations/sampling-sft-engine.md. Source/budget/lifecycle/sampler/engine/instrument regression:78 passed in82.80s. Canonical validation receipt:artifacts/validation/sampling-sft-ticket03/validation.json. Real L40S checks, category-complete T3 execution, measured profiles and runtime choice are not executed; software resolution is not scientific qualification. No paid job, model inference, rental, deletion or push occurred. Ticket04 owns exact pilot restart/resume.
