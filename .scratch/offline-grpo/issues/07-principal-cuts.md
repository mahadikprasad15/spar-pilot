# 07: Principal cuts and paired evaluation

Status: resolved (software; real execution pending)
Blocked by: 04 Behaviour and offline source handoff; 06 Internal measurements and comparisons

## What to build

Apply the sealed sampling-SFT cut definitions to the offline checkpoint and evaluate the resulting adapters on the paired held-out cohort using the established evaluation contract.

## Acceptance criteria

- [x] All eight conditions are included: original, principal-output cut, principal-input cut, three random-direction controls, output magnitude control and input magnitude control; ranks and construction match the sealed plan.
- [x] The paired evaluation covers 150 questions per condition, retaining answers and scores; reuse the original endpoint only when its complete evaluation contract matches.
- [x] CPU fixtures verify condition and control identities, cohort pairing and endpoint reuse decisions; constructing adapters or passing fixtures alone is not evidence of behavioural effects.

## Implementation evidence

The additive `offline-grpo-cuts-*` commands provide prepare, build, agreement,
evaluate, verify and report. Construction reuses the historical rank-one edit
math and checks the sealed historical per-module k50 ranks. Eight conditions
produce 1,200 paired responses on the 150-question cohort. The original endpoint
is freshly evaluated in this workflow. Batched inference requires the existing
numeric gate and output-cut greedy replay, with singleton fallback.

Source and payload verification, shared execution locking and budget accounting,
partial resume, model integrity proofs, JSON/Markdown reports and pointwise paired
bootstrap intervals are included. Normal verification checks sealed construction
and assembled factors without replaying QR. Explicit build verification with
`--deep` replays the numerical construction.

Validation passed 13 combined tests before the final verification-performance
refactor and 5 focused tests after it. Fixtures cover omitted historical base
vectors, missing source plans, tampering, all conditions, cohort pairing,
batching fallback, interruption/resume and public CLI operations.

Actual GPU evaluation remains pending. The inspected historical archive lacks
`plans/a100-stage0-cuts-v1/sampling-stage0.cuts.json`; restore this exact source
plan before real preparation. CPU fixture outputs establish software behavior.
