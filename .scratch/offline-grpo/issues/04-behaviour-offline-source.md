# 04: Behaviour and offline source handoff

Status: claimed
Blocked by: 02 Train and resume fixed-pool GRPO

## What to build

Provide a versioned, verified offline-adapter source handoff to existing behaviour evaluation and reporting while preserving historical sources and their verification contracts. Evaluate the specified checkpoint and sampled endpoints, with item-paired comparisons and explicit compatibility checks before any reuse.

## Acceptance criteria

- [ ] Source handoff checks plan, model, adapter, checkpoint and data identities; valid historical products remain readable without rewriting their evidence or requiring them to match new trainer code.
- [ ] Reports retain per-answer text, tokens and scores, greedy 150-item checkpoint results, sampled 0/64 results, paired transitions and pointwise paired intervals; sampled accuracy is mean reward per draw.
- [ ] CPU fixtures cover source identity failures, legacy-source compatibility, cohort pairing and endpoint reuse decisions. The real same-runtime checkpoint-0 token-for-token anchor is an execution gate, not a fixture result.
- [ ] Runtime differences and incompatible old outputs are recorded under a new plan; historical fingerprints and acceptance thresholds are not weakened.

## Root verification

Implemented additive public prepare/run/verify/report with existing GRPO policy, record, scoring, inference and bootstrap code. Same-runtime verified historical checkpoint-zero inference gates nonzero evaluation. Combined new and existing evaluation regression passed 11 tests with one optional model-stack skip. Final focused suite passed 3 tests after manifest addition. CPU source and inference fixtures establish software behavior; actual trained-artifact and GPU execution remain pending with tickets 02 and 03.
