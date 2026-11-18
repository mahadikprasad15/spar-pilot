# 07: Greedy generation-ablation stage

**Status:** needs-triage
**Type:** task
**Blocked by:** 06

**What to build:** `pilot_eval/c_attribution_generation.py`.

- [ ] **Conditions per arm,** from the freeze:
  1. intact
  2. `top10-adapters`
  3. `match10-adapters`
  4. `global64-reads`
  5. `match-global64`
  6. `hot64-reads`
  7. `match-hot64`
  8. `layer64-reads`
  9. `match-layer64`

  Plus untuned (regenerated, proposed) and an optional intact repeat (open decision 5). Adapter and read removals are never combined. A condition whose control is unavailable is recorded as skipped, with the reason.
- [ ] **Greedy only.** The settings:
  - FP32, cap 1,024, EOS 151645, repetition penalty 1.0, left padding;
  - explicit arguments over `generation_config`;
  - **one fixed question order and batch membership** for every condition and arm, at the qualified batch size.

  Sampled mode is not part of this launch; it is deferred to ticket 14's design.
- [ ] **Units.** Each unit saves responses JSONL (prompt, generated IDs including EOS, text, stop reason, flexible-v3 score, reward) and a config file (AGENTS.md). Each unit verifies the restored adapter hash after its condition.
- [ ] **Summary per condition.** Intervals use a paired bootstrap over question IDs: 10,000 resamples with the same indices across conditions and arms.
  - raw accuracy;
  - paired Δpp versus intact, with its interval;
  - 100 × ablated/intact, undefined if intact is 0;
  - target − control (per-question paired), with its interval;
  - gains and losses, and the exact McNemar p;
  - invalid and capped counts, and mean and median tokens;
  - the group's and the control's first-order sampled-J predicted removal effect, labelled;
  - the intact-repeat flip count, if run.
- [ ] **Labels:** "exploratory (selected on these questions)", "greedy ≠ sampled J", and "predicted effects are local first-order sampled-J approximations".
- [ ] **Tests:**
  - tiny-model greedy runs reproduce;
  - batch membership is identical across conditions;
  - an intervention changes logits only where expected;
  - summaries are hand-checked on a fixture, including a zero-intact denominator;
  - the labels are present;
  - a skipped condition is recorded, not silently absent.

## Comments

- 2026-11-18 (round 1): nine greedy conditions per arm. Sampled mode was removed from this launch. Paired reporting and fixed batching were added (F1, F5, F6, decisions 2, 4 and 10).

## Answer
