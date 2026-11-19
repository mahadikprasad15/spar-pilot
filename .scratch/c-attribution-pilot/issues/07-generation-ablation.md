# 07: Greedy generation-ablation stage

**Status:** resolved
**Type:** task
**Blocked by:** 06

**What to build:** `pilot_eval/c_attribution_generation.py`.

- [x] **Conditions per arm,** from the freeze:
  1. intact
  2. `top10-adapters`
  3. `match10-adapters`
  4. `global64-reads`
  5. `match-global64`
  6. `hot64-reads`
  7. `match-hot64`
  8. `layer64-reads`
  9. `match-layer64`

  Plus untuned (regenerated, proposed) and a second intact pass per arm on the same questions and batches [user, 2026-11-19]. Adapter and read removals are never combined. A condition whose control is unavailable is recorded as skipped, with the reason.
- [x] **Greedy only.** The settings:
  - FP32, cap 1,024, EOS 151645, repetition penalty 1.0, left padding;
  - explicit arguments over `generation_config`;
  - **one fixed question order and batch membership** for every condition and arm, at the qualified batch size.

  Sampled mode is not part of this launch; it is deferred to ticket 14's design.
- [x] **Units.** Each unit saves responses JSONL (prompt, generated IDs including EOS, text, stop reason, flexible-v3 score, reward) and a config file (AGENTS.md). Each unit verifies the restored adapter hash after its condition.
- [x] **Summary per condition.** Intervals use a paired bootstrap over question IDs: 10,000 resamples with the same indices across conditions and arms.
  - raw accuracy;
  - paired Δpp versus intact, with its interval;
  - 100 × ablated/intact, undefined if intact is 0;
  - target − control (per-question paired), with its interval;
  - gains and losses, and the exact McNemar p;
  - invalid and capped counts, and mean and median tokens;
  - the group's and the control's first-order sampled-J predicted removal effect, labelled;
  - the intact-repeat flip count per arm (the nondeterminism floor).
- [x] **Labels:** "exploratory (selected on these questions)", "greedy ≠ sampled J", and "predicted effects are local first-order sampled-J approximations".
- [x] **Tests:**
  - tiny-model greedy runs reproduce;
  - batch membership is identical across conditions;
  - an intervention changes logits only where expected;
  - summaries are hand-checked on a fixture, including a zero-intact denominator;
  - the labels are present;
  - a skipped condition is recorded, not silently absent.

## Comments

- 2026-11-19 (Claude): `pilot_eval/c_attribution_generation.py`. Conditions are read only from a sealed freeze (`read_freeze`), in the spec's order plus `intact-repeat`; untuned is `untuned_condition()` (adapter disabled via `frozen_disable_adapter`), run once. `greedy` replaces `generation_config` with fresh greedy defaults and passes every argument explicitly, left-padded. Each (arm, condition, batch) is sealed with `seal_run`; on resume a sealed unit is verified and reused, and a changed batch membership is refused. The restored LoRA hash is checked after each condition (all its units), not after each unit. Summaries use the shared resample counts from ticket 04 (seed 1234), so intervals pair across conditions and arms. The nondeterminism floor counts questions whose tokens or correctness differ between the two intact passes.

- 2026-11-18 (round 1): nine greedy conditions per arm. Sampled mode was removed from this launch. Paired reporting and fixed batching were added (F1, F5, F6, decisions 2, 4 and 10).

## Answer

`pilot_eval/c_attribution_generation.py`; tests in `tests/test_c_attribution_generation.py` (9 pass): spec order with a recorded skip, fixed batches, exact McNemar values, tiny-model greedy runs reproducing and resuming as reused units, refusal without a freeze and on changed batch membership, interventions changing logits only where expected (a layer-1 read edit leaves layer 0 unchanged) and restoring, hand-checked summaries including target − control, the unavailable-control case and the floor, and a zero-intact denominator. The real GPU run (qualified batch size, 3,150 answers) is in tickets 11–12.
