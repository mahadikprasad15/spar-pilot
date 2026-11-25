# 16: Launch record

**Status:** needs-info
**Type:** task
**Blocked by:** 01, 04, 15
**Group:** H

Freeze the launch record from the plan: run IDs for both arms, hardware, seeds, expected outputs, the frozen validation set. Needs the researcher's sign-off on D1 and the other proposed values listed in `spec.md` §Open decisions.

- [ ] A short preregistration of predictions is recommended before the run.

## Comments

- 2026-11-24 (ticket 01): inputs located and hashed (`../source-records.md`). PR 6's `plan` runs on the real
  `training.items.json` and `order.json`: reconstruction `matches`, `gold_order_sha256` `18ba3e0a…`. Validation set
  frozen (ticket 04 comment).

- 2026-11-25: launcher wired for production. `rewire_launch` accepts a production record and passes gold's
  cohort150 (`rewire_bundle.check_cohort150`: ticket 01's hash, 150 `gsm8k:test` items), scored at step 1,024 only
  (D12). The Pod is never stopped unless `run`/`rewire_launch.sh` gets `--stop-pod`. D1 rank 0 is back to 3e-4 in
  the draft and `PRODUCTION`, marked `[proposed]`, so the draft cannot be sealed until the researcher signs it.

## Answer
