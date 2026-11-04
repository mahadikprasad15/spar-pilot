# 03: Pre-update diagnostics and window-one check

Status: claimed
Blocked by: 02 Train and resume fixed-pool GRPO

## What to build

Record aligned per-token data-policy and pre-update learner log probabilities, masks and grouping for each consumed window. Report unclipped sequence sums, token-weighted rates and distributions with their assumptions. Independently compute the actual first training window's push and compare it to the training gradient.

## Acceptance criteria

- [ ] CPU fixtures verify alignment, grouping, sign, pre-update timing, sequence versus token aggregation, identical-policy zero under a predeclared criterion, and retention of finite negative estimates.
- [ ] The first-window check uses the identical batch, normalization and parameter ordering: the trainer's pre-clip gradient equals the negative of the independently computed negative-gradient push within a reviewed, predeclared criterion.
- [ ] The real-run gate records generation-score versus teacher-forced μ fidelity including EOS and a long or capped example, and a same-input GPU smoke with finite values, timing and memory; fixtures alone do not claim these passed.
- [ ] Outputs distinguish the stated sampling assumptions for forward KL from an empirical fixed-data log ratio, and do not clamp estimates or claim fixed-context KL equivalence.

## Comments

Implementation under review; ticket 02 remains scientifically unqualified. Focused CPU tests: eight passed and one pinned-TRL test skipped because the pinned stack is unavailable. A separate working-Torch fixture exercised token alignment, state restoration, an independent gradient oracle and failure before optimizer step. The real-CUDA preflight command is implemented but has not been run; generation-score fidelity, actual GPU smoke and pinned-TRL parity remain pending. Fixture results do not establish scientific readiness.

## Root verification

Integrated pool, training and diagnostic tests passed 17 tests with one pinned-TRL skip. The working-Torch diagnostic fixture passed independently, including failure-before-update and state restoration. The real generation-score command records raw differences and coverage; it does not assert numerical fidelity using an unapproved tolerance. Actual GPU evidence and pinned-TRL parity remain pending.
