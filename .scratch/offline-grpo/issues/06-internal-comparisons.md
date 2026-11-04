# 06: Internal measurements and comparisons

Status: ready-for-agent
Blocked by: 02 Train and resume fixed-pool GRPO; 03 Pre-update diagnostics and window-one check

## What to build

Use verified offline adapters with existing collectors for initial push, product-level adapter directions and norms, fixed-input writes and drift, and spectral geometry. Keep historical source contracts intact and explicitly report unavailable or incompatible products. Relate offline–online layer similarity to pre-update diagnostics descriptively.

## Acceptance criteria

- [ ] Existing collectors accept explicit verified offline sources while valid legacy sources remain analyzable; CPU fixtures exercise both and reject identity or aggregation mismatches.
- [ ] Results compare adapter products and effective writes with matched definitions, distinguish geometric observations from behavioural causes, and identify the separately computed actual first-window push.
- [ ] Report observed KL and adapter-norm trajectories without inferring matched-norm KL; an actual scaled-adapter matched-norm measurement is a deferred extension.
- [ ] Fixture tests verify source handoff and a representative aggregation; real measurements require their verified inputs and authorized execution.
