# 06: Internal measurements and comparisons

Status: resolved (software; real measurements pending)
Blocked by: 02 Train and resume fixed-pool GRPO; 03 Pre-update diagnostics and window-one check

## What to build

Use verified offline adapters with existing collectors for initial push, product-level adapter directions and norms, fixed-input writes and drift, and spectral geometry. Keep historical source contracts intact and explicitly report unavailable or incompatible products. Relate offline–online layer similarity to pre-update diagnostics descriptively.

## Acceptance criteria

- [x] Existing collectors accept explicit verified offline sources while valid legacy sources remain analyzable; CPU fixtures exercise both and reject identity or aggregation mismatches.
- [x] Results compare adapter products and effective writes with matched definitions, distinguish geometric observations from behavioural causes, and identify the separately computed actual first-window push.
- [x] Report observed KL and adapter-norm trajectories without inferring matched-norm KL; an actual scaled-adapter matched-norm measurement is a deferred extension.
- [x] Fixture tests verify source handoff and a representative aggregation; real measurements require their verified inputs and authorized execution.

## Implementation evidence

Additive offline source, exact-SVD geometry, bounded initial-push and fixed-context drift collectors, and structured/Markdown reports are implemented. CPU fixtures exercise source-plan and review tampering, rank-one gauge invariance, separate window-1 and dead-group denominators, the existing numerical validators, interrupted drift resume, sealed payload corruption, geometry and report aggregation. These fixtures do not qualify the real model or establish scientific measurements. Gold and Sampling SFT internal-source comparisons are explicitly unavailable in this version; online comparisons are included only when selected and verified. Real GPU execution, qualified training and resource approval remain pending. Reviewed software verification: combined regression 34 passed, 3 skipped; final focused suite 8 passed after source and report hardening. Real measurements remain execution work.
