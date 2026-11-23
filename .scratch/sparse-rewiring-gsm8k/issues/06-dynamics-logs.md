# 06: Dynamics logs and settle diagnostics

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 03
**Group:** C

Pure functions that turn gate and factor state into the per-step rows (counts, per-layer counts, norms), the every-16-step `logα` snapshot, and the every-64-step settle diagnostics (Jaccard, expected-active change). Logged only (D5, D20).

- [ ] Red/green: spec tests 14 and 18.
- [ ] Zero-signal state at step 0 (all changes zero, empty sets) is defined, not divided by zero.

## Comments

## Answer
