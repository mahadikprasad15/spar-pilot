# 18: Report and catalog registration

**Status:** ready-for-agent
**Type:** task
**Blocked by:** 10, 11, 12, 13, 17
**Group:** J

Register both arms, checkpoints, response sets and new cohort prior uses with `c_attribution_report.register`; `catalog verify` passes; write `report.md` and update `docs/experiments/status.md`.

- [ ] Round-trip the report through the real serializer before publishing.

## Comments

- 2026-11-25: `pilot_eval/rewire_report.py` (`build`, `register`) and `tests/test_rewire_report.py`, on a real tiny
  production chain's volume copy: report from sealed units only, JSON round trip before `report.md`, registration of
  both arms (checkpoint = `rewire_state.safetensors`), every greedy unit (val128 at each checkpoint, references,
  cohort150 at the final step), the `gsm8k-train-val128` cohort and prior uses; `catalog verify --root sl-0=...` now
  resolves a mounted remote location. Analyses are read from `sparse-rewiring/analysis/` when sealed there:
  `rank-comparison/rank_comparison.json`, `standalone-<arm>/standalone_analysis.json`, `attribution/overlaps.json`,
  `dynamics/*.png`. Left for after the run: `docs/experiments/status.md`.

## Answer
