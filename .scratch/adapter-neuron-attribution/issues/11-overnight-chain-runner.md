# 11: Overnight chain runner

**Status:** ready-for-agent — CPU implementation approved by the user 2026-11-13 (overnight requirement, spec §Overnight unattended chain). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 05

**What to build:** `pilot_eval/attribution_chain.py` (plus a launcher script) that runs preflight → qualification → base traces → attribution and freeze → Stage C → Stage D → report → volume copy → HF export → Pod stop unattended from a sealed launch record, reusing `neuron_audit_pod` patterns.

- [ ] Launch record: loads and verifies a sealed record (all fixed values, caps, Pod ID, stop command, code commit); refuses to start if any value is still `[proposed]` or null.
- [ ] Automatic sign-off after qualification by the record's rule; stops before production on any listed failure; never stops on analysis outcomes.
- [ ] Per-stage and total caps at unit boundaries, overrun projection rule, safe-boundary stop with verified partial export.
- [ ] End of run on every path (success, failure, overrun, exception): morning summary (`run_summary.md` + JSON), verified volume copy, HF export with readback (3 attempts), Pod stop with both command forms and logged output.
- [ ] Independent watchdog process started at launch (shell, outside the chain's Python): fires at total cap + margin, copies local outputs to the volume, stops the Pod; cancelled cleanly on normal completion.
- [ ] Resume: a relaunch after interruption continues from sealed units without redoing them.
- [ ] Tests (CPU, all external calls mocked): every stop path ends in summary → volume copy → stop; HF failure still stops; qualification failure stops before production; analysis failure does not stop; overrun ends at a unit boundary; watchdog fires on a hung stage; stop-command fallback; launch record with a null value refused; resume skips sealed units.

## Comments
