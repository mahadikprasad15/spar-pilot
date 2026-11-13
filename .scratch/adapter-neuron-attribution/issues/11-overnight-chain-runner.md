# 11: Overnight chain runner

**Status:** ready-for-agent — CPU implementation approved by the user 2026-11-13 (overnight requirement, spec §Overnight unattended chain). No GPU run, value or budget is approved by this status.
**Type:** task
**Blocked by:** 05

**What to build:** `pilot_eval/attribution_chain.py` (plus a launcher script) that runs preflight → qualification → base traces → attribution and freeze → Stage C → Stage D → report → volume copy → HF export → Pod stop unattended from a sealed launch record, reusing `neuron_audit_pod` patterns.

- [ ] Launch record: loads and verifies a sealed record (all fixed values, caps, Pod ID, stop command, code commit); refuses to start if any value is still `[proposed]` or null.
- [ ] Automatic sign-off after qualification by the record's rule; stops before production on any listed failure; never stops on analysis outcomes.
- [ ] Per-stage and total hard caps at unit boundaries with persisted elapsed budget; projections advisory only; safe-boundary stop with verified partial export.
- [ ] End of run on every path (success, failure, cap, exception), in order: stop and join writers; copy sealed, hash-verified units to the volume (timeout); HF export with readback (3 attempts, timeouts); finalize the sealed morning summary including both backup outcomes (never label unverified as verified); write it to the volume; request the Pod stop with both command forms (timeouts), recording requested vs externally verified. Do not reuse `may_stop_pod` for this decision (it requires a verified HF restore; the user's policy stops regardless).
- [ ] Independent watchdog (shell, outside the chain's Python), timed from the persisted remaining cap + margin: signals the writer, waits a bounded time, copies only sealed verified units (timeout), writes a watchdog summary, requests the stop (timeouts); cancelled cleanly on normal completion.
- [ ] Resume: only when this run's previous worker is verifiably gone and the volume identity and sealed units match the launch record; then take over its own marker. Live writer, foreign run ID or ambiguity → stop for investigation. The marker is advisory, not a distributed lock; never set `resume_own=True` blindly.
- [ ] Rehearsal mode validates against its own labelled two-arm expected set; production refuses missing arms.
- [ ] Tests (CPU, all external calls mocked): every exit path ends in writers joined → volume copy → export → sealed summary → stop requested; HF failure still stops; hung copy and hung stop command time out; neither backup verified is reported as such; killed-worker stale marker resumes, live writer and ambiguous marker stop; unavailable matched group is skipped, not fatal; longest-first packing does not stop a healthy run; qualification failure stops before production; analysis failure does not stop; overrun ends at a unit boundary; watchdog fires on a hung stage; stop-command fallback; launch record with a null value refused; resume skips sealed units.

## Comments
