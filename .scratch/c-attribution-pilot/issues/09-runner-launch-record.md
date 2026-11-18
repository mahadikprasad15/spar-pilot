# 09: Runner, launch record and teardown

**Status:** needs-triage
**Type:** task
**Blocked by:** 05, 07, 08

**What to build:** stages `preflight → qualification → rows → attribution → aggregate+freeze → generation → report → verified copy → HF export → summary → Pod stop`. Reuse the production-1 chain, units, watchdog and launcher where they fit; run under its own volume subtree (`$VOL/c-attribution/<run-id>/`).

- [ ] **Launch record:** every value type-checked at `seal`, nested leaves included (lessons pod-2 and review round 6: `19.5e12` read as text; nested numeric `step` falsely refused). Caps, Pod ID, stop command, tolerances, seeds and decisions all fixed; nothing chosen at run time.
- [ ] **Per-run state:** a new run ID per launch; volume state under the run ID, so an old run never blocks a new one (lesson pod-1 setup).
- [ ] **Preflight runs the real body:** disk ≥ 60 GB free on a 120 GB container disk, volume identity and separate mount, `HF_HOME` off the volume, CUDA, export repo private (lessons pod1, pod-1).
- [ ] **Budget** checked at unit boundaries, export time included; elapsed time persisted across resumes (lesson 2026-11-11).
- [ ] **Teardown:** sealed units only; verified volume copy; HF export with readback; summary written before the export and included in it (lesson 2026-11-14); stop tries `runpodctl stop pod <id>` then `runpodctl pod stop <id>`, logging output (lesson pod-v1).
- [ ] **Watchdog:** ownership taken only after the lock and precheck; a rejected second launch stops nothing (round 7); a startup crash before the chain's `try/finally` is covered (round 6); stop is never requested while a watchdog copy is in flight, with a bounded fallback (handover review P1 and P2). **No automatic rehearsal → production handover script.**
- [ ] **Resume** with an empty local disk restores from the volume, and expected work is reconstructed and compared with verified artifacts before any stage is skipped (lessons 2026-11-11 and round 5).
- [ ] Analysis outcomes never stop the chain; failed qualification does.

## Comments

## Answer
