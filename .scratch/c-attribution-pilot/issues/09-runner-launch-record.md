# 09: Runner, launch record and teardown

**Status:** needs-triage
**Type:** task
**Blocked by:** 05, 07, 08

**What to build:** the run's stages, in order:

1. preflight
2. qualification
3. rows
4. attribution
5. aggregate, split and freeze
6. greedy generation
7. report
8. verified copy
9. HF export
10. summary
11. Pod stop

Reuse the production-1 chain, units, watchdog and launcher where they fit. Run under the pilot's own volume subtree, `$VOL/c-attribution/<run-id>/`.

- [ ] **Launch record.** Every value is type-checked at `seal`, nested leaves included (lessons pod-2 and round 6). It fixes:
  - the Pod ID and stop command;
  - tolerances (production-1 tolerances unchanged; the new ones set by the user);
  - seeds;
  - the hot-layer list and its provenance;
  - the resolved decisions.

  Nothing is chosen at run time.
- [ ] **Per-run state.** A new run ID per launch, and volume state under that run ID (lesson pod-1 setup).
- [ ] **Preflight runs the real body:**
  - at least 60 GB free on a 120 GB container disk;
  - volume identity, on a separate mount;
  - `HF_HOME` off the volume;
  - CUDA available;
  - the export repo is private;
  - the catalog is present.
- [ ] **No time caps** (user, 2026-11-18). Elapsed time is still logged and persisted across resumes, for planning only.
- [ ] **Intervention safety.**
  - Each generation unit applies exactly one frozen group and verifies the restored adapter hash.
  - A hash mismatch stops the chain.
  - The fixed batch membership is checked against the freeze.
- [ ] **Teardown.**
  - Sealed units only.
  - A verified volume copy.
  - HF export with readback.
  - The summary is written before the export and included in it (lesson 2026-11-14).
  - Stop tries `runpodctl stop pod <id>`, then `runpodctl pod stop <id>`, logging the output (lesson pod-v1).
- [ ] **Watchdog.**
  - Ownership is taken only after the lock and the precheck.
  - A rejected second launch stops nothing (round 7).
  - A startup crash is covered (round 6).
  - Stop is never requested while a copy is in flight; there is a bounded fallback.
  - Adapt the old watchdog's required absolute deadline explicitly: in this pilot, a live writer is
    never stopped merely because elapsed time has passed. CPU-test the real watchdog with the
    deadline disabled, including a writer crash and a still-running writer. Keep visible progress
    and last-update timestamps so an alive but stalled writer can be recognized.
  - **No automatic rehearsal → production handover.**
- [ ] **Resume.** With an empty local disk, the run restores from the volume, and expected work is reconstructed and compared with verified artifacts before any stage is skipped.
- [ ] **What stops the chain.** Analysis outcomes, including stability or unavailable controls, never stop it. Failed qualification or input integrity does.

## Comments

- 2026-11-18 (round 1): caps now come from rehearsal measurements. Restore-hash and fixed-batch safety were added (F4, F5).

## Answer
