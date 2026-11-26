# 09: Runner, launch record and teardown

**Status:** resolved (2026-11-19); pre-GPU review findings fixed 2026-11-20 on CPU (see `review-implementation-09-10-response.md`)
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
- 2026-11-19 (implementation): built and committed; see Answer.

## Answer

- `pilot_eval/c_attribution_chain.py`: typed launch record (`SCHEMA`: every leaf, nested ones included, is
  type-checked at seal and load; unknown keys, bools, text-for-number, non-integer counts and `[proposed]`
  are refused); per-run volume state `$VOL/c-attribution/<run-id>/`; a finished run ID is refused and a
  different record under the same run ID is refused; no caps (elapsed per stage logged and persisted);
  automatic sign-off; resume from an empty local disk, skipping a stage only after its outputs re-verify;
  teardown copy -> pre-export summary -> export -> final summary -> stop with both syntaxes.
- `pilot_eval/c_attribution_stages.py` and `pilot_eval/c_attribution_backend.py`: the seven stage bodies
  composed from tickets 01-08 and the real backend.
- `scripts/c_attribution_launch.sh`; `scripts/attribution_watchdog.sh` accepts the deadline `none` and prints
  the writer's last progress update (`progress.json`) so a stalled writer is visible.
- Generation applies one frozen group per unit, checks the restored adapter hash after each unit (a mismatch
  raises before the unit is sealed), and refuses a batch plan other than the frozen one.
- Rehearsal record draft: `.scratch/c-attribution-pilot/launch-record-rehearsal.yaml` (values tagged
  [for approval] need the user's approval at ticket 11).
- Open for review: check (b) replay is recorded as not run (needs the source sampled path); cached-against-
  forced is `measure-only` in the rehearsal by design.
- Tests: test_c_attribution_chain.py (19), test_c_attribution_launcher.py (8), generation (+2).

