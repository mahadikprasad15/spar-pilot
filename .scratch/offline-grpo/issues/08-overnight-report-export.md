# 08: Overnight orchestration, report and verified export

Status: resolved (software; real execution pending)
Blocked by: 03 Pre-update diagnostics and window-one check; 04 Behaviour and offline source handoff; 05 Dataset analysis; 06 Internal measurements and comparisons; 07 Principal cuts and paired evaluation

## What to build

Provide one resumable overnight driver with a preflight report, dependency-aware stages, visible progress, final scientific report and private-HF export with independent manifest readback. Following one explicit prelaunch authorization, apply the approved checks as automatic pass/fail gates without routine approval pauses. Record any failure and continue independent work whose inputs remain valid; stop affected work on data-integrity or numerical failure.

## Acceptance criteria

- [x] Preflight records source and runtime identity, storage, access without exposing credentials, ledger and reservation constraints, and compatibility decisions; changed runtime or code uses a new explicit plan without altering old evidence or thresholds.
- [x] Status, seals and checkpoints support resumption at verified boundaries, prevent duplicate draws or updates, and mark completion only after all required outputs verify; progress is readable while running.
- [x] CPU end-to-end fixtures cover interruption, missing inputs or credentials, child failure, independent continuation, partial export and readback mismatch; failed or incomplete work remains honestly visible.
- [x] Export includes the required plans, raw records, diagnostics, reports, logs, adapters and recovery state, and is called verified only after independent private-HF readback matches its manifest. Fixture success does not authorize an actual launch or export.

## Implementation evidence

`offline-grpo-overnight-{template,prepare,preflight,run,status,verify}` provide the resumable driver. CPU fixtures run the real report, manifest, upload and readback against a local fake Hub, and cover child failure, independent continuation, partial export, readback mismatch and recovery, interruption, a killed attempt, missing sources and credentials. A completed run re-verifies instead of exporting again. These fixtures do not authorize or establish a real launch or export. The preflight parity gate stays closed until the pinned-TRL test from ticket 02 passes.
