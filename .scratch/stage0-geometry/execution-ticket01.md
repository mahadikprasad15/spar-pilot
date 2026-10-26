# Ticket 01 implementation evidence

Date: 2026-10-26. Software complete; real source audit pending.

## Delivered

- `pilot_eval/stage0_sources.py`: prepared-source/config/checkpoint seals,
  actual CPU FP32 rank-one factors, exact A0 comparison, endpoint readiness and
  recorded training metadata. Reuses the existing CPU safetensors reader.
- `pilot_eval/stage0_workflow.py`: immutable plans, per-source sealed audit units,
  fresh-process status, explicit writer recovery, errors/progress, and a shared
  cumulative budget ledger. Unknown cost remains unknown. A published ledger
  head detects removal of trailing observations.
- Five public CLI operations and two shared notebook entrypoints. GPU notebook
  currently restores status only; all present cells can run on CPU.
- `docs/stage0-geometry.md`: source selection, outputs, restart, budget and
  explicit evidence limitations.

## Red/green evidence

The public prepare command was initially absent and the first workflow test
failed. Added behavior, then verified actual initialization comparison and resume.
Additional failing tests exposed missing training metadata, absent writer recovery
and a ledger-truncation rollback; each was implemented before the green run.

Final affected/regression command:

`pytest -q tests/test_stage0_workflow.py tests/test_trace_session.py tests/test_notebook_session.py tests/test_behaviour_breakdown.py`

Result: **68 passed in 40.61 seconds**, including **15 Stage 0 tests**. Session
baseline before changes was 485 passed, 35 skipped and 4 warnings. Some baseline
model tests were skipped due to the existing local Torch library failure.

## Release and artifacts

Implementation commit: `76c20655ae5acb1a26ed6dfa06d39e0831b3e5f8`.
Both notebook pins reference this implementation. A static Git-content comparison
verified the pinned command/source implementations match the tested files.

Structured local validation record:
`artifacts/validation/stage0-ticket01/verification.json`.
No real source run or GPU artifacts were produced. Generated fixture artifacts
were confined to pytest temporary roots. Source adapter payloads are not present
locally, so no actual SFT/GRPO A0 equality or historical KL availability is asserted.

The system `python3` resolves to unsupported Python 3.9; use the supported project
Python (tests use miniforge Python 3.12). Notebook commands use their kernel's
`sys.executable`, avoiding that interpreter mismatch. No global environment was
modified and no packages were installed.

## Next

Publish the release before opening its pinned Colab notebook. On CPU, run the
storage/checkout, source selection, audit and status cells. Preserve the saved
audit JSON for reviewing actual source facts. Ticket 02 can now implement the
spectral cache; optional Ticket 11 is also unblocked but not on the primary path.
