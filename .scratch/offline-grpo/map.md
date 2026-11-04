# Offline GRPO implementation map

## Notes

Tickets 01 and 02 were implemented as additive public workflows. Synthetic validation is separate from actual sampling or training.

## Decisions-so-far

- Ticket 01 is resolved for implementation and CPU validation. See its Answer.
- Ticket 02 is implemented and claimed pending actual pinned-TRL parity validation. See its Review status.
- Historical training and R0 source modules remain unchanged.

## Fog

- Historical raw pool availability, release/runtime compatibility and real GPU smoke remain unverified.
- Default local Torch fails to import and the alternate working CPU environment lacks pinned TRL.

- Ticket 03 is implemented and claimed pending actual pinned-TRL and real-GPU evidence. Pre-update token diagnostics are linked to accepted history and window-one push agreement gates the optimizer step. Integrated CPU validation passed 17 tests with one skip.

- Ticket 04 behavioral source handoff and evaluation/report commands are implemented. Checkpoint-zero token comparison gates nonzero evaluation. Regression passed 11 tests with one skip, with 3 focused tests after final manifest addition. Actual GPU evaluation awaits verified real offline training and runtime evidence.

- Ticket 05 additive dataset scoring and reporting is implemented and claimed for review. Synthetic CPU fixtures validate resume, multiplicity and saved-score identity; the inspected historical product cannot be reused until its missing source plan is restored. No GPU scoring has run.

- Ticket 05 implementation and CPU verification resolved. Public dataset scorer/report supports verified historical score reuse with exact token contracts, all pool memberships and derived paired cohorts. Combined regression 15 passed; final focused 4 passed. Actual historical source graph restoration and GPU scoring remain execution work.

- Ticket 06 software implementation reviewed: additive offline source, geometry, initial-push and fixed-context drift collectors and report. Combined regression 34 passed, 3 skipped; final focused suite 8 passed. CPU fixtures do not establish actual GPU results; Gold and Sampling SFT internal-source comparisons remain explicitly unavailable.

- Ticket 07 software implementation resolved. Eight principal-cut conditions, batch agreement, resumable paired evaluation and JSON/Markdown reports are available. Combined regression passed 13 tests before the last performance refactor, then final focused verification passed 5. Real preparation awaits restoration of the historical cut plan, and GPU measurements remain pending. The original endpoint is freshly evaluated.

- Ticket 08 software implementation resolved; see its evidence. Code review follow-up added a shared attempt lifecycle (`offline_grpo_support.py`) that adopts the Stage 0 rule for sealed units from killed or expired attempts, fails on re-saved checkpoint-0 bytes before training, binds pool sampling to the frozen online settings, states a missing online comparison in reports, records external export sources, and pairs checkpoint N with window N+1 in descriptive slopes.

- Pool route selection added: `offline-pool-route` runs after pool preparation and seals a decision. It keeps the historical 128 groups and samples the remaining 384 only when the source audit, recorded runtime and exact two-group replay all agree; otherwise it records why and samples 512. CPU fixtures cover both routes; whether the historical raw groups exist and replay on a real GPU remains unverified.

- External review follow-up: a detected integrity failure now records a failing closing result, so its units are not reused as if interrupted; a replay's failed base check stops route selection; replay draws and integrity receipts are bound to the decision; backups carry the specification, wrapper state, attempt timing and budget ledger, with `offline-grpo-overnight-restore` to put them back; the final report lists omitted comparisons. Restore-and-resume is fixture-tested with mocked child verifiers at the same root and session.

- Prelaunch requirements reduced at the researcher's direction: predictions, reporting protocol, whole-run estimate and budget ledger are no longer required by the overnight wrapper. Approval has an optional `max_seconds`; without it a 30-day fault guard is the only time bound. Standalone child commands still use the Stage 0 ledger.

- The overnight run now generates the spectral cache as a CPU stage with the existing `stage0-spectral-*` commands, so it is no longer a template input. Geometry and everything after it depend on that stage; pool, training and behaviour do not. The spectral plan and cache are never exported; a resumed run regenerates them from the same pinned inputs.

- Route rule relaxed at the researcher's direction: the replay and hardware differences are recorded, not decisive, so an A100 can keep the 128 historical groups and sample 384. See the spec's decision update.

- Online behaviour comparison now runs across runtime differences, recording them, at the researcher's direction. See the spec's decision update.

## Open decisions

- Fidelity tolerance: `offline-grpo-train-check-diagnostics` now requires an explicit `--fidelity-tolerance` in nats (template input `fidelity_tolerance_nats`). No value has been chosen; a reviewed number is required before a real run.
- A new session or machine still needs a new plan, because plans bind the runtime session identity. A stale `writer.json` after a hard kill still requires inspection and removal by hand.
- Completion does not imply every comparison was made: Gold and Sampling SFT internal comparisons are unavailable and the online comparison can be omitted. Read `omissions` in the final report; decide whether any should block completion.
