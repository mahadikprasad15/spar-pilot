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
