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
