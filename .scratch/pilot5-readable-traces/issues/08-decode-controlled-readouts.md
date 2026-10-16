# 08: Decode saved means and reveal controlled token lists

**What to build:** After the vocabulary/inference commitments are frozen, read saved means, build primary/secondary vectors and expose the direct token lists with qualified readouts and controls. No decoder capture repeats.

**Blocked by:** 07: Capture, resume and durably verify raw evidence; 03: Freeze vocabulary and inference commitments before outcome reveal.

**Status:** claimed (software implemented and CPU-tested; real execution and prerequisite approvals pending)

- [ ] Public readout refuses to publish learned lists until the reveal barrier is satisfied, including earlier preview entrypoints.
- [ ] Pooled primary and selected secondary vectors preserve site/position/form identities; zero/unsupported/unqualified vectors remain explicit.
- [ ] Save positive/negative ranked token IDs, decoded strings and scores, norm/coherence and vector/lens provenance.
- [ ] Random-adapter, random-direction and base/adapted-mean controls remain distinct; do not recapture data to generate Gaussian-vector controls.
- [ ] Side-by-side logit/Jacobian display uses qualified artifacts or an explicit approved partial mode; useful readout units resume without decoder inference.

## Implementation handoff · 2026-10-16

See `TICKET_07_08_REPORT.md` and `docs/pilot5-capture-and-readouts.md`. Public saved-workflow commands and notebook stages are implemented with synthetic CPU execution evidence. No actual production or readout result is claimed. Ticket 06 qualification/scope/budget/predictions and Ticket 05 lens disposition remain explicit gates.
