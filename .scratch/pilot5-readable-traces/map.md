# Pilot 5 design map

## Notes

Readable traces of rank-1 SFT versus GRPO on unrelated fixed text. No new fine-tuning. Dedicated readable notebook planned. See design.md.

## Decisions-so-far

- 2026-10-14: checked community Jacobian lens first; own fitting needs separate approval.
- 2026-10-14: primary vector pools positions 0–4 at block 14; individual positions secondary.
- 2026-10-14: retained activation shards durable before GPU release.
- 2026-10-14: notebook exposes readable named stage calls and objects; process/scientific machinery is in package modules.

- 2026-10-14: bounded qualification for both zero adapters; exact diagnostic cohort pending.

- 2026-10-14: raw-only first execution approved; user/assistant forms deferred. Endpoint-only capture remains a recommendation pending explicit approval.

- 2026-10-14: assistant-form implementation approved as an optional separate notebook script, disabled by default, with separate plans/results. User form remains deferred.

## Fog

See design.md section 10 for unapproved choices and source-qualification prerequisites. Design is a draft, not a frozen execution spec or implementation ticket set.

## Specification and ticket preparation

- Public saved-workflow test seam confirmed by the user on 2026-10-14.
- spec.md published locally with ready-for-agent status; scientific execution review gates remain explicit.
- ticket-proposal.md contains 12 vertical slices and blocking edges, awaiting user approval before individual issue publication.
- Documentation checks passed. Current Python has no pytest; no implementation code was changed in this synthesis.

- Supplied ticket review reconciled: ticket-proposal.md revised to 13 slices; previous proposal preserved separately. Individual issues still await approval.

## Published implementation frontier

- Second review corrections applied; 13 individual ready-for-agent issues published under issues/.
- First frontier: 01, restore/source audit and early Colab smoke.
- 06 solely owns capture-affecting commitments and primary identity; 03 owns reveal/inference rules within the retained-data family.
- Required raw report/validation do not depend on optional grading/assistant work.
- No Pilot 5 implementation or paid execution has started.

## Implementation progress · 2026-10-14

- Ticket 01 claimed: bootstrap restore, source audit, pinned notebook and bounded environment/storage smoke implemented. Ten CPU tests pass.
- Real Colab smoke/reconnect evidence is still pending; Ticket 01 is not resolved.
- No Pilot 5 model capture, lens readout, grading or paid GPU execution has been run.

- Bootstrap published to GitHub main with explicit user approval. Repository regression completed: 279 passed, 35 skipped; new bootstrap tests: 10 passed.
