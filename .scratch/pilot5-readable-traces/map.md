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
- No Pilot 5 production model capture has started. Bootstrap and corpus implementation are recorded below.

## Implementation progress · 2026-10-14

- Ticket 01 claimed: bootstrap restore, source audit, pinned notebook and bounded environment/storage smoke implemented. Ten CPU tests pass.
- Real Colab smoke/reconnect and all three source audits passed (user-reported public validator outputs). Ticket 01 resolved.
- No Pilot 5 model capture, lens readout, grading or paid GPU execution has been run.

- Bootstrap published to GitHub main with explicit user approval. Repository regression completed: 279 passed, 35 skipped; new bootstrap tests: 10 passed.

- Ticket 02 implemented with 17 CPU corpus tests; 32 related tests pass. Real pinned-data corpus preparation is pending. Cells 7–9 need CPU only.
- Approved new-50 selection: same pinned first-2,000 FineWeb pool, disjoint document/text identities, 128-content-token minimum, separate local seed 42, source order. Prior exposure remains explicitly recorded by the reviewer.

## Ticket 03 progress · 2026-10-15

Exposure history, verified-corpus review drafts and retained-evidence availability guards are implemented at the public CPU workflow seam. Scientific decisions remain empty; no reveal/freeze or learned readout is claimed. Real corpus completion and review approvals are pending. See `docs/pilot5-prereveal-review.md`.

## Corpus target verification · 2026-10-15

Ticket 02 resolved from user-reported public Colab verification: 200 documents, 150 legacy/50 new, 128 content tokens, no added special tokens, return code 0. Ticket 03's real-corpus dependency is satisfied; vocabulary/statistical approvals remain pending. No new GPU work is required to review those choices.

## Ticket 03 implementation handoff · 2026-10-15

Approved v1.1/preflight procedure implemented and verified: 73 affected CPU tests. See `TICKET_03_REPORT.md`. Shared notebook cells 10–13 prepare/verify/review references; fresh CPU 1–2 → 10–12, then explicit review 13. Real reference execution and receipt are pending, so Ticket 03 remains claimed. No learned readout, GPU capture or grader result is asserted.
