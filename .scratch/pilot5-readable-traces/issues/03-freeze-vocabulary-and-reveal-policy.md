# 03: Freeze vocabulary and inference commitments before outcome reveal

**What to build:** Review and freeze the task-enriched vocabulary and outcome-dependent analysis/grade rules while capture qualification and lens inspection proceed independently. The background vocabulary uses the actual declared corpus; this is why the corpus artifact is a genuine dependency.

**Blocked by:** 02: Prepare and freeze the raw corpus.

**Status:** claimed (partial safeguards implemented; scientific approvals pending)

- [ ] Public vocabulary preparation uses the fixed training-solution and background source, with explicit counts/smoothing/exclusions/ties and no learned token lists as input.
- [ ] Freeze reveal-only decisions: vocabulary, null/correction/threshold rules, sample-size draw counts/seeds and SNR estimator within the allowed retained-data family, plus blinding/grade rubric. Reference the primary/capture contract; do not redefine its sites, positions, pooling, conditions, retained precision or primary identity.
- [ ] Prior exposure records cover earlier pilots and first looks: named reviewer/decision-maker, which outputs or token lists were seen, when known, and unknown history. Scalar Pilot 4 plots are distinguished from learned lens lists; do not invent an earlier readout.
- [ ] Freeze legacy-150/new-50 robustness analysis and its uncertainty/sample-size interpretation before reveal.
- [ ] Missing approval blocks outcome publication; an existing-data preview or bounded adapted preview cannot bypass this barrier.
- [ ] Outcomes revealed before freeze are marked exploratory and prior exposure is recorded; they cannot be relabelled confirmatory by a later freeze.
- [ ] Every allowed analysis is a function of the retained-data capability contract. Requests for unsupported per-position uncertainty at other sites are rejected before analysis rather than silently creating a dependency on capture or promising a rerun-free result. Capture does not wait for vocabulary approval.
- [ ] The notebook reveal barrier prevents accidental output through supported operations, not deliberate file access. Record its scope and any known bypass/exposure.

## Implementation progress · 2026-10-15

- Public CPU exposure-recording/inspection and corpus-bound draft preparation are implemented in `pilot_eval/trace_review.py` and the CLI. Red tests preceded each committed slice.
- Immutable event IDs preserve prior history; changed payloads fail verification. Scalar results, learned token lists and other outputs are explicitly distinguished. Unknown history/date must be stated.
- Availability validation rejects per-position example analyses outside blocks 14 and 17–21 and rejects all-site pooled-window requests other than positions 0–4. Capture ownership is unchanged.
- Review drafts bind verified corpus inputs and keep all six scientific sections empty. They never grant reveal. No production freeze, vocabulary ranking, readout consumer or exploratory/confirmatory classification has been implemented.
- Prasad has been asked to approve the proposed vocabulary rule, review the remaining settings and confirm the real corpus run. No answer/approval is inferred from elapsed time.
- Ticket remains unresolved; Ticket 02's real pinned-data run remains unconfirmed. See `docs/pilot5-prereveal-review.md` for exact boundaries.
