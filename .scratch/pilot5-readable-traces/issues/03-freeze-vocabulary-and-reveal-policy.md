# 03: Freeze vocabulary and inference commitments before outcome reveal

**What to build:** Review and freeze the task-enriched vocabulary and outcome-dependent analysis/grade rules while capture qualification and lens inspection proceed independently. The background vocabulary uses the actual declared corpus; this is why the corpus artifact is a genuine dependency.

**Blocked by:** 02: Prepare and freeze the raw corpus.

**Status:** claimed (implementation complete; real CPU reference preparation and review pending)

- [x] Public vocabulary preparation uses the fixed training-solution and background source, with explicit counts/smoothing/exclusions/ties and no learned token lists as input.
- [x] Freeze reveal-only decisions: vocabulary, null/correction/threshold rules, sample-size draw counts/seeds and SNR estimator within the allowed retained-data family, plus blinding/grade rubric. Reference the primary/capture contract; do not redefine its sites, positions, pooling, conditions, retained precision or primary identity.
- [x] Prior exposure records cover earlier pilots and first looks: named reviewer/decision-maker, which outputs or token lists were seen, when known, and unknown history. Scalar Pilot 4 plots are distinguished from learned lens lists; do not invent an earlier readout.
- [x] Freeze legacy-150/new-50 robustness analysis and its uncertainty/sample-size interpretation before reveal.
- [x] Missing approval blocks outcome publication; an existing-data preview or bounded adapted preview cannot bypass this barrier.
- [x] Outcomes revealed before freeze are marked exploratory and prior exposure is recorded; they cannot be relabelled confirmatory by a later freeze.
- [x] Every allowed analysis is a function of the retained-data capability contract. Requests for unsupported per-position uncertainty at other sites are rejected before analysis rather than silently creating a dependency on capture or promising a rerun-free result. Capture does not wait for vocabulary approval.
- [x] The notebook reveal barrier prevents accidental output through supported operations, not deliberate file access. Record its scope and any known bypass/exposure.

## Implementation evidence · 2026-10-15

The approved v1.1 design and preflight defaults supersede the original single-reference wording. Three references use questions (primary), gold solutions and accepted GRPO completions, against the separately selected 2M-token background. Public prepare/verify/review commands, sealed background resume, normalized scoring, seeded statistics and blinded grading utilities are implemented and CPU-tested. See `TICKET_03_REPORT.md` for exact scope and evidence.

The existing 200-document corpus has been verified. Notebook cells 10–13 add the CPU reference workflow. Real reference preparation/top-30 review has not run yet; no learned lists, real nulls, model capture or grader calls have been produced. The ticket stays claimed until the actual reference run and reviewer receipt are checked. Later capture/readout consumers retain their separate qualification gates.
