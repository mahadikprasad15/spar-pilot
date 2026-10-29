# Sampling SFT design map

## Notes
- Original supplied design preserved in sources/; working design in design.md.
- Read-only source/upstream audits completed; targeted baseline: 5 passed.
- Rounds 1–3 answered on 2026-10-28. No implementation or cloud work begun.

## Decisions so far
- Data-recipe claim; descriptive mechanism measures; common-coverage k contrasts.
- Generate k0/2/5; later approved training k0/k5 only; k2 training conditional and separately approved.
- No automatic merge or causal/equivalence claim.
- Coverage threshold 0 / 1–5 / >5; smaller-batch edge cases remain open.
- Length-based extension with doubled block safety bound; four terminal states and bounded retries.
- Frozen sampling configuration and exact pilot restart/resume qualification.
- USD25 ceiling; sampling10/training6/storage-and-slack9; 80% review checkpoints.

## Decisions added in round 2
- T16 covers batches3–7; hand loss, finite gradients, one scheduler step; no duplication.
- Three-arm common gate cohort; arm-specific training coverage; flag pairwise uncovered-set differences >5.
- Per-attempt terminal states, per-problem dispositions, separate interruption events.
- Exact block-boundary replay, prefix-cache resets and ordered requalification.
- Immutable automatic scores plus confirmed human eligibility overrides.
- Conservative reservations, ongoing storage lifetime, stage-attributed costs and automatic post-export stop; deletion requires approval.

## Decisions added in round 3
- Defer512×8 temperature1 baseline and every NLL-offset column; no historical reuse.
- T3: exact alignment, mean error<=0.001 nats/token and max token error<=0.01; fixed coverage of short/long/stop/cap/empty-prefix fixtures.
- J64 fixed; initial pilot unfinished>2%, correct<90%, or over-budget Stage2 reservation stops for review.
- Equal-problem NLL including stop; common-cohort paired bootstrap10000/seed42/percentile95%; differences of medians for length; approved leakage diagnostics.

## Remaining facts/preflight
- Actual training runtime identity, engine compatibility/numeric flags and stop semantics.
- Deterministic fixtures/metric serialization and bounded profiling.
- Atomic resume, export/download receipts and confirmed Pod shutdown.
- Full local regression:644 passed,39 skipped,4 warnings in2578.05s; some Torch-backed checks skipped due a broken local Torch import.

## Final confirmation
User confirmed the consolidated design and checklist on2026-10-28. Explicit additions: data seed0, L40S sampling, verified L4 training, no gold rerun, exact checkpoint0 and gold logs, optional optimizer/gradient capture. Weights-only mid-training resume is invalid; without recovery state restart from checkpoint0 under the budget policy. First sampling-GPU spend needs a separate post-preflight go-ahead.

## Specification and proposed tickets
- User requested to-spec and to-tickets; approved the public saved-workflow test boundary on2026-10-28.
- spec.md published locally with ready-for-agent status.
- User approved all10 slices and edges; issues/ contains10 separate ready-for-agent tickets. Tickets01–02 software implemented and resolved;03–10 remain ready-for-agent, unclaimed.
- Source original and consolidated design remain preserved; no implementation or cloud launch started.

## Next
Ticket03 (bounded sampler, score qualification and profile) is the next implementation frontier, awaiting a request. No paid job or push is implied.

## Ticket01 implementation evidence
- Public prepare/audit/status/verify and reviewed writer recovery; clean CPU notebook and guide.
- Exact source/prompt/order/checkpoint identities and blocked missing-evidence reports; engine controls/runtime remain explicitly unqualified.
- Red/green workflow development; final affected regression47 passed in114.26s; full notebook execution and schema validation passed.
- Canonical synthetic demonstration: artifacts/validation/sampling-sft-ticket01/passing-cpu-fixture-v1, latest plan source-fixture-v2. No real Drive audit, GPU qualification, inference, training, paid launch or push.
- Session full baseline intentionally interrupted as duplicate after253 passed/24 skipped. Inherited later full run completed669 passed/39 skipped; collection preceded Ticket02 and is not a final commit-isolated suite claim.

## Ticket02 implementation evidence
- CLI/run script primary; optional notebook budget and disabled offline smoke cells.
- Shared atomic USD ledger, storage/interim-cost handling,80% review and conservative launch reconciliation; no new stage-specific budget authority.
- Verified bundle/client receipt/own-Pod-stop handshake, external shutdown proof and actual-bill settlement; trusted scientific entrypoints only.
- Final budget/lifecycle suite37 passed in11.65s; earlier combined affected regression78 passed in84.35s; notebook schema/execution checked.
- Canonical demo artifacts/validation/sampling-sft-ticket02; simulation only, no live provider calls or model/GPU work.
- Real RunPod/storage preflight remains pending; Ticket03 unclaimed. No new commits pushed without a request.

## Ticket03 implementation
- Resolved software slice: bounded likelihood search, score and throughput workflow; public commands and runbook in docs/operations/sampling-sft-engine.md.
- CPU regression78 passed; artifacts/validation/sampling-sft-ticket03/validation.json. Real L40S/T3/profile execution remains pending, not inferred from fixtures.
- Ticket04 is next: fixed32 pilot and exact interruption/resume. No paid work or publication authorized.

## Ticket04 implementation
- Resolved software slice: fixed32 k5 pilot, block-sealed recovery, real fault-injection entrypoint, exact scientific replay and CPU alarm/forecast report. See docs/operations/sampling-sft-pilot.md.
- Combined88 CPU tests passed; actual local SIGKILL/fresh-process demonstration saved under artifacts/validation/sampling-sft-ticket04. CPU evidence stays unqualified for L40S.
- Runtime/profile/T3/real pilot execution and paid approval remain pending. Ticket05 production generation is the next implementation slice.

## Ticket05 implementation
- Resolved software slice: full k0/k2/k5 collection with fixed512 source problems, bounded model retries, separate engine failures, block-sealed resume and reconciled automatic dispositions.
- Stage2 real qualification/plan review and active budgeted paid-job guards precede model loading. Completed collection is separate from human coverage review and training approval.
- Combined98 CPU checks passed in169.44s; relocated synthetic three-arm evidence verified under artifacts/validation/sampling-sft-ticket05. No real GPU execution, spending or push.
- CLI/runbook remains primary; optional notebook adds CPU generation-plan/status only. Ticket06 is next.

## Ticket06 implementation
- Resolved software slice: source-bound intended-answer review, separately confirmed human/agent revisions, immutable automatic scoring and reviewed coverage with original marked slots. Runbook:docs/operations/sampling-sft-review.md.
- Combined111 CPU checks passed in311.75s; raw synthetic review, XML and validation receipt persisted under artifacts/validation/sampling-sft-ticket06. No real audit/paid execution or push.
- Coverage conditions remain0 passes coverage only;1–5 requires T16 and later approval;>5 blocks. Common cohorts and pairwise uncovered differences are reported without changing arm-specific exposure.
- Ticket07 descriptive gate report is next; actual collection, human decisions and training approval remain separate.

## Ticket07 implementation
- Resolved software: independent resumable HF score collection, CPU recipe gate tables/paired intervals/plots and explicit source/report/budget-bound training-selection intent. Runbook:docs/operations/sampling-sft-gate.md.
- User approved unicode-words-v1 and occurrence-weighted4-gram overlap on2026-10-29; short responses undefined. No baseline/offset, automatic equivalence or causal attribution.
-72 affected CPU checks passed in249.06s;13 review/UI baseline checks passed separately. Raw synthetic report/receipt under artifacts/validation/sampling-sft-ticket07; no real model work or paid execution.
- Training remains unauthorized until Ticket08 readiness and Ticket09 paid controls. Ticket08 is next. Unrelated Stage0 changes preserved; local commits only.

## Ticket09 implementation — 2026-10-29

- Added selected k0/k5 training, fresh paid-job gates and exact original-step
  boundaries. CLI/RunPod remains primary; notebook17–18 only prepare/inspect.
- User approved preserving longer sampled targets, recording each arm's derived
  maximum and stopping at model context overflow; no truncation.
- Recovery is optional at declared checkpoints0/8/16/32/64; adapter-only resume
  rejects. A new approved/reserved attempt either restores complete state or
  explicitly restarts from gold checkpoint0. No source sampler-module changes.
-9 Ticket09 CPU checks and48 affected checks passed. Raw validation artifacts:
  artifacts/validation/sampling-sft-ticket09/. Runbook:
  docs/operations/sampling-sft-training.md. Real L4 execution is not performed.
- Ticket08 full T16 remains an actual-execution blocker. Ticket10 is the remaining
  software handoff slice; no training or paid launch has been authorized here.
