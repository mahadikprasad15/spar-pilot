# Ticket 09: matched merged-weight intruder catalogues

## Implemented

- Public commands `stage0-intruder-edits-{prepare,run,status,verify,recover-writer}`.
  Parent principal catalogue and intruder geometry must verify and share sources.
- Pure matching/operator math: distinct top-40 controls, exact stored-FP64-value
  rational costs, minimum total absolute singular difference, index tie-break.
  Ratio sensitivity uses original matched pairs, inclusive [0.8,1.25], no rematch.
- Both-arm conditional catalogues: separate full, matched, matched control,
  ratio, ratio control and unhalved qualification matrices. Insufficient-control
  modules are excluded from both matched sides; full V6 remains independently
  available. No eligible flags yields a verified not-applicable receipt.
- Per-unit full FP32 matrices, per-module disable masks, hashes, operator checks,
  before/after values, timing/storage and immutable source identities. Matrices
  with identical selected directions share a payload. Source adapters/base
  weights remain unchanged.
- Mixed-loader interface compatible with the existing greedy engine. Original
  adapter stays in untouched modules; edited modules use saved merged matrices
  with that adapter disabled. Parameter objects and adapter flags restore after
  normal execution and failures. Unhalved double-application is bounded-diagnostic
  negative-control support, not a normal variant.
- CPU notebook sections 26–28 and `docs/stage0-geometry.md` handoff. GPU notebook
  does not acquire an unqualified inference stage.

## Tests and failure evidence

- Startup affected baseline: 20 passed in 67.03s (intruders and principal edits).
- Initial math tests red before implementation; first math slice passed.
- Public not-applicable test found an incorrect assumed parent metadata field;
  fixed by checking actual verified adapter payload identities.
- One positive synthetic prerequisite fixture hit the existing Ticket 03 V3-2
  projection gate: residual 1.0366522139637947e-6 exceeded 1e-6. The original
  failed fixture was preserved at
  `artifacts/validation/stage0-ticket09/failed-fixture-unbalanced-v1`.
  No upstream code or gate changed. The conditional positive test uses the same
  diagonal rank-one operator represented as B=1, A=150 rather than B=150, A=1,
  avoiding an irrelevant absolute-factor-residual boundary in its prerequisite.
- First complete public workflow/loader slice: 5 passed in 28.97s.
- Overflow-ratio test red on JSON Infinity; explicit null/status handling then
  passed, alongside exhaustive distinct-control assignment checks.
- Notebook handoff test red before sections 26–28 existed, then passed.
- Expanded conditional tests: 8 passed in 58.74s. Cover eligible/full/matched/
  empty-ratio cases, no-applicability, interrupted reuse, corrupted and resealed
  matrix rejection, source mutation, owner recovery, no-model CPU verification,
  mixed-scope error restoration, ratio endpoints and top-40 control filtering.
- Final affected regression and release pin are recorded at closeout.

## Execution boundary

Software fixtures only. No real Qwen matrices were constructed for this ticket,
no CUDA inference ran, and no Colab units were spent. Catalogue status remains
`gpu_qualified: false` and `measurement_complete: false` even after CPU build.
Ticket 10 owns bounded mixed-loader qualification (16 sequences, paired greedy
checks, double-adapter negative control, measured gate review), budget approval,
conditional collection and findings. The primary report is not blocked by it.

## Closeout, 27 October 2026

- Final affected regression: **52 passed in 230.18s**. This is an affected-suite
  run, not a full legacy-suite claim.
- A final focused rerun of the eligible-arm interruption/source-rejection test
  passed in 25.28s after making its partial-file path inventory-driven.
- Both notebooks parse, contain no saved execution outputs, and pass their
  public-workflow handoff checks. Diff whitespace checks pass.
- Verified eligible and not-applicable synthetic snapshots, independent matching
  oracle, the original failed prerequisite fixture and structured validation are
  retained under `artifacts/validation/stage0-ticket09/`.
- No real weight construction, model download, GPU run, Drive upload or push.

- Scientific implementation/notebook release pin: `549f612775df7663847fbcb029af593d005f0e75`.
