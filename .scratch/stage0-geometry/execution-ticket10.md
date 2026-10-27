# Ticket 10: conditional intruder evaluation and findings extension

## Scope and implementation

- `stage0-qualification-prepare --catalogue`: verifies the conditional catalogue,
  binds all matrix/mask/control identities, excludes empty sensitivity variants
  from production and stops on a verified not-applicable catalogue. Primary
  `--edits` remains the default alternative; the inputs are mutually exclusive.
- Same first 16 held-out sequences per distinct applicable source-arm/module-mask
  layout. Full/matched/ratio masks can differ; identical masks share one diagnostic.
  Use unhalved saved matrices for each layout, against its original adapter.
- `MixedQualificationEngine`: actual shared mixed-loader scope, teacher-forced
  answer logits, greedy common-prefix evidence, intentional double adapter and
  original parameter-object restoration. Per-position errors and equality flags
  are reconciled against recorded positions/tokens; token manifests are frozen
  rather than silently overwritten during resume. Full integrity is checked once
  before diagnostic publication, with parameter guards between units.
- Existing numerical rules remain: E>1e-3 stops; per-case negative error >=1000x
  real error and non-vacuous; only measured shared-prefix near-ties may explain a
  greedy mismatch. Exact 3E review and distinct measured production-budget review
  are still required. No synthetic fixture receipt authorizes CUDA.
- Qualification status and explicit owner-token recovery are public commands.
  Live same-host writers and incorrect tokens are refused. Recovery releases only
  ownership, retaining seals, reservations and cumulative cost evidence.
- Shared greedy collector now counts the frozen variant inventory, instead of
  assuming exactly ten arms. It reruns both references with all conditional edits
  in the same runtime; EOS/cap/reward/scoring semantics and singleton real batches
  remain unchanged. Each response retains its full identity/token/text record.
- Shared fixed-context collector schedules the conditional variants plus the same
  bounded zero controls on the original 300 input sequences. Full-vocabulary
  KL(variant || untuned), chunk checks, scalar arrays and sufficient-sum schemas
  retain their original meanings.
- In merged modules, effective writes use the full (W_edit-W0)x action, promoted
  to FP64 for analysis, on actual adapted-model module inputs. Untouched modules
  use the original rank-one branch checks. Do not claim a rank-one reconstruction
  for the full edit; its sum_factor_norm_products is explicitly null. No new
  surrogate norm or numerical allowance is chosen.
- Current-variant read-only CPU matrix cache avoids rereading payloads for every
  prompt. Cache lifetime is bounded to one variant, clears on switch/close, and
  full source verification is repeated at attempt end. An altered source marks
  integrity failed and invalidates attempt evidence through existing collectors.
- Conditional findings freeze their own contrast family. Every executable edit
  is compared to its own reference, with direct matched/control and retained-pair
  sensitivity comparisons where applicable. Full/matched/ratio scopes, skipped
  modules, undecidable modules and empty sensitivities stay visible.
- Existing 10,000-draw paired bootstrap, seed 42, pointwise 95% intervals and strict
  lower-bound > -5 pp preservation rule are reused. Existing blinded targeted
  review, exact-text label reuse and partial-review report snapshots work for the
  conditional family. Primary findings remain independently runnable.

## Shared notebook interface

Both notebooks select `CONDITIONAL_EXTENSION` in section 1; default False keeps
primary products. Build/verify conditional CPU sections 23–28 first, then select
True in the GPU notebook and use its existing qualification and collection cells
5–19 after setup. Use True in CPU sections 17–22 for the separately named report.
Review/budget files are namespaced by qualification, so the extension cannot
silently overwrite the primary review selections. The budget ledger is shared
and never reset. No Colab runtime was used to validate this notebook handoff.

## TDD and validation record

- Startup: 8 Ticket 09 tests passed in 43.47s.
- Initial public-boundary test red: `--catalogue` was absent; after wiring the
  shared qualification/evaluation flow, the first end-to-end test passed in 44.62s.
- Extended public test red before conditional family registration existed.
- Added scalar-evidence reconciliation test red before its validator existed;
  it then passed with the independent full-operator write test.
- Notebook selection test red before the flag and namespace wiring, then passed.
- Distinct-mask layout test red before layout deduplication, then passed.
- Qualification status/recovery test red before its public commands existed.
- First affected regression: 58 passed, one failure in 378.48s. The failed race
  test mocked an incomplete configuration and assumed a constant 1,500 total;
  its fixture now supplies 150 item IDs and ten variants. The same assertions
  pass (1 focused test in 0.13s). No acceptance value was changed.
- One intermediate development run rejected changed implementation hashes while
  source files were still being edited. This was expected source-identity refusal,
  not a numerical gate change or a scientific measurement failure.
- Final targeted regression, unique affected-test inventory, retained fixtures
  and notebook release pin are recorded at closeout below.

## Execution status

Implementation and CPU fixtures are distinct from real qualification. No real
model was downloaded or loaded, no GPU job or training ran, no Colab units were
spent, and no Drive upload or GitHub push was performed. Real source preparation,
CPU matrix construction, recorded runtime/budget checks, numerical approval,
production freeze and paired measurements remain pending. Optional supporting
Tickets 11–15 do not block the primary or conditional software paths.

## Closeout, 27 October 2026

- Final targeted suite: **26 passed in 254.78s** (conditional flow/layouts,
  qualification and paired evaluation). Together with the unaffected regressions
  and focused fixture correction, all **62 unique affected tests** passed across
  recorded runs. This is not a full legacy-suite claim.
- Source parses, notebook code parses and contains no saved outputs, and diff
  whitespace checks pass. No acceptance number in the spec changed.
- Canonical labelled synthetic snapshot, raw response/KL/report products, test
  logs, collected-test inventory and structured verification are retained at
  `artifacts/validation/stage0-ticket10/`.
- Real GPU qualification, numerical/budget reviews, execution freeze and
  scientific measurements remain pending. No upload or GitHub push.

- Verified scientific implementation/notebook pin: `d1ba54f674fa0d2034af845b45bd5227f89c0088`.
