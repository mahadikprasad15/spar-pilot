# Ticket 03 implementation evidence

Date: 2026-10-26. Software complete; real source/cache/edit execution pending.

## Delivered

- `stage0_edit_math`: principal input/output removal, three saved Gaussian/QR
  output controls, and per-module magnitude controls that preserve the original
  direction. k=0 copies factor bytes exactly. Zero updates are handled explicitly.
- `stage0_edits`: public prepare/run/verify workflow with immutable source/cache
  identity, sealed module units, missing-unit-only resume, errors/status, final
  standard safetensor PEFT adapters and an inspectable JSON/Markdown catalogue.
- Ten variants: V0a, V0b, V1, V2, V3-0/1/2, V4, V4-input and V5. References copy
  source config/tensor files byte-for-byte; source checkpoints are never modified.
- CPU notebook sections 10-12 call the public workflow and do not require the
  optional trained-overlap reveal. No model is instantiated.
- Per-module evidence retains original/edited norms, residual checks and random
  bases. Serialization is checked against constructed factors and an independent
  safetensors implementation in a separate CPU test.

## Red/green and regression evidence

The first public prepare test failed because edit commands were absent. Added
the implementation before rerunning. Later failing tests covered missing notebook
stages and missing original norm metadata before those features were implemented.

Independent matrix calculations check projected dense rank-one updates; factor
norm controls, preserved directions, random-draw differences, zero cases, exact
reference bytes, cache corruption, edited payload corruption, interruption,
conflicting ownership and clean notebook cells are covered. A fresh Python
process verifies the catalogue without importing Torch, Transformers or PEFT.

Final affected/regression command:

`pytest -q tests/test_stage0_edits.py tests/test_stage0_spectral.py tests/test_stage0_workflow.py tests/test_trace_session.py tests/test_notebook_session.py tests/test_behaviour_breakdown.py`

Result: **85 passed in 121.11 seconds**, including **eight edit tests**. The
fresh-process assertion added during the regression run was also run separately:
**1 passed in 19.19 seconds**. Session baseline was 485 passed, 35 skipped;
baseline model-backed skips included the existing local Torch library issue.

## Release and artifacts

Implementation pin: `a69768afd39fcc81b99c5a5fb07f1997d4e416d6`.
Both notebooks select this tested implementation. Static Git-content comparison
verified pinned command/math/workflow files match the tested sources.

Structured verification record:
`artifacts/validation/stage0-ticket03/verification.json`.
The same validation root retains an explicitly synthetic catalogue for inspection.
These are small test adapters, not edits of the real trained model. No GPU,
model download or real production adapter mutation occurred.

## Next

Ticket 04 implements bounded GPU qualification and measured-budget approval.
Do not allocate GPU time yet. Real CPU execution first needs the selected Drive
source audit and verified pinned-weight cache; implementation completion does
not establish that either has run. Publish the release before fetching its pin
in Colab. The cumulative Stage 0 ceiling remains 20 compute units.
