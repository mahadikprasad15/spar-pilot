# Ticket 02 implementation evidence

Date: 2026-10-26. Software complete; production source/weight execution pending.

## Delivered

- Public `stage0-spectral-prepare/run/verify/report` commands called from the
  CPU notebook's sections 5-9. The first run can be bounded to one module.
- `stage0_weights`: local safetensor snapshot inventory and FP32 parameter hashing
  using recorded PEFT names, including base-layer weights/biases and tied
  embedding deduplication. No model construction or network call.
- `stage0_spectral_math`: float64 reduced SVD, energy ranks, full reconstruction
  and full/stored orthogonality checks, factor projections and fixed Gaussian
  control diagnostics. Original acceptance numbers remain unchanged.
- `stage0_spectral`: hash-bound source/weight plans, separately sealed module
  cache units, missing-unit-only resume, measured time/storage and independent
  verify. Trained projections require an explicit review/reveal step.
- Structured projection rows, zero-factor reasons, initialization/chance/Gaussian
  and available random-adapter references; SVG/PNG figure and report explanation.
- Guide covers optional pinned weight staging, real source identity checks and
  the distinction between cache creation and trained-overlap reveal.

## Red/green evidence

The first public prepare test failed because no spectral commands existed.
Known diagonal matrices verify singular values, k30/k50/k70 and retained width.
Zero spectrum, source mismatch, corrupted basis, interrupted resume and forbidden
reveal cases are covered. A known small diagonal fixture fails the unchanged
Gaussian mean gate; its report stops while its SVD cache remains intact. The
successful reporting fixture independently checks projection answers and
undefined zero factors without changing any production gate or seed.

The actual local safetensor reader is exercised on 196 small synthetic module
matrices plus known bias/embedding/norm parameters. An independently constructed
199-parameter hash verifies the historical PEFT naming/FP32/tied-weight contract.
This is backend correctness evidence, not a real Qwen measurement.

Final affected/regression command:

`pytest -q tests/test_stage0_spectral.py tests/test_stage0_workflow.py tests/test_trace_session.py tests/test_notebook_session.py tests/test_behaviour_breakdown.py`

Result: **77 passed in 118.35 seconds**, including **nine spectral tests**. The
successful fixture figure was visually inspected for readable axes and legend.
No GPU work or model download was performed.

## Release and durable evidence

Implementation pin: `6664bc55df4b8e16136fac29749b609b6cd0d352`.
Both notebook release pins select that implementation. A static Git-content
check verified pinned implementations against the tested files.

Structured validation: `artifacts/validation/stage0-ticket02/verification.json`.
The known-answer projection fixture is retained below the same validation root
and is explicitly synthetic, not an experimental finding.

Ticket 01's scientific implementation files remain unchanged, so adding these
commands does not by itself invalidate its frozen source-audit plans. Spectral
plans bind their own implementation separately.

## Next

Publish before running the pinned Colab checkout. On CPU, run source setup/audit
as necessary, stage the exact local weights, prepare the cache, measure one unit,
resume all and verify. Leave overlap reveal disabled when preserving prospective
gradient-to-overlap predictions. Ticket 03 constructs edits from verified bases;
the optional conditional-intruder Ticket 08 is also software-unblocked.
