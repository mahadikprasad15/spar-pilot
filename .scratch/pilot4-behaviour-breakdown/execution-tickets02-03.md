# Tickets 02–03: real execution evidence

Executed 2026-10-22, CPU only, against the verified Ticket 01 bundle:
`artifacts/plans/pilot4-behaviour-breakdown-2026-10-22-v1/behaviour.analysis.json`.
No training, model loading, new generation, GPU allocation or source edits.

## Ticket 02: paired greedy transitions

Command: `behaviour-transitions`, report name
`pilot4-behaviour-transitions-2026-10-22-v1`.

Outputs: `artifacts/reports/pilot4-behaviour-transitions-2026-10-22-v1/`.
Status completed, nine report units. `complete.json` binds 24 output files,
including per-question rows, all six scorer/method flow tables, three direct
comparisons, an SVG figure, and actual bootstrap indices with item identities.

Flexible-v3 counts, 150 paired questions per method:

| Method | Gross gains | Gross losses | Churn | Net | Net W flow | Net U flow |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 21 | 24 | 45 | −3 | −9 | +6 |
| GRPO | 18 | 6 | 24 | +12 | +6 | +6 |

GRPO flows: W→C 12, U→C 6, C→W 6, C→U 0. A U→C transition is not by itself
proof of an extraction fix; the human intended-answer audit remains pending.

Paired accuracy differences (10,000 sorted question-ID resamples, local seed 42,
2.5/97.5 percentile bounds):

- SFT64 − SFT0: −2.0 percentage points, interval [−10.7, +6.7].
- GRPO64 − GRPO0: +8.0 points, interval [+2.0, +14.0].
- Direct GRPO64 − SFT64: +10.0 points, interval [+2.0, +18.0].

The bootstrap receipt records the exact item ordering, NumPy generator/version
and retained draws. A fresh sorted-ID resampling realization need not reproduce
every Monte Carlo percentile from an older report using another item ordering.
Original method-specific baselines and cohorts remain retained. All actual
150-question cohorts and declared input/decoding contracts match.

SFT per-question records preserve endpoint lengths, gold calculator-annotation
counts and nonempty gold-solution line counts; structured strata include their
denominators and transition cells. These are descriptive, not causal or validated
difficulty measures.

## Ticket 03: observed sampled reliability

Command: `behaviour-sampling`, explicit frozen source index 2, report name
`pilot4-behaviour-sampling-2026-10-22-v1`.

Outputs: `artifacts/reports/pilot4-behaviour-sampling-2026-10-22-v1/`.
Status completed, four report units. `complete.json` binds 15 output files.
The source is the native saved GRPO evaluation: 150 held-out questions × eight
distinct draws × two endpoints = 2,400 responses. Both endpoints use temperature
1, top-p 1, top-k 0, one beam and cap 1,024. Prompt/gold identities match the
verified greedy cohort. No additional untuned-sampling cross-check was selected
or pooled; no new sampling occurred.

| Flexible-v3 metric | Step 0 | Step 64 | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| Mean draw reward | 54.42% | 67.50% | +13.08 points | [+9.83, +16.50] |
| Observed eight-draw coverage | 90.67% | 92.00% | +1.33 points | [−2.00, +5.33] |

Coverage gains: five questions; losses: three. Among 133 questions observed
solved at both endpoints, the mean correct-draw count increases by 1.12 of eight.
This is compatible with increased reliability on already-observed solves; it
does not uniquely prove sharpening or establish newly acquired ability.

The last-number diagnostic is also reported separately: reward 60.67%→69.08%,
coverage 91.33%→92.67%. It remains extraction-dependent and is not human truth.

Both analyses condition on one trained run per method and saved responses.
Question-level intervals do not capture training-seed or fresh-decoding variation.
The blinded human audit and combined interpretation remain Tickets 04–05.

## Tests and notebook handoff

- Affected regression run: 61 passed in 37.32 seconds. Includes the 32 Ticket 01
  tests, six transition tests, ten sampling/notebook tests and predecessor offline
  scoring/session regressions.
- Independent fixtures check flow arithmetic, exact paired equality, saved draw
  identities, overlap denominators, separate baselines, missing/duplicate draws,
  prompt/policy conflicts, training-cohort rejection and source/output corruption.
- Bounded stops reuse sealed units without overwriting them on resume.
- Notebook sections 6 and 7 each execute and replay in independent subprocesses
  with Torch, Transformers, PEFT, Datasets, Accelerate and Hub imports blocked.
- The public command is the verification/replay boundary: rerunning a completed
  report validates parent receipts and output hashes, and returns saved results.
- The session-wide test command was started before implementation; its final
  outcome is reported separately. No real Colab run or human audit is claimed.

Notebook: `notebooks/pilot-4-behaviour-breakdown.ipynb`; guide:
`docs/pilot4-behaviour-breakdown.md`. It remains an unpublished local release.
