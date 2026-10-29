# Sampling SFT: descriptive data gate

Ticket 07 has two execution phases: collect independent HF FP32 scores through
an approved budgeted model-work job, then calculate statistics and figures on
CPU. It compares data recipes; it does not isolate a causal effect of being
on-policy. No training starts from a report command.

## 1. Prepare the score/report plan on CPU

Finish and freeze the [reviewed coverage](sampling-sft-review.md) first. Save an
explicit word-policy JSON, recording the approved normalization:

```json
{
  "version": "unicode-words-v1",
  "approved": true,
  "reviewer": "REVIEWER",
  "notes": "Unicode NFKC, case-folding, normalized apostrophes; occurrence-weighted response4-grams. Approved 2026-10-29."
}
```

```sh
python -m pilot_eval sampling-sft-gate-prepare --review plans/sampling-review-v1/sampling-sft.review.json --word-policy artifacts/plans/sampling-review-v1/word-policy.json --name sampling-gate-v1 --output-root artifacts
python -m pilot_eval sampling-sft-gate-status --config plans/sampling-gate-v1/sampling-sft.gate.json --output-root artifacts
```

Config paths are relative to the artifact root or absolute. Policy, forecast and
approval-file paths are relative to the checkout or absolute. The plan freezes
the reviewed selection and original token IDs. Gold uses exactly the original
SFT loss-bearing target tokens, including its supervised end-turn. Each sampled
arm uses its own reviewed eligible responses. Nothing is decoded/re-encoded to
replace scoring targets.

There are512 score units, one per training problem. Each unit contains gold and
the eligible k-arm responses available for that problem, in fixed method order.
Thus the maximum is2,048 answer scores; omissions reduce it. All scored targets
have an actual final stop token. Missing arms are not silently filled.

## 2. Collect HF scores through a reserved job

Use the [job/export/stop lifecycle](sampling-sft-runpod.md). Its child argv is:

```json
["python", "-m", "pilot_eval", "sampling-sft-gate-score",
 "--config", "plans/sampling-gate-v1/sampling-sft.gate.json",
 "--job", "runs/sampling-sft/sampling-sft-v1/jobs/gate-score/job.plan.json",
 "--output-root", "artifacts"]
```

This is model work, not a CPU report step. Reserve it against the sampling
sub-cap before launch, count setup/debugging/idle time, and obtain the explicit
paid-job approval. The real route requires the qualified pilot/runtime/T3
handoff and real collected arms; CPU fixtures cannot satisfy it. It verifies
the exact launched job before importing a model.

The existing independent HF oracle loads the frozen local base snapshot in
FP32 with TF32 disabled and eager attention. It uses the original scorer prompt,
full-vocabulary log-softmax and the causal prediction shift. No temperature,
repetition penalty or gold-guided proposer context enters the score. The
collector does not reuse the sampler's cached acceptance score as HF evidence.

Each answer retains aligned IDs, prediction positions and per-token raw log
probabilities, including the actual stop. A malformed request replays once;
another failure stops with diagnostics. It never becomes a successful score.
Context overflow blocks instead of truncating.

Completed problem units are hash-sealed with worker/runtime evidence. Resume
skips them; partial unsealed work is preserved as abandoned history and redone.
If only completion remains, the collector finalizes without loading the model.
`--max-new-units` bounds a diagnostic invocation. It does not certify the rest of
the score matrix. Actual execution costs remain in the shared budget ledger.

One writer owns scoring. After inspecting a genuinely dead worker, release its
recorded token through `sampling-sft-gate-recover-writer --config ...
--owner-token ... --review-notes ... --output-root ...`. An active local PID or
wrong token is rejected. Never delete the writer marker to bypass this check.

## 3. Produce and verify the CPU report

```sh
python -m pilot_eval sampling-sft-gate-report --config plans/sampling-gate-v1/sampling-sft.gate.json --output-root artifacts
python -m pilot_eval sampling-sft-gate-verify --config plans/sampling-gate-v1/sampling-sft.gate.json --output-root artifacts
```

The CPU commands load no language model. They require all512 score units and
reconcile the report with reviewed coverage and raw HF scores.

| View | Denominator and interpretation |
|---|---|
| Own covered cohorts | Gold512 and each arm's own eligible problems; descriptive selection differences remain visible |
| Three-arm common cohort | Same problems across gold/k0/k2/k5; paired contrasts use only this set |
| NLL | Negative mean log probability per answer, stop included; then equal-problem mean |
| Length | Retained response-token count, stop included; median and inclusive-linear p90 |
| Format | Strict final-line validity and literal marker presence, with explicit answer denominators |
| Initial pass/unfinished | Automatic initial-attempt outcomes, denominator512 per k arm |
| Attempts/retries | All retained attempts and distinct retried problems; separate from reviewed coverage |
| Acceptance | Strict improvement decisions, overall and by refinement step/cut-position decile |
| Calculator marker | Literal `<<` occurrence rate; not proof of copying |
| Word4-gram overlap | Fraction of response4-gram occurrences also appearing anywhere in cleaned gold |

Word normalization is Unicode NFKC, case-folding, curly-to-straight apostrophes
and Unicode letter/digit words with internal apostrophes retained. Calculator
markup and the final marker are cleaned by the frozen proposer gold-cleaning
rule before constructing gold4-grams. Repeated response occurrences count
separately. Responses under4 words are undefined, not zero; the report gives
defined and undefined counts. This is lexical overlap, not semantic leakage or
a memorization test.

Paired comparisons resample whole problem IDs10,000 times with seed42, using
NumPy's PCG64 generator. Each resample uses the same indices for both arms.
Length is the difference of medians recomputed within that resample, not the
median of per-problem differences. NLL is the difference of equal-problem means.
Intervals are percentile95% with linear quantiles. An empty common cohort is
explicitly undefined. A confidence interval containing zero does not establish
equivalence or trigger an automatic arm merge.

The temperature1 baseline and typical-NLL offset are absent. Relative sharpening
is marked not measured. Later Stage0 measurements must match these definitions
or describe their differences explicitly.

## 4. Record the training selection, separately from execution

After inspecting the report, supply a measured training forecast:

```json
{
  "arms": [0, 5],
  "per_arm": {
    "0": {"rate_usd_per_hour": "ACTUAL_QUOTE", "estimated_hours": "MEASURED_ESTIMATE", "setup_hours": "SETUP_ESTIMATE"},
    "5": {"rate_usd_per_hour": "ACTUAL_QUOTE", "estimated_hours": "MEASURED_ESTIMATE", "setup_hours": "SETUP_ESTIMATE"}
  },
  "notes": "Measured training basis and reason for selecting these arms."
}
```

```sh
python -m pilot_eval sampling-sft-gate-request --config plans/sampling-gate-v1/sampling-sft.gate.json --forecast artifacts/plans/sampling-gate-v1/training-forecast.json --output-root artifacts
```

The request binds report, reviewed coverage, HF score evidence, proposed arms
and the current shared budget. The default scientific scope is k0/k5, one
training seed42. Choosing only one requires the explicit recorded reason; k2 is
not selected here and requires separate approval. No automatic meaningful-
difference rule, merge or causal attribution is applied.

Cost uses exact microdollars: quoted rate × (estimated hours ×1.5 + setup time),
summed across selected arms. It must fit the $6 training and $25 total ceilings.
Current or projected80% commitments require explicit review. This request does
not reserve funds or launch a job.

An approval JSON contains exactly `request_path`, `request_sha256`,
`report_sha256`, `budget_sha256`, `reviewer`, `notes`, `gate_approved: true`,
`acknowledged_coverage_flags` and boolean `review_80_percent`. Copy identities
and required flag strings from the request. A changed report, request or ledger
requires a fresh bound decision.

```sh
python -m pilot_eval sampling-sft-gate-approve --config plans/sampling-gate-v1/sampling-sft.gate.json --approval /absolute/path/training-selection.json --output-root artifacts
```

More than5 uncovered problems blocks the selection. Marked omissions and
uncovered-set difference flags require explicit acknowledgment. Recording a gate
selection still leaves `training_authorized: false`: Ticket08 must verify the
exact training bundle and T15/T16 readiness, and Ticket09 controls paid training.
Synthetic fixtures remain labelled as such and cannot substitute for real
qualification or real execution evidence.

## Durable outputs

```text
artifacts/runs/sampling-sft/<source-plan>/gate/<name>/
  meta/run_manifest.json
  scores/env.<attempt>.json
  scores/events.jsonl
  scores/units/000/scores.json       # through511
  scores/units/000/complete.json
  scores/results.json
  scores/complete.json
  results/answers.jsonl
  results/report.json
  results/summary.csv
  plots/common-nll.svg
  plots/common-length.svg
  plots/paired-nll.svg
  plots/paired-length.svg
  plots/manifest.json
  report.html
  complete.json
  training-requests/<hash>.json
  training-selections/<hash>.json
```

Retain raw JSON/JSONL and score units, not just the HTML. Figure manifests state
their metric, cohort, problem count and numeric inputs. Repeating an unchanged
report verifies and reuses it. Altered score, source, coverage or report files
block reuse instead of silently changing the data or acceptance criteria.
