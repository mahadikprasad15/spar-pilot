# Pilot 5: capture acceptance from saved evidence

## Run sequence

After the completed JLens acceptance and production proposal, use CPU setup **1–2 → 21A**. Stop and share the result. After reviewing a separately saved JSON, **21B** writes a sealed capture qualification receipt. **22** then consumes that receipt through the separate production review; scope, budgets and predictions are not selected here. No GPU is required by 21A/21B. Existing 14–20C and 21 need no rerun.

This closes a missing producer/consumer handoff: the production freeze already required a capture qualification receipt, but the bounded diagnostics did not produce that receipt. Profile completion alone still does not mean acceptance.

## Evidence and work

```text
Saved zero-adapter diagnostic ───── exact identity + unchanged base + GPU norm/head oracle
Saved four-document endpoint profile ─ batching and storage errors, all 29 sites/128 positions
Pinned saved norm/head/token inventory ─ independent logit arithmetic + primary ranking sensitivity
                         ↓
              21A: five resumable CPU units
                         ↓
              unapproved review.draft.json
                         ↓ explicit numerical limits and bounded-scope acceptance
              21B: passing OR failed sealed receipt
                         ↓
              22: separately reviewed production freeze
```

The five units are capture metrics, an independent logit arithmetic oracle, then SFT-64, GRPO-64 and random-64 logit retention sensitivity. The latter three use the chosen batch and storage precision, not a new batch search. They reuse the approved 1,000 paired whole-document resamples with seed 42, deduplicating repeated multisets. The first cohort mean and all resamples are compared before/after storage conversion, over the full vocabulary and both ranking signs. No learned token IDs/strings are published and no transformer is constructed.

Batching reports the worst absolute activation difference between the chosen arrangement and saved singleton outputs, with coordinates, across all four conditions and sites. It is **not** a claim of identical batch rankings or batch-independent numerical results. Retention reports the worst adapted-minus-base conversion error at all 29 sites and 128 content positions. Readout sensitivity is restricted to block 14 pooled positions 0–4. Intermediate checkpoints and full-corpus effects are not tested by these four documents.

The standalone JLens receipt remains independent: 21A measures logit-lens sensitivity; 20B/20C measured and reviewed the Jacobian-lens path. Neither substitutes for the other. No acceptance thresholds are inherited automatically, including the separately approved JLens thresholds.

## Review fields

Copy `review.draft.json` to a separately named file; preserve the original. Supply:

- `approved`, named `reviewer`, `notes`, `bounded_scope_accepted`.
- The unchanged `diagnostic_sha256`.
- Seven explicit limits: `batch_max_absolute_difference`, `retention_max_absolute_error`, `numeric_lens_max_score_error`, `cohort_max_score_error`, `bootstrap_max_score_error`, `cohort_min_top20_overlap`, `bootstrap_min_top20_overlap`.

The numeric lens gate uses the maximum of the new CPU logit oracle error and the original actual-GPU final-norm/head oracle error for real runs. Therefore the JLens transport arithmetic error near 1e-14 is not the sole numeric measurement. Undefined zero vectors are checked explicitly; undefined nonzero-cohort/resample readouts fail acceptance. Limits must be finite and explicit; zero/base integrity remain exact mandatory checks. Limits are never widened after failure.

A successful receipt uses protocol `pilot5-capture-qualification-v1`, matching the existing production consumer. It binds the exact profile, model revision, execution kind, selected batch and retained precision; named checks bind zero identity, base integrity, numeric lens, batching, retention, ranking and bootstrap. All evidence and the reviewed file are hash-bound. A failed decision is preserved at its own reviewed hash, raises an error and cannot be used as a qualified receipt.

## Artifacts and recovery

- Plan: `plans/<name>/traces.capture-qualification.json`.
- Evidence: `runs/pilot-5/<name>/capture-qualification/`.
- Sealed units: `units/<unit>/result.json` and `complete.json`.
- Saved resampling schedule: `bootstrap-indices.npy`.
- Summary: `results/results.json`; metadata/status/logs follow the standard tree.
- Receipt: `reviews/<review-hash>/qualification.json` plus `.seal.json`.

Prepare, diagnose, verify and approve are public commands under `python -m pilot_eval traces-capture-qualification-<operation> --help`. Diagnosis supports `--max-new-units` for a bounded pause. Completed units are verified/reused; interrupted unsealed units repeat. Changed sources, implementation, payloads or draw schedules are refused under the same identity. Changing reviewed limits creates a separate reviewed decision, not a replacement diagnostic. No existing source/profile/lens artifact is rewritten.

CPU/Drive timing for the real workload has not been measured. No minute estimate is promised. Local CPU fixtures validate implementation, CLI reconnection, resume/corruption handling and actual production-freeze receipt consumption; they do not qualify the user's real artifacts. The user must run 21A and review its evidence.
