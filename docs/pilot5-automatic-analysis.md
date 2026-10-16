# Pilot 5 Ticket 09: automatic relevance and robustness

## Operating sequence

All stages here use **CPU and saved artifacts**. The existing production GPU job is unaffected; this release does not edit any implementation file bound into production or accepted lens/capture receipts.

1. Finish capture **23**, verify in **24**, then release the GPU.
2. On CPU, complete the reviewed readout workflow **25–26**.
3. **26A** prepares the immutable analysis plan and freezes the seeded null evidence **before 27** publishes learned token lists. Null chunks resume. It does not change the reference sets.
4. Run **27–28** to decode and verify learned readouts.
5. **29** runs/resumes automatic relevance and paired-document analysis.
6. **30** verifies the analysis and displays the report.

Do not run 26A/29 in the GPU runtime merely because they follow capture. Do not stop or restart a running production job to install this release; finish its existing pinned capture/verification first. A fresh CPU notebook can then use the new pinned harness against the existing artifacts.

## Scientific objects and boundaries

```text
Frozen references: questions / gold / GRPO
                │
Qualified norm/head and J14 ── seeded unit-vector nulls ── sealed null chunks
                │                                      before learned publication
Saved capture ── FP64 per-document pooled0–4 vectors
                │       retained per-position differences (disclosed precision)
                ↓
             lens projections → normalized unique positive top20 strings
                ↓
        overlap / word overlap / format overlap / soft-rank
                ↓
    primary family │ direct paired contrasts │ secondary robustness/curves
                ↓
     structured results + Markdown report + saved draws/provenance/costs
```

The primary family is exactly SFT64/GRPO64 × logit/Jacobian, at block14 pooled positions0–4 on raw FineWeb, using `ref_questions` overlap. Holm correction remains across all four tests. An absent lens blocks preparation; the family is not silently reduced. Primary overlap is positive-score descending only. Negative readout lists are not searched for a more favorable primary score.

For each primary score: one-sided empirical isotropic-null p `(1 + count(null >= observed))/(1 + Nnull)`, inclusive empirical percentile, 1,000 whole-document bootstrap draws and a linear-interpolated 95% percentile interval, vector norm and document SNR. Seeds/counts are taken from the exact already-approved Ticket03 procedure bound in the reference freeze. Production has no count-override CLI.

For each lens: direct SFT-minus-GRPO contrast, paired 2,000-document resamples (one ID draw shared across both arms), approved two-sided bootstrap-tail statistic with ties in both tails, 95% interval and a separate two-lens Holm correction. The tail statistic's nominal calibration is not established; no exact-test or calibrated-significance claim is made. One arm meeting a cutoff while the other does not is not an arm contrast. Gold/GRPO-reference contrasts are secondary.

## Secondary evidence

- All three frozen references: overlap, word/format-number overlap and soft-rank. Normalization, ascending-ID tie-breaking, duplicate merging, the first1,000-candidate search and short-list denominators match the approved scorer. Undefined soft rank stays null in JSON.
- Random-adapter delta and pooled base/SFT-adapted/GRPO-adapted activation baselines on the same FineWeb documents. The random adapter is a reference intervention, not a null distribution. Per-position base/adapted baselines are unavailable because production retained pooled base vectors, not those individual base positions.
- Legacy150 and new50 primary-window results, each with a 2,000-draw within-stratum interval and SNR. The new50 overlap is located in the empirical distribution of 1,000 size50 legacy subsets. This is robustness, not an exposure-effect estimate.
- N∈{25,50,100,200}, 200 subsets without replacement per size, saved seed, IDs/counts, overlap, vector norm and SNR. Report median and5–95% range. N=200 is the same full cohort in every draw; repeated values are not additional independent evidence.
- Logit-lens pooled0–4 layer curves at all29 sites with per-document SNR, and descriptive pooled5–127 curves from FP64 cohort sums. No per-document uncertainty is invented from the latter sums.
- Logit max-over-five secondary statistic: every null draw uses five random directions; every bootstrap draw recomputes all five position means before taking the maximum. Five per-position SNR values are retained. This secondary search is not part of the primary family.
- If present in the reference freeze, α0=100/10,000 reference-set overlap sensitivities are computed from the same observed lists. They do not redefine the primary α0=1,000 references.

The scoped JLens receipt only qualifies block14 pooled0–4. Other layers, individual positions, tail windows and max-five JLens analyses remain explicitly unavailable. Intermediate checkpoints are deliberately absent from the accepted endpoint-only production. The activation-shaped null is also unavailable: having raw base moments does not provide the separately reviewed covariance/PSD qualification required by the preflight. No eigenvalue clipping or replacement covariance is invented.

## Null timing and exposure

26A records whether completed readouts or learned readout-unit seals already existed when its null run began. If they did, results are explicitly marked exploratory with respect to null timing. The original human reference-review exposure status is preserved. This is a supported-artifact check, not an assurance about unrecorded manual access.

The isotropic random directions are unit length, independent of learned norms, with seed20261015. The same draw is reused across references/arms for a matching lens/form. The pooled and five-position schedules are separate saved arrays. Both signs are not combined. Invalid/undefined null evidence does not trigger resampling or a smaller declared family.

## Recovery, artifact identities and costs

Plan: `plans/<name>/traces.analysis.json`.

Run: `runs/pilot-5/<name>/automatic-analysis/`:

- `nulls/`: seeded vector arrays, exposure-at-start, sealed score chunks, result and completion inventory.
- `draws.npz`: primary/C3/stratum/matched/N-curve integer schedules. Array indices reference the saved lexicographic document-ID order.
- `document-order.json`: document IDs, strata, retained precision and primary precision.
- `units/`: observed results, paired/resampling chunks, layer curves and max-position units, each hash sealed.
- `results/results.json`: primary, contrasts, robustness, N-curves, prior sensitivities and limitations.
- `results/scores.jsonl`: structured direct/secondary score rows; raw chunk evidence remains under units.
- `results/report.md`: primary/contrast/subgroup/N-curve tables and limitations.
- Standard metadata/status/logs and `checkpoints/progress.json`.

JSON/JSONL/NPZ are canonical rather than adding a Parquet-only dependency. This is an output-format mapping from the original design's proposed Parquet paths; the estimator and draw inventory are unchanged. Nonfinite/undefined numbers serialize as JSON null with context, never as invented zeros.

Projection chunks contain up to16 vectors. Vocabulary columns are projected in4,096-row blocks; the output-score workspace is bounded by the vector chunk. Exact duplicate vectors within a chunk are projected once, retaining every declared draw in the saved scores. Completed chunks are verified/reused; an interrupted unsealed chunk repeats. Schedule files publish by atomic replacement. Parent byte hashes and numerical-library identity bind the plan; changed implementation/inputs require a new named analysis. Writer locks prevent competing writers.

The approved counts request **7,000 null-vector projections** and up to **45,350 analysis-vector projections** before duplicate reuse. This is substantial CPU linear algebra plus vocabulary scoring, not free analysis merely because no transformer is loaded. The plan records these counts; chunks print completion, save elapsed seconds and resume state. End-to-end real CPU/Drive timing has not been measured. No minute estimate is promised.

## Software evidence versus scientific execution

CPU tests use tiny, explicit synthetic weights and source artifacts; they download no model. The public seam covers null timing, interruption/reuse, fresh-process CLI verification, literal overlap/zero contrasts, independent Ticket03 scorer agreement, strict tie ordering, scale invariance, secondary max recomputation, schedule identities and tamper refusal. Synthetic count overrides exist only in the Python test interface and require `execution_kind=synthetic-cpu`.

No production scientific result is claimed by passing software tests. Human grading is Ticket10; final research presentation is Ticket11. Qualification limitations and inferential status travel with the structured results.
