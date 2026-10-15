# 04: Qualify bounded paired capture and logit-lens mathematics

**What to build:** Run a declared diagnostic capture through the public workflow; validate named sites, primary pooling, paired differences and the logit readout using independent oracles. Save bounded numerical evidence without exposing learned task-token lists before ticket 3.

**Blocked by:** 02: Prepare and freeze the raw corpus.

**Status:** claimed (CPU instrument primitives in progress; bounded GPU cohort pending clarification)

- [ ] Independent tiny-model/full-array answers agree with streamed per-position sums and named-site capture.
- [ ] Both zero adapters pass the reviewed bounded identity checks, with unchanged base weights and finite/reconstruction checks.
- [ ] Retained precision has explicit quantization evidence and no claim that approximate shards are exact.
- [ ] Logit-lens norm gain, positive scaling and zero-vector behaviour match independent toy examples.
- [ ] GPU diagnostic failures save comparison/site/token coordinates, observed values and allowances without requiring another full run to name the failed check.
- [ ] Identity/reconstruction/nonfinite failures stop correctness qualification. They do not become null results or trigger threshold widening.

## Approved Ticket 03 data dependency (A6–A7)

Carry per-document block-14 positions 0–4 through durable capture for whole-document bootstrap. For the optional activation-shaped null, capture base block-14 positions 1–127: FP64 Σh and Σhhᵀ plus count 25,400 (200 × 127), bound to source/runtime/site identities. At width 1,536 the second-moment matrix alone is 18 MiB; profile accumulation and transfer costs. Qualification owns numerical allowances. Missing moments mean this secondary null is unavailable; never fabricate covariance evidence or silently repair a failed matrix.

## Comments · 2026-10-15

Prasad approved implementing diagnostics first and reviewing numerical/FP16 sensitivity evidence before accepting GPU gates. No GPU tolerance was silently adopted. Proposed bounded zero diagnostic is four fixed documents (two legacy, two new); Prasad requested context, so that cohort is not approved yet. No GPU execution launched. Zero-adapter checks alone cannot establish nonzero delta quantization sensitivity; nonzero qualification/profile evidence remains separately declared.

Initial CPU red/green slices committed: explicit paired position/site reductions with per-example primary vectors and base moments, RMS-normalized logit-lens gain, and retention-error diagnostics without a qualification claim. Three independent CPU oracle tests pass. Real model hooks, saved qualification workflow, GPU diagnostics, sensitivity analyses and notebook integration remain to implement. The session-start full regression is running separately; no complete ticket claim is made.
