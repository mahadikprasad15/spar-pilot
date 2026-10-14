# 04: Qualify bounded paired capture and logit-lens mathematics

**What to build:** Run a declared diagnostic capture through the public workflow; validate named sites, primary pooling, paired differences and the logit readout using independent oracles. Save bounded numerical evidence without exposing learned task-token lists before ticket 3.

**Blocked by:** 02: Prepare and freeze the raw corpus.

**Status:** ready-for-agent

- [ ] Independent tiny-model/full-array answers agree with streamed per-position sums and named-site capture.
- [ ] Both zero adapters pass the reviewed bounded identity checks, with unchanged base weights and finite/reconstruction checks.
- [ ] Retained precision has explicit quantization evidence and no claim that approximate shards are exact.
- [ ] Logit-lens norm gain, positive scaling and zero-vector behaviour match independent toy examples.
- [ ] GPU diagnostic failures save comparison/site/token coordinates, observed values and allowances without requiring another full run to name the failed check.
- [ ] Identity/reconstruction/nonfinite failures stop correctness qualification. They do not become null results or trigger threshold widening.
