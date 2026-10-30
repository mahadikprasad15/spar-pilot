# Stage 0: matched-128 dataset distance and initial pushes

Status: approved by the user on 2026-10-29, before any E0.6a/E0.5 GPU run.

R0 would generate 384 × 8 missing untuned answers, estimated at 8–15 Colab units
against 3 remaining. E0.6a (dataset distance) and E0.5 (initial push) instead use the
128 questions whose original Pilot 4 untuned eight-draw groups exist.

Decision: see `spec.md`, "Matched-128 E0.6a/E0.5 amendment". In summary:

- A new CPU product, `stage0-matched-cohort-v1`, packages the 128 verified historical
  groups. It applies the existing cohort rule and gold restricted to the same questions,
  in the layout E0.6a and E0.5 already consume. A shared loader dispatches on the plan
  protocol, so the 512 R0 path is unchanged.
- Scoring and gradient sessions freeze at their first GPU attempt. Only the code commit
  may differ later, and it is recorded per unit.
- The E0.5 gradient reconstruction gate is the larger of the legacy relative 1e-5 and
  an a-priori FP32 accumulation bound. It is pre-registered here, before any real-GPU
  observation, because GRPO gradients sum advantage-weighted terms of both signs. This
  cancellation pattern broke result-relative gates in ADR 0015.
- Consistency is reported with its 1/√N pure-noise reference.
- NumPy versions are recorded rather than required to match. Recomputed floats agree to
  relative 1e-12.

Limits: 128 questions, so intervals are wider than 512 would give. Historical samples
were drawn on a different GPU from the one that scores them. E0.5 has about 16 GRPO
batches and fewer SFT batches per a₀. The E0.5-to-geometry link stays confirmatory only
while E0.2/E0.3 remain unviewed until the E0.5 predictions are frozen.

## Review follow-up (29–30 October 2026, before any GPU run)

An independent review of `5c72c88` found five gaps. All were fixed before use:

- The notebook `run_stage0` now stops and reaps its worker on interrupt: SIGINT, then
  terminate, then kill. A cancelled cell can no longer leave GPU work running or a writer
  lock held.
- E0.6a, E0.5 and the matched cohort fingerprint their whole transitive `pilot_eval`
  import chain, including the model loader. The loaded model's attention implementation
  is checked to be eager, rather than trusted from the declared runtime. A commit-only
  difference can therefore no longer hide a numerical code change.
- The matched cohort reload compares floats to relative 1e-12. IDs, order, memberships
  and counts stay exact.
- Cohort preparation rejects historical sampling whose settings differ from temperature 1,
  top-k 0, top-p 1, cap 1024 and one question per group.
- Reports derive question counts from the cohort and carry
  `cohort: matched-128-historical`. E0.6a limitations state the gold-versus-own selection
  effect: own_correct covers only questions with a correct sample.

The files fingerprinted by the completed behavior, internals and findings products are
unchanged.

Second review round (30 October 2026): cohort preparation also requires the baseline's
separate `decoding` block to match its `settings`: sampling on, one beam, no repetition
penalty, and the same temperature, top-p, top-k and cap. Contradictory provenance is
rejected. Every standalone export (CSV, Parquet, JSONL, endpoint JSON and HTML) carries
the cohort label and question counts.
