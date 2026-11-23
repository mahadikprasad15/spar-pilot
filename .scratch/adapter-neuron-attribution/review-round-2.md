# Adapter and neuron attribution — review round 2

Reviewed draft v2 at `b7a2eef` against all 22 round-1 points, including the revised spec, plan and every ticket. **Recommendation:** the research design is substantially improved and CPU implementation can begin. Make the three small corrections below during implementation and before freezing the affected metrics or running production. They require no additional experiment families or GPU conditions; a further full design-review cycle is unnecessary if these corrections and their tests are made. GPU execution and proposed values remain unapproved.

## Round-1 dispositions

1. **Resolved.** Rehearsals use disjoint calibration questions; prior joint-audit exposure is disclosed and confirmation is labeled partially independent.
2. **Resolved.** CPU construction, rehearsal, qualification and production are separate, with explicit execution gates.
3. **Resolved.** Provenance and the base-correct selection limitation are carried through to the report.
4. **Resolved.** Trace likelihood uses target-aligned logits and the saved generated IDs.
5. **Partly resolved.** The marker-token and empty-mask problems are addressed, but the asserted pilot stop IDs conflict with the actual pilot decoder; see point 24.
6. **Resolved.** Attribution sums include all unpadded causally relevant positions and have a prompt-only fixture.
7. **Partly resolved.** The proposed PEFT branch hook is appropriate, but the frozen-model autograd path needs an explicit requirement; see point 23.
8. **Resolved.** Exclusions, matching, missing controls and the two metric-specific group controls are stated. During the existing discovery pass, retain the per-adapter write summaries and per-`down_proj` signed read summaries needed by matching and c̄; ticket 03 should name these outputs so selection does not need an unplanned rerun.
9. **Partly resolved.** Per-arm diagnostics and failure handling are specified. The 4-of-5 sign rule is a limited check on selected removals; even a pass does not establish the ordering or magnitude of all 196 causal effects. Label it accordingly.
10. **Resolved.** Failure of a gate no longer substitutes nonexistent 196-adapter causal vectors, and neuron concentration remains descriptive.
11. **Partly resolved.** The intended contrasts are clearer, but the equivalence margin is estimated from the same confirmation data; see point 25.
12. **Partly resolved.** Paired correlation differences and a family are specified, but the primary threshold decisions do not yet consistently propagate uncertainty and multiplicity; see point 25.
13. **Resolved.** The natural, unrescaled read and its own full-removal reference are included, with an unavailable rule.
14. **Resolved.** Undefined concentration/cancellation and the random-order reference are addressed.
15. **Resolved.** FP32 is retained, generation batches are qualified, and same-study references are generated. The 20-item diagnostic remains limited evidence, not a guarantee of identical pilot outputs.
16. **Resolved as a design.** The proposed scalar-gradient/recompute route has an identity test and measured-memory qualification; its actual memory and runtime remain to be demonstrated.
17. **Resolved.** This study has explicit units, identities, resume, partial exports and cumulative caps.
18. **Resolved.** Rehearsals name a small covering set and exercise the real export path.
19. **Resolved.** Utilization is treated as an estimated performance diagnostic rather than a validity gate.
20. **Partly resolved.** The current read no longer depends on initialization. For the optional historical description, correct the claimed sign-alignment convention: `scripts/neuron_audit_adapter_init_check.py` computes saved-factor cosine and drift directly, without the stated sign alignment. Report that gauge dependence or omit the claim.
21. **Resolved.** The signed, token-weighted constant is defined and limited to teacher forcing.
22. **Resolved.** Proxy names and behavioral scope are corrected; the six-condition concordance analysis is explicitly exploratory.

## New material points

23. **blocking** — **Specify how gradients exist with frozen parameters.** `spec.md` §Stage A+B; `issues/03-attribution-instrument.md`: v1 explicitly required gradients on the embedding output, but v2 removes that instruction while requiring every parameter frozen and a `g = 1` hook. In a tiny Qwen2/PEFT 0.18 CPU reproduction with all parameters frozen and a constant `1.0 * out` hook, the logits did not require gradients and backward failed; setting `requires_grad_(True)` on the embedding output made backward succeed. The existing wording “in the graph” does not specify a differentiable input or gate. Explicitly create and verify the graph (the embedding-output approach is sufficient), keep model/adapter weights frozen, and ensure the two backwards capture separate metric gradients rather than accumulated `.grad` values. Add a CPU test that runs both backwards and checks each metric independently against finite differences. This is a reproducible risk in a possible literal implementation, not a claim that an implementation already exists.

24. **blocking** — **Use the actual pilot decoder's stop IDs.** `spec.md` §Metrics; `issues/01-trace-preparation.md`: the draft says the pilot stop set contains at least both `<|im_end|>` and `<|endoftext|>`. The pilot evaluation backend passes `tokenizer.eos_token_id` as the generation stop setting (`pilot_eval/backend.py:21–29`), and the pinned-tokenizer test records EOS `<|im_end|>` = 151645 and PAD `<|endoftext|>` = 151643 (`tests/test_neuron_audit_sequences.py:317–321`). A separate Sampling SFT engine uses both IDs, but that is a different generation path. Adding the padding ID to the stop-pull sum when it was not a pilot stop event changes the primary metric. Remove the assumption, seal the exact EOS/stop IDs actually passed to the chosen pilot-compatible decoder, and test the single-ID case. If two IDs are deliberately selected, label and qualify the decoding change.

25. **blocking** — **Carry uncertainty through the primary decisions.** `spec.md` §Statistics; `candidate-plan-v1.yaml` §statistics; `issues/10-report.md`: primary 1 estimates its equivalence margin as 25% of a *confirmation-sample* reference effect, but compares a CI for the other effect to that estimated margin as though the margin were fixed. Primary 3 applies its 0.70/0.30 classification thresholds to point estimates even if their intervals cross those thresholds. These rules can claim separation or sparsity without the intended uncertainty support. Either freeze a meaningful absolute equivalence margin before confirmation or use the existing paired bootstrap to propagate both effects jointly into the relative-margin decision; mark a near-zero reference inconclusive. For primary 3, require the interval for the retained effect to support the claimed threshold, and state how the threshold decisions and pairwise contrasts enter the declared multiplicity family, or label the threshold classifications descriptive. No new GPU measurements are needed.
