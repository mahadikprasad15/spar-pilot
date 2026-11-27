# Sampling SFT k = 2 readiness check at `b3a4838`

The earlier Gold-cache comparison failure is fixed for the existing cache and for local-versus-streamed model metadata. The focused suite, including the actual 25 MB Gold cache, passed.

## Production seal is still blocked by two proposed decisions

The matched and scaled production drafts still mark K2-D4 and K2-D5 as `[proposed]`. The real draft sealer refused the matched production draft with those two unresolved leaves in a CPU probe. The rehearsal draft records both as user decisions, but the production drafts and design record have not been reconciled. Record the actual confirmed generation engine and storage/export decisions in the production drafts and decision record before sealing. Preserve the scientific protocol and avoid assuming an engine choice from a broader sampler description.

[Matched production draft](launch-record-k2-matched-draft.yaml#L114) · [Scaled production draft](launch-record-k2-scaled-draft.yaml#L111) · [Rehearsal decisions](launch-record-k2-rehearsal-draft.yaml#L113) · [Unresolved-leaf validation](../../pilot_eval/c_attribution_chain.py#L119)

## Interpretation

The revised Gold report now presents the CE/accuracy explanation as a hypothesis in its opening. Section 8 still says the decoded alignment “strongly supports” that explanation, and the takeaway labels a cosine-aligned pair as “genuine cross-layer substitutes.” These remaining statements are stronger than the deferred causal checks support. Bring them into line with the hypothesis framing when updating the report. This does not block training.

[Report source](../../notebooks/_gold_down_proj_report.py#L1302) · [Cross-layer claim](../../notebooks/_gold_down_proj_report.py#L1020) · [Deferred checks](../sparse-rewiring-gsm8k/decision-log.md#L254)

## Verification and remaining execution steps

- **70 focused CPU tests passed**, including comparison with the actual Gold cache. `git diff --check 9258e2c...b3a4838` passed.
- The full test suite stopped during collection because PEFT and PyArrow are missing locally. The focused K2 reference fixture substitutes a reference worker when PEFT is unavailable.
- Held-out K2 target generation, the actual training export and dense adapter restore, and real GPU qualification have not been verified in this review. The inputs must be generated or restored and checked on the target environment before production.
