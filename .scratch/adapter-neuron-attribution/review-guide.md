# Adversarial review guide — draft v1 (first pass)

For the external reviewer (Codex). Review `spec.md`, `ticket-proposal.md`, `candidate-plan-v1.yaml` and every ticket in `issues/` before any implementation. All are `needs-triage`; nothing authorizes GPU spend. Read the joint audit spec (`.scratch/joint-neuron-adapter-audit/spec.md`), `AGENTS.md`, `CONTEXT.md` and ADRs 0001 and 0005 first.

Reply in `review-round-1.md` in this folder: numbered points, each tagged **blocking** / **non-blocking** / **question**, with file and section. Claude answers in `review-response.md` with a disposition per point (adopt / qualify / reject + reason), as in the joint audit.

1. **Estimand.** Is attr_i the right first-order estimate of removing adapter i (sign, gate placement on s·B(Ax), reduction over tokens)? Is neuron-to-adapter attribution correctly distinguished from neuron ablation? Are gauge claims right?
2. **Metrics.** M_len definition (mask, EOS id, `####` first token, exclusion window): does it target early stopping without leaking the final answer line? Equal-example weighting. Is M_acc on base-correct vs gold traces the right primary?
3. **Data.** Reuse of sequences-v2 discovery/held-out as discovery/confirmation; base-correct trace generation and its selection effect (only questions the base model solves); generation on the ADR 0001 cohort; any leakage path.
4. **Selection and inference.** Freeze points (top 10, top 5, matched controls, neuron rankings) and whether confirmation can leak in; write-matched controls; validity gates; primary thresholds; bootstrap design; Holm families; anything that lets a result be chosen after the fact.
5. **First-order validity.** Removing a whole adapter is not small. Is a pooled Spearman over 10 singles per arm (60 points per metric) a meaningful gate? Are finite differences enough? What happens to the conclusions if the gate fails?
6. **Implementation.** Hook points against the installed PEFT (branch output vs fused), `retain_grad` on frozen models, memory of an FP32 backward holding z for all 28 `down_proj` layers in one batch, batch invariance, edited-checkpoint equivalence, constant-c̄ generation path.
7. **Tests.** Are the CPU tests sufficient to catch wiring errors (mutation tests)? Missing negative controls?
8. **Ticket graph.** Dependencies, sizes, anything that should be split or merged; what can run before 05.
9. **Scope.** This is a deliberately small first pass (see "Deferred" in the spec). Is anything cut that the three primaries actually need? Is anything left that they don't?
