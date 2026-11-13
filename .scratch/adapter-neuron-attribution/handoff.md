# Implementation handoff — tickets 01–05 (CPU)

For the implementing agent. Approved: CPU implementation of tickets 01–05 only (user, 2026-11-13). Not approved: any GPU run, Colab or Pod session, proposed value or budget.

Read first, in order: `AGENTS.md`; `docs/agents/experiment-lessons.md`; `CONTEXT.md`; this folder's `spec.md` (draft v3), `ticket-proposal.md`, `candidate-plan-v1.yaml`; `review-round-1.md`, `review-response.md`, `review-round-2.md`, `review-response-round-2.md`; the joint audit's `spec.md` and `pilot_eval/neuron_audit_*` modules you will reuse.

Rules that bind here:
- Branch from `adapter-attribution-spec`; commit after each green step with short messages.
- Red/green TDD; unit tests on CPU with a tiny random Qwen2 + PEFT 0.18.0, no downloads.
- Frontier order: 02 and 05 and 01 (unblocked), then 03 and 04 (blocked by 02). Claim a ticket by setting `Status: claimed`; resolve with `## Answer` and a pointer in `map.md` Decisions-so-far.
- Proposed values stay `[proposed]`; do not freeze or change thresholds or tolerances. Tolerance fields that are `null` stay null until 06b.
- Any decision not in the spec or an ADR: stop and ask (add a `## Comments` note on the ticket).
- Do not compute anything on confirmation rows or the ADR 0001 cohort.
- Record each serious reusable failure in `docs/agents/experiment-lessons.md`.
