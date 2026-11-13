# Implementation handoff — tickets 01–05, 10, 11 (CPU)

For the implementing agent. Approved: CPU implementation of tickets 01–05, 10 (report code on fixtures) and 11 (overnight chain runner) (user, 2026-11-13). Not approved: any GPU run, Colab or Pod session, proposed value or budget.

Read first, in order: `AGENTS.md`; `docs/agents/experiment-lessons.md`; `CONTEXT.md`; this folder's `spec.md` (draft v3), `ticket-proposal.md`, `candidate-plan-v1.yaml`; `review-round-1.md`, `review-response.md`, `review-round-2.md`, `review-response-round-2.md`; the joint audit's `spec.md` and `pilot_eval/neuron_audit_*` modules you will reuse.

Rules that bind here:
- Branch from `adapter-attribution-spec`; commit after each green step with short messages.
- Red/green TDD; unit tests on CPU with a tiny random Qwen2 + PEFT 0.18.0, no downloads.
- Frontier order: 02, 05, 01 (unblocked); then 03, 04 (after 02); then 10 (after 04) and 11 (after 05). The end goal is one unattended overnight Pod run that finishes by stopping the Pod (spec §Overnight unattended chain); 11 must make every exit path end in summary → verified volume copy → Pod stop. Claim a ticket by setting `Status: claimed`; resolve with `## Answer` and a pointer in `map.md` Decisions-so-far.
- Proposed values stay `[proposed]`; do not freeze or change thresholds or tolerances. Tolerance fields that are `null` stay null until 06b.
- Any decision not in the spec or an ADR: stop and ask (add a `## Comments` note on the ticket).
- Do not compute anything on confirmation rows or the ADR 0001 cohort.
- Record each serious reusable failure in `docs/agents/experiment-lessons.md`.

## Reuse map — build on these, do not rewrite them

| Need | Existing code | Use |
| --- | --- | --- |
| Sources, adapters, scales, a₀ | `neuron_audit_inventory` (inventory-v2) | Load and verify; no re-inventory |
| Frozen rows, calibration split, leakage record | `neuron_audit_sequences` (sequences-v2) | Calibration, discovery, held-out rows; provenance |
| FP32 / TF32-off settings, live-adapter checks, `down_proj` pre-hook | `neuron_audit_collector` (`set_ieee_precision`, `check_live_adapter`, `NeuronCollector`, `load_saved_factors`) | Same precision and routing checks; z capture convention |
| Gradients on a frozen model through LoRA-branch hooks, reduced in FP64 inside backward hooks, bounded output head | `stage0_push_backend.collect` | Pattern for pass 1: per-module forward hook + `output.register_hook` reduction; head checkpointing over position chunks (16k tokens × 151,936 vocab × 4 B ≈ 10 GB of logits otherwise) |
| Target-aligned positions, full-vocabulary KL | `token_measurement_math` (`prediction_positions`, `full_vocabulary_kl`), `stage0_drift_math.kl_values` | Trace log-likelihood alignment; KL per example |
| Fixed-context engine with LoRA-branch hooks | `stage0_drift_backend.DriftEngine` | Teacher-forced Stage C runner pattern |
| Edited adapters and in-memory edits | `stage0_edit_math`, `offline_grpo_cuts`, `stage0_merged_backend.merged_scope` | Removal edits, edited checkpoints, equivalence checks |
| FP32 greedy batched generation with left padding, determinism, sealed decoding (pad/EOS) | `stage0_evaluation_backend.EvaluationEngine` (`generate`, `verify_loaded_adapter`) | Stage D and base-trace generation; pilot-compatible decoding |
| v3 scoring, ADR 0001 cohort | `scoring.score_gsm8k_flexible_v3`, `protocol.select_gsm8k_indices` | Unchanged |
| Paired intervals, McNemar | `stage0_findings_math` (`paired`, `mcnemar`, `noninferiority`) | Bootstrap and accuracy contrasts |
| Token-budget packing, batch manifests, qualification records | `neuron_audit_qualify` (`pack`, `batch_manifest`, `Qualification`) | Sweeps and frozen manifests |
| Sealed blocks, resume | `neuron_audit_blocks.BlockStore`, `mirror_store` | Units and volume mirror |
| Export, readback, restore, stop guard | `neuron_audit_export` (`snapshot`, `upload`, `restore`, `may_stop_pod`, `publish_final_manifest`) | Ticket 05 |
| Overrun rule, single-writer marker | `neuron_audit_production` (`overrun_check`, `Marker`) | Ticket 11 |
| Pod facts, disk check, rehearsal, chain, stop command | `neuron_audit_pod` (`facts`, `disk_problems`, `rehearsal`, `chain`, `stop_command`) | Ticket 11 |

New code is only what has no precedent: the `lora_B` gate and per-metric `autograd.grad`, pass-2 recompute, stop pull, matched controls, the natural 64-term and constant reads, the report's primary rules, the NVML monitor, the watchdog and the launch record.
