# Adapter and neuron attribution — ticket proposal (draft v3)

Status: tickets 01–05, 10 (code) and 11 ready-for-agent (CPU implementation, user 2026-11-13); 06a–10 needs-triage. No GPU spend, value or budget approved. Implementation tickets authorize no GPU spend; each execution ticket names the authorization it needs.

## Implementation (CPU, fixture-tested)

| # | Ticket | Blocked by | Delivers |
| --- | --- | --- | --- |
| 01 | Trace preparation code | — | Generation/scoring/tokenization path keeping generated IDs; masks; accept/reject records |
| 02 | Proxy metrics | — | `attribution_metrics.py`: trace log-likelihood, stop pull, KL |
| 03 | Attribution instrument | 02 | `attribution_instrument.py`: lora_B gate, pass 1, pass 2 recompute |
| 04 | Ablation engine | 02 | `attribution_ablation.py`: removal, matched controls, 64-term read, constant read, edited checkpoints, generation path |
| 05 | Units, manifest, export | — | Sealed resumable units, partial manifests, export/restore over joint-audit primitives, GPU monitor |
| 10 (code) | Report code | 04 | `attribution_report.py` on fixtures |
| 11 | Overnight chain runner | 05 | Unattended chain from a sealed launch record, caps, summary, volume copy, export, Pod stop, watchdog |

## Execution

| # | Ticket | Blocked by | Authorization | Inputs |
| --- | --- | --- | --- | --- |
| 06a | Colab rehearsal | 01–05 | User go-ahead + Colab cap | Calibration rows, 20 rehearsal generation items |
| 06b | Overnight chain: qualification and preparation | 06a, 11 | Approved launch record | Calibration rows, rehearsal items; then blind traces for discovery and confirmation |
| 07 | Attribution run and freeze | 06b (same chain) | Launch record | Discovery |
| 08 | Stage C | 07 | (same cap) | Confirmation |
| 09 | Stage D | 07 | (same cap) | ADR 0001 cohort |
| 10 | Report (run) | 07–09 (same chain) | Launch record | Sealed outputs |

Graph: {01, 02, 05}; 02 → {03, 04}; 04 → 10 (code); 05 → 11; {01–05, 10, 11} → 06a (attended Colab) → attended Pod rehearsal → overnight chain [06b → 07 → 08 → 09 → 10 → export → Pod stop].

Reuse: `neuron_audit_inventory`, `neuron_audit_sequences`, `NeuronCollector` conventions, `check_live_adapter`, `BlockStore`, `neuron_audit_export`, `neuron_audit_pod`.
