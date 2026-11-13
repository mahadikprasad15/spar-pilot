# Adapter and neuron attribution — ticket proposal (first pass)

Status: needs-triage. Not approved.

Graph: 01, 02 → 03, 04 → 05 → 06 → {07, 08} → 09. 09 is fixture-implementable after 04.

| # | Ticket | Blocked by | Where | Delivers |
| --- | --- | --- | --- | --- |
| 01 | Inputs and base-correct traces | — | GPU generation + CPU | Sealed base-correct rows, M_len masks, cohort refs |
| 02 | Proxy metrics | — | CPU | `attribution_metrics.py` |
| 03 | Merged attribution instrument | 02 | CPU | `attribution_instrument.py` (196 adapters + 28 `down_proj` neuron reads, one pass) |
| 04 | Ablation engine | 02 | CPU | `attribution_ablation.py` (removal, 64-neuron read, constant c̄, matched controls, edited checkpoints) |
| 05 | GPU qualification | 01, 03, 04 | GPU | Finite differences, batch invariance, timing, cap |
| 06 | Attribution run + selection freeze | 05 | GPU | Sealed attributions; frozen conditions |
| 07 | Stage C on confirmation | 06 | GPU | 216 teacher-forced ablations |
| 08 | Stage D generation | 06 | GPU | 8 runs: v3 accuracy, length, early stops |
| 09 | Report | 06, 07, 08 | CPU | Gate, three primaries, figures |

Reuse: `neuron_audit_inventory`, `neuron_audit_sequences`, `NeuronCollector` hook conventions, `check_live_adapter`, `BlockStore`, `neuron_audit_export`, `neuron_audit_pod`.
