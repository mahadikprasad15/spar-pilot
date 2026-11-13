# Adapter and neuron attribution — ticket proposal

Status: needs-triage. Not approved.

Graph: 01 → 05; 02 → 03 → 05; 02 → 04 → 05; 05 → 06 → {07, 08} → 09. 09 is fixture-implementable after 04.

| # | Ticket | Blocked by | CPU / GPU | Delivers |
| --- | --- | --- | --- | --- |
| 01 | Freeze inputs and base-correct traces | — | GPU (generation) + CPU | Sealed base-correct set, M_len masks, cohort refs |
| 02 | Per-example metrics | — | CPU | `attribution_metrics.py` |
| 03 | Attribution instrument | 02 | CPU | `attribution_instrument.py` (Stage A + B) |
| 04 | Ablation engine | 02 | CPU | `attribution_ablation.py`, condition builder |
| 05 | GPU qualification | 01, 03, 04 | GPU | Finite-difference, batch invariance, timing, cap |
| 06 | Stage A/B production + selection freeze | 05 | GPU | Sealed attributions, frozen conditions |
| 07 | Stage C on confirmation | 06 | GPU | Teacher-forced ablation effects |
| 08 | Stage D generation | 06 | GPU | v3 accuracy, length, early stops |
| 09 | Report | 06, 07, 08 | CPU | Gates, primaries, secondaries, figures |

Reuse, not rewrite: `neuron_audit_inventory` (sources), `neuron_audit_sequences` (frozen rows), `NeuronCollector` hook conventions, `check_live_adapter`, `BlockStore`, `neuron_audit_export`, `neuron_audit_pod`.
