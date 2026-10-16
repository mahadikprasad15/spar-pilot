# Pilot 3 — fixed-input SFT writes

## Question

Where do SFT adapters contribute locally, and how do their effects propagate
through decoder blocks? How do these measurements vary across checkpoints and
GSM8K question/solution versus unrelated text?

## Design and configuration

| Field | Protocol |
|---|---|
| Model/adapters | Reuse the pinned model and Pilot 2 SFT checkpoints |
| Checkpoints | 0, 8, 16, 32, 64 |
| Inputs | 150 held-out GSM8K sequences with complete gold solutions; 150 frozen FineWeb passages |
| Views | GSM8K question, GSM8K solution, FineWeb content; explicit token masks |
| Block measurement | Adapted minus untuned block output for identical tokens |
| Module measurement | Local `(alpha/r) B A x` on the actual adapted-pass module input |
| Numerical runtime | FP32, eager attention, eval/no-grad, no autocast/TF32; FP64 reductions |
| Aggregation | Token-weighted and equal-example-weighted summaries; whole-example bootstrap |
| Source variants | Original T4 execution and later separately named A10040 remeasurement |

Fixed inputs avoid conflating generated text differences with model-induced
activation differences. All 196 adapted projections are instrumented. The
reference model is shared per compatible batch. Validation checks zero writes,
frozen weights, module mapping, rank-1 reconstruction, numerical agreement and
negative controls.

## Decisions

Batch/runtime qualification, numerical calibration and production recovery have
separate records. Changes use distinct execution identities. The later matched
A10040 SFT measurement did not retrain SFT or regenerate behavioral answers.

For a block, the primary relative-write statistic is
`norm(mean(delta)) / mean(norm(base))`. For a direct module contribution, the
reported primary numerator is `mean(norm(direct_write))`. Per-token ratios and
alternative block denominators are distinct fields; see [methods](../methods.md).

## Results and execution status

Completed scalar SFT write reports are preserved in the local A10040 snapshot.
Their endpoint comparison with GRPO/random is presented in [Pilot 4](pilot-4.md).
The report records matching A10040 runtime/batch membership across arms and
successful measurement gates. This is report-level evidence; the documentation
refresh did not independently hash all original binary shards in Drive.

## Outputs

Prepared token sequences/masks; calibration/profile and reviewed execution
plans; per-example block/module sums and counts; vectors and magnitude
summaries; bootstrap evidence; depth/checkpoint figures and completion inventory.
Full per-token activations were not retained as a universal product; Pilot 5
adds a separate retained-position experiment.

## Limitations

Local adapter writes and propagated block changes are different estimands.
A large layer-output difference does not establish a causal circuit or identify
a useful intervention target. Small mean vectors can reflect cancellation.
Numerical acceptance envelopes are engineering checks, not confidence bounds.

## References

- [Specification](../../.scratch/pilot3-write/spec.md)
- [Instrument guide](../pilot3-instrument.md)
- [Agreement calibration](../adr/0008-pilot3-calibrated-agreement.md)
- [Production recovery](../adr/0009-pilot3-production-integrity-recovery.md)
- [Notebook](../../notebooks/pilot-3-colab.ipynb)
- Implementation: `activation_prepare.py`, `activation_engine.py`, `activation_math.py`, `activation_measurement.py`, `activation_report.py`.
