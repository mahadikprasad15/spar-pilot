# SPAR Evaluation and Fine-tuning Pilots

Vocabulary for comparing selective fine-tuning evaluations on fixed benchmark questions.

## Language

**Reference result**:
A previously reported aggregate measurement whose full evaluation procedure may be unavailable.
_Avoid_: Reproduced baseline

**Untuned baseline**:
A measurement of the instruction-tuned model before this project's additional fine-tuning, on a specified evaluation cohort under a specified evaluation procedure.
_Avoid_: Base-model baseline

**Evaluation cohort**:
A fixed set of benchmark questions used across evaluation variants so that results can be compared item by item.
_Avoid_: Random test sample

**Training cohort**:
A fixed set of examples used to train an intervention, kept distinct from its evaluation cohort.
_Avoid_: Dataset, when the specific selected examples matter

**Adapter checkpoint**:
The saved state of an adapter at a specified point in training, evaluated together with its associated untuned model.
_Avoid_: Trained base model, when only the adapter changed

**Optimizer step**:
One update of the trainable parameters, which may combine gradients from several batches of examples.
_Avoid_: Batch, when referring to the number of parameter updates

**Decoder block**:
One transformer layer containing attention, an MLP and residual operations; its output is the representation passed onward through the model.
_Avoid_: Module, when referring to an entire decoder layer

**Adapted linear module**:
A linear transformation inside a decoder block to which a LoRA adapter is attached.
_Avoid_: Block, when referring to an individual projection

**Direct module contribution**:
The output contribution of a module's own adapter on a fixed input, excluding changes to that input caused by upstream interventions.
_Avoid_: Total module-output change

**Block-output change**:
The difference between adapted and untuned activations at a decoder block's output for identical input token sequences, including propagated upstream effects.
_Avoid_: Direct adapter contribution

**Instrument validation**:
Checks that activation measurements observe the intended quantities and satisfy known identities before interpreting the measurements.
_Avoid_: Accuracy evaluation

**Fixed measurement sequence**:
A predetermined token sequence supplied identically to the adapted and untuned model for paired activation measurements.
_Avoid_: Generated response, when describing teacher-forced inputs

**Token-weighted measurement**:
An aggregate in which every counted token receives equal weight, so longer examples contribute more tokens.
_Avoid_: Equal-example average

**Equal-example-weighted measurement**:
An aggregate in which each example's token-level measurements are averaged before giving every example equal weight.
_Avoid_: Token-weighted average
