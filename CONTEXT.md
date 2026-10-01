# Pilot 1 Evaluation

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
