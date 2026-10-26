# Stage 0 preregistration preparation

Status: approved claim rule recorded; complete preregistration not yet frozen.
Date: 2026-10-26. No Stage 0 GPU execution has started.

## Main result

E0.4 adapter-component interventions with matched controls; explanatory gradients,
geometry and dataset measures are supporting evidence and do not block the main
intervention comparisons. KL denotes distributional drift, not demonstrated harm.

## Accuracy preservation

For each declared edited-versus-reference comparison, accuracy is preserved only
if the lower 95% paired-bootstrap interval bound on the accuracy difference is
strictly greater than -0.05. Use 10000 resamples over paired evaluation items,
seed 42, percentile intervals. Publish counts, point estimate and interval.

At n=150, non-inferiority may remain inconclusive even if the point estimate
looks favorable. Failure to establish non-inferiority is not itself evidence of
a loss greater than the margin. Do not revise the margin or sample size after
seeing outcomes to turn an inconclusive result into a success.

The complete preregistration must freeze the comparison family, predictions and
interpretation rules before relevant results are inspected. This partial file
does not establish that the entire study is preregistered.

## Resource limit

Total Stage 0 Colab budget ceiling: 20 compute units across all sessions and
jobs. Stop and request approval before exceeding it. Measure timings and propose
job costs before committing to production workloads; supporting work uses only
remaining approved budget.
