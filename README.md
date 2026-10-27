# SPAR selective fine-tuning pilots

Evaluation and interpretation of rank-1 SFT and GRPO adapters for
`Qwen/Qwen2.5-1.5B-Instruct`. The experiments compare behavior, fixed-input
activation changes, and task-related readouts of those changes.

## Experiments

| Pilot | Question | Documentation | Notebook |
|---|---|---|---|
| 1 | Establish a frozen behavioral baseline and scoring procedure | [Evaluation](docs/experiments/pilot-1.md) | [Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-1-colab.ipynb) |
| 2 | Measure behavior across rank-1 gold-solution SFT | [SFT](docs/experiments/pilot-2.md) | [Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-2-colab.ipynb) |
| 3 | Measure local and propagated SFT writes on fixed sequences | [Internal measurements](docs/experiments/pilot-3.md) | [Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-3-colab.ipynb) |
| 4 | Compare GRPO with SFT: behavior, writes, coefficients and KL | [GRPO comparison](docs/experiments/pilot-4.md) | [Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-4-colab.ipynb) |
| 5 | Test task readability from activation differences on unrelated text | [Readable traces](docs/experiments/pilot-5.md) | [Colab](https://colab.research.google.com/github/mahadikprasad15/spar-pilot/blob/main/notebooks/pilot-5-colab.ipynb) |

Pilot 1 supplies evaluation cohorts. Pilot 2 supplies SFT checkpoints; Pilot 3
measures them internally. Pilot 4 adds GRPO checkpoints and matched measurement
products. Pilot 5 reuses the adapters for a separate raw-text experiment; it
does not require completion of Pilot 4's supplemental KL measurements.

Pilot 4 also has a [CPU behavioural follow-up](docs/pilot4-behaviour-breakdown.md)
for saved-answer transitions, sampling reliability and intended-answer review.
Its [local notebook](notebooks/pilot-4-behaviour-breakdown.ipynb) ends with the
combined findings report; it does not repeat model inference.

[Stage 0 geometry experiments](docs/stage0-geometry.md) reuse the existing
adapters for controlled interventions. The [CPU notebook](notebooks/stage-0-cpu-colab.ipynb)
prepares edits and inspects saved results; the [GPU notebook](notebooks/stage-0-gpu-colab.ipynb)
qualifies the loader and collects paired responses plus fixed-context KL/effective
writes. The CPU notebook also produces paired primary findings and targeted
answer-review snapshots. The primary software path is implemented; real Stage 0
qualification and experimental results are still pending. The optional CPU
rotation/intruder branch is also implemented; it supplies eligibility evidence
for later conditional edits.

See [execution and result status](docs/experiments/status.md),
[measurement conventions](docs/methods.md), and [domain terminology](CONTEXT.md).
The status record distinguishes software validation, saved scientific evidence
and user-reported execution. A resolved implementation ticket is not evidence
that its real-model experiment finished.

## Setup

Python 3.12 is used for local CPU validation; the package declares Python >=3.10.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
python -m pilot_eval --help
```

Tests use controlled model/data boundaries and require no pretrained model
download. Optional integration or plotting tests need their additional
packages; skipped tests must be reported separately. See
[validation record](docs/testing.md).

For real experiments, use the appropriate notebook's environment and pinned
harness. The package's broad optional dependency ranges are not a substitute
for a recorded experiment environment. Do not use **Run all** to resume an
existing experiment; use its saved stage selection and verification commands.

## Artifacts and reproducibility

**Scientific artifacts are not included in the Git clone.** The canonical
artifact tree is `artifacts/`; Colab runs persist it under
`My Drive/SPAR/spar-pilot/artifacts`. Access is currently arranged separately;
there is no public artifact download configured in this repository.

[Artifact access and schemas](docs/artifacts.md) describes configs, responses,
checkpoints, activation shards and completion inventories. Reproducing a report
requires its declared sources, the corresponding code revision and numerical
environment. Adapter weights alone are insufficient for training resume.

The notebooks pin code independently of the branch from which the notebook was
opened. New code must not be substituted silently into an existing frozen run.
CPU reporting and rescoring reuse saved outputs; training, generation and
activation capture are separate operations.

## Repository layout

```text
pilot_eval/       CLI and experiment implementations
notebooks/       Colab stage launchers
scripts/         Profiling and recovery utilities
tests/           CPU and optional integration tests
docs/experiments/ Questions, configurations, results and execution status
docs/adr/        Recorded decisions and amendments
docs/operations/ Execution, environment and recovery references
.scratch/        Original specifications and implementation tickets
output/learning/ Optional HTML explanations
artifacts/       Git-ignored scientific outputs and validation records
```

Start with a notebook's public command in `pilot_eval/cli.py`, then follow the
experiment module. The package includes shared evaluation/scoring,
`sft*`, `activation_*`, `grpo_*`, `token_measurement_*` and `trace_*` workflows.

## Documentation

- [Experiment index](docs/experiments/README.md)
- [Measurement and training conventions](docs/methods.md)
- [Artifacts and access](docs/artifacts.md)
- [Operations index](docs/operations/README.md)
- [Validation](docs/testing.md)
- [Decisions](docs/adr/)
- [Original specifications and tickets](.scratch/)

The experiments use frozen benchmark subsets and one training seed per learned
arm. Cohort bootstrap intervals do not establish cross-seed reproducibility.
Behavior, local writes, propagated changes, conditional KL and lens readability
are distinct measurements; none alone identifies a causal mechanism.
