# Testing

## CPU suite

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

The suite uses synthetic models, arrays and saved workflow fixtures; tests must
not download pretrained weights. Optional integrations use installed packages and
can skip when absent. A passing minimal-dependency run does not establish that
all optional Torch/Transformers paths were exercised.

`pyproject.toml` extras describe installable capabilities. Real Colab notebooks
use separately pinned package environments; installing the broad `gpu` extra is
not a substitute for those pins. Record Python/package versions with validation.

## Validation boundaries

CPU tests cover mathematical oracles, scoring, artifact identities, public CLI
workflows and interruption/resume. They cannot establish real-GPU memory use,
throughput, cross-runtime numerical behavior or successful production execution.
Those require saved target-runtime diagnostics and completion/verifier evidence.

The notebook is a thin launcher for the same public workflow commands. Notebook
release checks validate harness pins and stage ordering. Real-Colab bootstrap
checks include imports, durable storage and fresh-runtime restoration. An accepted
bootstrap is not model/capture qualification.

## Documentation validation

For a documentation-only change, check local links, JSON syntax, cited source
values and the separation between implemented and executed stages. Preserve
scientific thresholds and frozen artifacts. Session validation records live under
`artifacts/runs/diagnostics/`; they are local artifacts, not distributed by Git.

## Recorded documentation-refresh validation

16 October 2026, base commit `d522267`, Python 3.12.14 on macOS:

- Full suite: **394 passed, 35 skipped, 132 warnings**, 1,165.42 seconds.
- Installed numerical/report packages: NumPy 2.5.4, Matplotlib 3.11.2, safetensors 0.8.0; pytest 8.4.2.
- Torch and Transformers were absent; optional integration skips are not real-GPU validation.
- Documentation checks: 105 local Markdown links resolve; 78 inventory file hashes match; selected source values and capture-summary arithmetic agree.
- CLI help exits successfully; `git diff --check` is clean.

The local validation directory is
`artifacts/runs/diagnostics/repository-documentation-2026-10-16/`, containing the
full test log, runtime, documentation checks and validation script. This change
updates documentation only; it does not revalidate live Drive artifacts or alter
scientific execution settings.

## Recorded Stage 0 collection validation

26 October 2026, local Python 3.12 CPU validation:

- Full repository regression: **533 passed, 35 skipped, four warnings**.
- Final Ticket 05 runner checks: **13 passed**, covering collection/resume,
  source/runtime rejection, loaded factors, scoring, budget, ownership and
  concurrent completion/status behavior.
- Skips/warnings include the existing unavailable local Torch shared library.
  These checks do not qualify CUDA inference or establish a real Stage 0 result.
- Structured evidence and explicitly synthetic retained response fixtures:
  `artifacts/validation/stage0-ticket05/verification.json`.

The implementation boundary and real-execution prerequisites are described in
[the Stage 0 guide](stage0-geometry.md). Real Colab qualification and collection
remain pending.

## Recorded Stage 0 fixed-context validation

26 October 2026, Ticket 06:

- Focused fixed-context suite: **16 passed in 20.80s**.
- Affected Stage 0 regression (drift, evaluation, qualification, workflow):
  **49 passed in 55.42s**.
- Session-start full baseline interrupted in legacy activation measurement after
  **253 passed, 24 skipped, four existing warnings in 795.34s**. No failure had
  been reported at interruption. This is not a full-suite pass.
- Independent FP64 KL and dense-write oracles, actual hook/chunk backend exercised
  with a tiny CPU NumPy model, zero controls, corruption, changed runtime/source,
  restart/resume, writer recovery and 20-unit budget rejection are covered.
- Both notebook source pins and code syntax checked. No real CUDA qualification,
  model download, training, Drive upload or scientific measurement occurred.
- Local structured validation and a labelled synthetic fixed-context fixture:
  `artifacts/validation/stage0-ticket06/verification.json`.
