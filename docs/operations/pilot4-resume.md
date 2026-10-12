# Pilot 4 notebook operations

The notebook is a CLI client. Scientific validation, immutable configs and
completion seals remain owned by the existing experiment commands.

## Reconnect

1. Open `notebooks/pilot-4-colab.ipynb`; use its pinned checkout.
2. Run setup (1). It restores selected names and artifact paths from Drive,
   then displays saved progress without treating counters as verified completion.
3. Run environment setup (4). Enable `GPU_WORK` for GPU stages. This installs
   the pins and tests actual Qwen/PEFT/TRL imports in a fresh process.
4. Inspect readiness and continue at the next pending stage. Do not Run all.

To deliberately select a new numerical execution, turn off
`USE_SAVED_SELECTION` for that setup run and choose a new `WRITE_VARIANT`.
For the current A100 40GB continuation, `a10040-v2` avoids older plan names.
Selected source/run names are stored in
`artifacts/notebook/operations/<plan>/selection.json`. These are mutable UI
preferences, not scientific configuration or approval records.

## Boundaries

- A CPU runtime cannot launch GPU work. A second active experiment child is
  rejected before another GPU stage starts.
- Completed measurements, calibration phases and profiles are routed through
  their existing CPU verification functions. A completion marker is a candidate
  for verification, not permission to trust corrupt evidence.
- Hardware/dependency changes are reported before model loading. They are not
  silently allowed. Incomplete calibration/profile execution retains its pinned
  implementation; use an explicit new plan if it changes.
- Code-only measurement recovery retains the existing reviewed policy. It does
  not authorize different hardware or numerical settings.
- SFT remeasurement uses the original Pilot 3 prepared inputs and trained
  adapters. When selected, it follows the same calibration/profile/measurement
  stages as GRPO and the random reference. No training or generation is added.

## Process reporting

`NotebookSession.run` streams combined stdout/stderr, prints a child PID and
an elapsed-time heartbeat during silence, and appends a console log under the
canonical operation folder. A heartbeat means the child remains active, not
that scientific work has advanced. Actual nonzero exits include the error tail.
Interrupting the notebook runner terminates and waits for its child; recovery
continues from scientific seals, not from the PID or an old status counter.

Local tests cover fresh-kernel restoration, source-contract fixtures, missing
GPU, hardware mismatches, active-child protection, readable failures, ambiguous
profiles, CPU reuse with corruption rejection and saved selection restoration.
A real fresh Colab GPU session is still a separate validation environment.
