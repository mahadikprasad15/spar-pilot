# Stage 0: matched-128 dataset distance (E0.6a) and initial pushes (E0.5)

Decision record: ADR 0016 and the dated amendment in `.scratch/stage0-geometry/spec.md`.
Open both notebooks from the working branch, not `main`. Run section 1 in each new
runtime first, so that `REPO_DIR`, `ARTIFACT_ROOT`, `STAGE0_NAME`, `PLAN` and `run_stage0`
are defined.

The order matters. Do not reveal E0.2 (CPU section 9) or open E0.3 (CPU section 25)
until step 9 has frozen the E0.5 predictions. Revealing earlier writes an exposure
receipt, and the predictions are then labelled exploratory automatically.

## CPU runtime (free)

### 1. Run the torch-dependent tests on Colab CPU

These tests skip in the development container because it has no torch. Colab CPU
runtimes do have it.

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/test_stage0_pushes.py", "tests/test_stage0_matched.py"],
                   cwd=REPO_DIR, capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-2000:])
```

Expect no failures and no skips. If the collector test fails, stop and report it.
Do not start the GPU.

### 2. Find the Pilot 4 untuned eight-draw baseline config

```python
import json
hits = []
for p in (ARTIFACT_ROOT / "plans").rglob("*.json"):
    try: v = json.loads(p.read_text())
    except Exception: continue
    if isinstance(v, dict) and v.get("protocol_version") == "pilot4-sampling-baseline-v1" and v.get("adapter") is None:
        hits.append(p)
print(hits)
BASELINE = hits[0]  # check that exactly one is listed
```

### 3. Build the matched-128 cohort

The audit hashes the pinned weights, so this takes a while on Drive.

```python
AUDIT_RESULT = ARTIFACT_ROOT / json.loads(PLAN.read_text())['run_path'] / 'results' / 'source-audit.json'
revision = json.loads(AUDIT_RESULT.read_text())['sources']['sft']['identity']['model_revision']
WEIGHTS_DIR = ARTIFACT_ROOT / 'inputs' / 'models' / revision
COHORT_NAME = STAGE0_NAME + '-matched128-cohort-v2'
COHORT_PLAN = ARTIFACT_ROOT / 'plans' / COHORT_NAME / 'stage0.cohort.json'
run_stage0('cohort-prepare', '--audit', PLAN, '--baseline', BASELINE, '--weights-dir', WEIGHTS_DIR, '--name', COHORT_NAME)
run_stage0('cohort-verify', '--config', COHORT_PLAN)
```

### 4. Prepare both measurements

```python
SPECTRAL_PLAN = ARTIFACT_ROOT / 'plans' / (STAGE0_NAME + '-spectral-v1') / 'stage0.spectral.json'
DISTANCE_NAME = STAGE0_NAME + '-matched128-distance-v2'
DISTANCE_PLAN = ARTIFACT_ROOT / 'plans' / DISTANCE_NAME / 'stage0.distance.json'
PUSH_NAME = STAGE0_NAME + '-matched128-pushes-v2'
PUSH_PLAN = ARTIFACT_ROOT / 'plans' / PUSH_NAME / 'stage0.pushes.json'
run_stage0('distance-prepare', '--rollouts', COHORT_PLAN, '--name', DISTANCE_NAME)
run_stage0('pushes-prepare', '--spectral', SPECTRAL_PLAN, '--rollouts', COHORT_PLAN, '--name', PUSH_NAME)
run_stage0('pushes-check', '--config', PUSH_PLAN)   # tiny CPU FP64 instrument oracle; needs torch
d = json.loads(DISTANCE_PLAN.read_text()); p = json.loads(PUSH_PLAN.read_text())
print('distance answers:', len(d['unique_answer_ids']), '| push batches:', len(p['unit_keys']), '| A0 cases:', len(p['cases']))
```

Release the CPU runtime. GPU steps follow.

## GPU runtime: one session for both measurements

There is no cumulative Stage 0 cap for these two jobs (ADR 0017). Every reading still goes
in the ledger, and each job is still profiled, forecast and frozen with a wall-time limit.
The `-v1` distance and push plans are superseded by `-v2`; they measured nothing.

Run GPU notebook sections 1–4 first: setup, the dependency check and status. Then define a
helper for usage readings. Every reading is the current cumulative Stage 0 total from
Colab, and it must never be lower than the previous one.

```python
import uuid
from datetime import datetime, timezone
from pathlib import Path
import platform
from pilot_eval.stage0_budget import reservations
SESSION_ID = platform.node() + ':' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def record_usage(units, note):
    obs = ARTIFACT_ROOT / 'notebook' / f'usage-{uuid.uuid4().hex}.json'; obs.parent.mkdir(parents=True, exist_ok=True)
    obs.write_text(json.dumps(dict(event_id=obs.stem, session_id=SESSION_ID, total_units=float(units),
        observed_at=datetime.now(timezone.utc).isoformat(), note=note)))
    run_stage0('budget-record', '--config', PLAN, '--observation', obs)
    for rid, row in reservations(ARTIFACT_ROOT).items():
        if row['state'] == 'active': run_stage0('budget-settle', '--reservation-id', rid)
def approval(name, seconds, notes):
    path = ARTIFACT_ROOT / 'notebook' / f'{name}-{uuid.uuid4().hex}.json'
    path.write_text(json.dumps(dict(reviewer='Prasad', notes=notes, session_id=SESSION_ID, source_reviewed=True,
        max_units_per_hour=1.6, max_seconds=float(seconds))))
    return path
```

Re-define `COHORT_PLAN`, `DISTANCE_PLAN`, `PUSH_PLAN` and `SPECTRAL_PLAN` with the names from step 4.

### 5. E0.6a: profile one answer, forecast, freeze, run

```python
record_usage(CURRENT_UNITS, 'Before matched-128 distance profile.')  # set CURRENT_UNITS to the actual reading first
run_stage0('distance-run', '--config', DISTANCE_PLAN, '--approval', approval('distance-profile', 900, 'One-answer profile.'), '--max-new-units', 1)
run_stage0('distance-forecast', '--config', DISTANCE_PLAN)
DISTANCE_DIR = ARTIFACT_ROOT / json.loads(DISTANCE_PLAN.read_text())['run_path']
forecast = json.loads((DISTANCE_DIR / 'results/forecast.json').read_text()); print(forecast['conservative_total_seconds'])
```

Record usage again, then freeze with roughly three times the forecast and run:

```python
from pilot_eval.training import file_hash
record_usage(CURRENT_UNITS, 'After distance profile.')  # update CURRENT_UNITS first
seconds = 3 * forecast['conservative_total_seconds'] + 600
review = ARTIFACT_ROOT / 'notebook' / 'distance-production-review.json'
review.write_text(json.dumps(dict(reviewer='Prasad', notes='Matched-128 scoring production.', session_id=SESSION_ID,
    source_reviewed=True, max_units_per_hour=1.6, max_seconds=seconds,
    forecast_sha256=file_hash(DISTANCE_DIR / 'results/forecast.json'))))
run_stage0('distance-freeze', '--config', DISTANCE_PLAN, '--review', review)
run_stage0('distance-run', '--config', DISTANCE_PLAN)
run_stage0('distance-verify-measurement', '--config', DISTANCE_PLAN)
```

### 6. E0.5: profile one batch per objective, forecast, freeze, run

```python
record_usage(CURRENT_UNITS, 'After distance production.')  # update CURRENT_UNITS first
run_stage0('pushes-run', '--config', PUSH_PLAN, '--approval', approval('push-profile', 1800, 'One batch per A0/objective.'))
run_stage0('pushes-forecast', '--config', PUSH_PLAN)
PUSH_DIR = ARTIFACT_ROOT / json.loads(PUSH_PLAN.read_text())['run_path']
pf = json.loads((PUSH_DIR / 'results/forecast.json').read_text()); print(pf['conservative_total_seconds'])
```

Look at one profile unit's bound evidence before freezing:

```python
meta = json.loads(next((PUSH_DIR / 'units').glob('*/metadata.json')).read_text())
worst = max(meta['proof']['reconstruction_allowances'].items(), key=lambda kv: kv[1]['absolute_error'] / kv[1]['allowed'])
print(worst)
```

Then:

```python
record_usage(CURRENT_UNITS, 'After push profile.')  # update CURRENT_UNITS first
seconds = 3 * pf['conservative_total_seconds'] + 600
review = ARTIFACT_ROOT / 'notebook' / 'push-production-review.json'
review.write_text(json.dumps(dict(reviewer='Prasad', notes='Matched-128 gradient production.', session_id=SESSION_ID,
    source_reviewed=True, max_units_per_hour=1.6, max_seconds=seconds,
    forecast_sha256=file_hash(PUSH_DIR / 'results/forecast.json'))))
run_stage0('pushes-freeze', '--config', PUSH_PLAN, '--review', review)
run_stage0('pushes-run', '--config', PUSH_PLAN)
run_stage0('pushes-verify-measurement', '--config', PUSH_PLAN)
record_usage(CURRENT_UNITS, 'After push production.')  # update CURRENT_UNITS first; then release the GPU
```

If any run stops, paste the printed error. Do not re-run first.

## CPU runtime again (free)

### 7. E0.6a report

```python
run_stage0('distance-report', '--config', DISTANCE_PLAN)
run_stage0('distance-verify', '--config', DISTANCE_PLAN)
```

### 8. E0.5 summary (endpoints stay hidden)

```python
run_stage0('pushes-report', '--config', PUSH_PLAN)
```

### 9. Freeze E0.5 predictions, then reveal

Write your predictions after reading the step 8 summary and before any trained-geometry
view.

```python
review = ARTIFACT_ROOT / 'notebook' / 'push-prediction-review.json'
review.write_text(json.dumps(dict(reviewer='Prasad', notes='Written after the push summary, before E0.2/E0.3.',
    predictions='<your predicted SFT vs GRPO pi_out ordering and cosine expectations>', prior_exposure=False)))
run_stage0('pushes-predict', '--config', PUSH_PLAN, '--review', review)
run_stage0('pushes-reveal', '--config', PUSH_PLAN)
run_stage0('pushes-verify', '--config', PUSH_PLAN)
```

Then reveal E0.2 (CPU section 9) and view E0.3 (CPU section 25).
