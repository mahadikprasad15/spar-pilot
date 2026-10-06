# Pilot 3 calibrated-agreement recovery

Use this in your current connected GPU session. No install, preparation, training,
or answer-generation rerun is required. Stop any live previous profile child first.
Keep the same GPU and numerical settings through calibration, validation and profiling.

## 1. Update code and restore helpers

Set `RECOVERY_COMMIT` to the full commit supplied in the accompanying response.
Paste the following in a new cell. It preserves prepared inputs and v1 evidence.

```python
assert not live_children(), 'A previous child is still running; inspect it first.'
RECOVERY_COMMIT = 'REPLACE_WITH_SUPPLIED_COMMIT'
dirty = subprocess.check_output(['git','status','--porcelain'],cwd=REPO_DIR,text=True).strip()
assert not dirty, 'Preserve local checkout edits before switching code.'
subprocess.run(['git','fetch','origin',RECOVERY_COMMIT],cwd=REPO_DIR,check=True)
subprocess.run(['git','checkout','--detach',RECOVERY_COMMIT],cwd=REPO_DIR,check=True)
HARNESS_COMMIT = RECOVERY_COMMIT
for module_name in list(sys.modules):
    if module_name == 'pilot_eval' or module_name.startswith('pilot_eval.'):
        del sys.modules[module_name]
RUN_NAME = 'pilot3-write-v2'
REPORT_NAME = 'pilot3-write-report-v2'
EXECUTION_PATH = ARTIFACT_ROOT / 'plans' / RUN_NAME / 'activation.execution.json'
REPORT_DIR = ARTIFACT_ROOT / 'reports' / REPORT_NAME
SESSION_DIR = ARTIFACT_ROOT / 'notebook' / RUN_NAME
SESSION_DIR.mkdir(parents=True,exist_ok=True)
nb = json.loads((REPO_DIR / 'notebooks/pilot-3-colab.ipynb').read_text())
UPDATED_STAGES = {c['metadata']['stage']: ''.join(c['source'])
                  for c in nb['cells'] if c['cell_type']=='code'}
exec(UPDATED_STAGES['helpers'],globals())
PREPARED, ROWS = restore_preparation()
print('Restored prepared inputs:',len(ROWS))
```

## 2. Collect calibration

Separate cell; GPU required. Existing reviewed input variables are reused.

```python
exec(UPDATED_STAGES['calibrate'], globals())
```

There are seven variants, each across all five adapters, on 16 calibration examples.
This has a real compute cost; it is not a quick single forward pass. Progress shows
variant, example offset and checkpoint. Repeating the cell verifies completed
units and computes missing ones.

## 3. Inspect the proposed rule

```python
exec(UPDATED_STAGES['calibration-review'], globals())
```

Read the printed envelope, variant statuses and coverage. Check that the cohorts
and padding/repeat controls are what you intended. The 3x margin is an engineering
rule, not a confidence interval. Unresolved directions are not reliable estimates.
Then in a separate cell, record your review and freeze it:

```python
CALIBRATION_REVIEWED = True
CALIBRATION_NOTES = 'REPLACE WITH YOUR OBSERVATIONS'
run_command(['activation-calibration-freeze','--config',CALIBRATION_PATH,
             '--review-notes',CALIBRATION_NOTES], 'calibration-freeze')
```

Do not rerun the display cell after this to freeze: its defaults reset the checkbox.
Do not revise thresholds after looking at validation data.

## 4. Collect independent validation

```python
exec(UPDATED_STAGES['calibration-validate'], globals())
```

This uses 16 distinct examples. Failed batch candidates are excluded. Repeat or
padding failure, or an accepted wrong-sign/scale/count control, stops authorization.
Raw arrays and failure evidence remain saved. Send that evidence if it fails.

## 5. Profile eligible candidates

```python
exec(UPDATED_STAGES['profile'], globals())
exec(UPDATED_STAGES['review'], globals())
```

The old `profile/` remains untouched; the new path is `profile-v2-<rule hash>`.
This checks the independent longest-example stress workload. A mismatch stops;
there is no automatic tolerance increase. After reading the candidate table:

```python
SELECTED_BATCH_SIZE = 0  # replace with an actually passing batch from the table
BENCHMARK_REVIEWED = True
BENCHMARK_NOTES = 'REPLACE WITH YOUR REVIEW'
exec(UPDATED_STAGES['freeze'], globals())
```

## 6. Continue measurement and reporting

```python
exec(UPDATED_STAGES['measure'], globals())
```

Then separately:

```python
exec(UPDATED_STAGES['verify'], globals())
exec(UPDATED_STAGES['report'], globals())
exec(UPDATED_STAGES['read'], globals())
```

All important files remain below `SPAR/spar-pilot/artifacts`. The new calibration
is under the prepared run's `calibration/pilot3-noise-v2/`, and the production
execution/report use distinct v2 names. Scientific acceptance happens only after
the real GPU validation/profile and verified measurement complete.
