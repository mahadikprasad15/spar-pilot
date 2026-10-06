# Resume stopped Pilot 3 production with bounded hashes

The old child is stopped (`Z`/defunct). Keep the same Colab runtime/GPU and Drive.
Use the same execution config and directories. No setup, install, training,
calibration, profiling or freeze rerun is required.

## Cell 1 — update only code and helpers

```python
assert not live_children(), 'Inspect any live earlier child before switching code.'
RECOVERY_COMMIT = '8972c433ff00b7823906e64a1ccd291e6289d08e'
dirty = subprocess.check_output(['git','status','--porcelain'],cwd=REPO_DIR,text=True).strip()
assert not dirty, 'Preserve local checkout edits first; do not discard them.'
subprocess.run(['git','fetch','origin',RECOVERY_COMMIT],cwd=REPO_DIR,check=True)
subprocess.run(['git','checkout','--detach',RECOVERY_COMMIT],cwd=REPO_DIR,check=True)
HARNESS_COMMIT = RECOVERY_COMMIT
for module_name in list(sys.modules):
    if module_name=='pilot_eval' or module_name.startswith('pilot_eval.'):
        del sys.modules[module_name]
nb=json.loads((REPO_DIR/'notebooks/pilot-3-colab.ipynb').read_text())
updated={c['metadata']['stage']:''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'}
exec(updated['helpers'],globals())
EXECUTION, _, _ = restore_execution()
MEASUREMENT_DIR = ARTIFACT_ROOT / EXECUTION['run_path']
print('Same saved execution:',EXECUTION_PATH)
print('Same result directory:',MEASUREMENT_DIR)
```

## Cell 2 — verify saved work and record the reviewed code change

```python
run_command(['activation-recover-measurement','--config',EXECUTION_PATH,
             '--review-notes','Approved production fix: change only hash scheduling and batch publication; preserve numerical settings and verified old units.'],
            'production-recovery',MEASUREMENT_DIR)
```

This verifies the old calibration/profile/execution and marked payloads. It prints
how many of the 190 units are verified. It records original and actual code/runtime
identities in `meta/production-recovery.json`; frozen configs and old units remain
unchanged. It loads runtime metadata, not the model. Hardware, precision or package
changes are rejected. Send the error if it rejects; do not edit saved manifests.

## Cell 3 — resume measurement

```python
exec(updated['measure'],globals())
```

The model computes only missing units. An unfinished batch's pending checkpoints
are buffered until its final exact hash passes. The progress count can pause
while those checkpoints run, then advance as they are saved. The monitor now shows
batch, checkpoint and hash stages. A failure before the boundary check publishes
no new payloads; previously completed units remain safe.

For batch 8, a fresh run uses 38 batches × two boundary hashes = 76, plus startup
and final checks, instead of 380 per-checkpoint hashes. Forward passes, rank-1
checks, invariance checks and disk verification still cost time; this is not an
instant run. Real speedup has not yet been measured.

## After completion

Use the existing verify/report/read cells, or:

```python
exec(updated['verify'],globals())
exec(updated['report'],globals())
exec(updated['read'],globals())
```

Keep the original session variables and report name. If measurement had actually
completed before stopping, use verification/reporting instead of recovery.
