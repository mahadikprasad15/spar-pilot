# Recover a notebook progress-reader error

The notebook polls `meta/status.json` while a separate Python process evaluates
the model. An empty or partial read can raise `JSONDecodeError` in the notebook.
The old helper caught only keyboard interrupts, leaving its evaluation child
running after this error. The log alone cannot prove why the read was empty;
a Drive visibility delay is one possible cause.

For the reported incident, `pgrep` confirmed that PID 62557 was still evaluating
the existing logit 5-shot config. **Do not start another evaluation while that
process is running.** The completed 0-shot run is already saved.

## 1. Monitor the existing process without starting another

Add this cell to the same notebook. It reads progress and checks the process
identity every ten seconds. It changes no artifacts. If interrupted, it leaves
the evaluation running; repeat this monitoring cell to reconnect.

```python
import json, time
from pathlib import Path

EVAL_PID = 62557  # PID reported by pgrep for this incident
config_path = TIE_CONFIG_PATHS[4]
config = json.loads(config_path.read_text())
run_dir = run_folder(config)
last = None
while True:
    proc_dir = Path('/proc') / str(EVAL_PID)
    try:
        arguments = (proc_dir / 'cmdline').read_bytes().split(b'\0')
        state = (proc_dir / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        active = str(config_path).encode() in arguments and state != 'Z'
    except FileNotFoundError:
        active = False
    try:
        status = json.loads((run_dir / 'meta/status.json').read_text())
        progress = (status['state'], status['completed'], status['total'])
        if progress != last:
            print(f'{progress[0]}: {progress[1]}/{progress[2]}', flush=True)
            last = progress
    except (json.JSONDecodeError, OSError):
        print('Progress temporarily unavailable; checking the process again.', flush=True)
    if not active:
        break
    time.sleep(10)

summary_path = run_dir / 'results/results.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    print('Saved results:', summary['total'], 'items; accuracy:', summary['accuracy'])
    print('Next: run section 19 to verify and build the combined report.')
else:
    print('Evaluation process exited without a summary. Inspect the console/error logs, then install the helper fix below before resuming.')
```

## 2. Install the corrected helper for future calls

This fetches a pinned notebook-helper fix and loads only its function
definitions. It does not switch the active evaluation code, change dependencies,
run audits, or start inference. No new plan is needed. The corrected helper skips
unreadable progress snapshots and keeps waiting; config and final summary JSON
remain strict. Any other exception stops and waits for its own child process,
preventing the same orphan-process problem in future calls.

```python
import ast, json, subprocess
HELPER_COMMIT = '4280f0032e3e31795e2d447b1faa1f485b5d891a'
subprocess.run(['git', 'fetch', 'origin', HELPER_COMMIT], cwd=REPO_DIR, check=True)
notebook = json.loads(subprocess.check_output(
    ['git', 'show', f'{HELPER_COMMIT}:output/jupyter-notebook/pilot-1-colab.ipynb'],
    cwd=REPO_DIR, text=True))
source = next(''.join(cell['source']) for cell in notebook['cells']
              if cell['cell_type'] == 'code' and 'def run_cli(' in ''.join(cell['source']))
tree = ast.parse(source)
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(tree, 'corrected-notebook-helper', 'exec'), globals())
print('Corrected helper loaded; no evaluation started.')
```

If the existing process finished successfully, run the combined-report cell
(section 19). If it failed, inspect its logs before resuming with
`run_cli(TIE_CONFIG_PATHS[4])` using the corrected helper. Saved response IDs are
verified and skipped on resume. Never run two writers for the same run.
