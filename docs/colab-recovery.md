# Recover the remaining evaluations in the same Colab notebook

Keep your current notebook, GPU session and Drive artifacts. Add the four cells below after the failed full-run cell. Run them in order. Do not rerun the original section 9.

This preserves the completed batch-8 GSM8K and zero-shot MMLU runs. It restarts five-shot MMLU and runs both logit settings at batch size 2. The partial batch-8 five-shot run remains untouched. A smaller batch reduces memory pressure but does not guarantee every prompt will fit.

The recovery plan copies the original inputs exactly, including the known unintended system message. The combined report labels that discrepancy; this workflow does not certify protocol compliance. Scoring rules are unchanged.

On reconnect, mount Drive and rerun the original setup with the source plan's saved dependency constraints, restore the `run_cli`/`run_folder` helpers, then run these recovery cells. The update is pinned separately from the original notebook's `HARNESS_COMMIT`, so its existing identity metadata remains valid. No dependency upgrade is required.

## Cell 1: update the Python code

Make sure the previous evaluation process has stopped. This fetches the new code without changing any artifacts. It checks for local repository edits before switching revisions.

```python
from pathlib import Path
import json, os, subprocess, sys

REPO_DIR = Path("/content/spar-pilot")
ARTIFACT_ROOT = Path("/content/drive/MyDrive/SPAR/pilot1/artifacts")
SOURCE_PLAN = "baseline-batch8-v1"
RECOVERY_PLAN = "baseline-batch2-recovery-v1"
RECOVERY_COMMIT = "893c295770fe3d73d40290efc965c800cf2d275b"

assert (ARTIFACT_ROOT / "plans" / SOURCE_PLAN / "manifest.json").exists(), "Mount Drive and check SOURCE_PLAN."
assert "run_cli" in globals() and "run_folder" in globals(), "Restore the helper definitions from section 6 first."
dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip()
artifact_link = REPO_DIR / "artifacts"
if artifact_link.is_symlink() and artifact_link.resolve() == ARTIFACT_ROOT.resolve():
    dirty = "\n".join(line for line in dirty.splitlines() if line != "?? artifacts")
assert not dirty, f"Preserve these local repository edits before switching code:\n{dirty}"
subprocess.run(["git", "fetch", "origin", RECOVERY_COMMIT], cwd=REPO_DIR, check=True)
subprocess.run(["git", "checkout", "--detach", RECOVERY_COMMIT], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_DIR, check=True)
print("Recovery code ready; existing artifact files are unchanged.")
```

## Cell 2: fork the frozen inputs

No Hub resolution, dataset download or resampling occurs. A distinct recovery plan name avoids overwriting any earlier batch-2 attempt. The runner's console logs will now go to this recovery plan's notebook metadata directory.

```python
subprocess.run([sys.executable, "-m", "pilot_eval", "fork-plan",
                "--source", SOURCE_PLAN, "--plan", RECOVERY_PLAN, "--batch-size", "2",
                "--output-root", str(ARTIFACT_ROOT)], cwd=REPO_DIR, check=True)
recovery_manifest = json.loads((ARTIFACT_ROOT / "plans" / RECOVERY_PLAN / "manifest.json").read_text())
RECOVERY_CONFIG_PATHS = [ARTIFACT_ROOT / path for path in recovery_manifest["configs"]]
SESSION_META = ARTIFACT_ROOT / "notebook" / RECOVERY_PLAN
SESSION_META.mkdir(parents=True, exist_ok=True)
(SESSION_META / "recovery-code.json").write_text(json.dumps({
    "source_plan": SOURCE_PLAN, "recovery_plan": RECOVERY_PLAN, "harness_commit": RECOVERY_COMMIT,
    "allocator_configuration": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    "policy": "Reuse identical frozen prompts; restart remaining cells at batch 2; retain source runs."
}, indent=2) + "\n")
for index in (2, 3, 4):
    config = json.loads(RECOVERY_CONFIG_PATHS[index].read_text())
    print(config["scorer"], config["shots"], "shot; batch", config["batch_size"])
```

## Cell 3: smoke-test and run only the remaining settings

The fixed five-item smoke tests check execution at the smaller batch size; they do not cover every long prompt. Prompt/scorer behavior was already inspected in the source audit and is copied unchanged. The first recovery five-shot full run starts from zero. Subsequent attempts resume that new run normally. Completed source GSM8K and zero-shot MMLU are not invoked.

```python
RECOVERY_INDICES = [2, 3, 4]
for index in RECOVERY_INDICES:
    run_cli(RECOVERY_CONFIG_PATHS[index], audit_items=5)
RECOVERY_SUMMARIES = {index: run_cli(RECOVERY_CONFIG_PATHS[index]) for index in RECOVERY_INDICES}
```

## Cell 4: create and display the combined report

The selection explicitly chooses the original first two cells and the recovery last three. The report verifies completeness, source records and derived scores, scientific settings, prompt identity between text/logit settings, and runtime provenance. It performs no inference and changes no source runs. Incompatible, corrupted or incomplete sources stop reporting.

```python
REPORT_NAME = "baseline-combined-v1"
selection = {
    "gsm8k-0shot": SOURCE_PLAN,
    "mmlu_text-0shot": SOURCE_PLAN,
    "mmlu_text-5shot": RECOVERY_PLAN,
    "mmlu_logits-0shot": RECOVERY_PLAN,
    "mmlu_logits-5shot": RECOVERY_PLAN,
}
report_dir = ARTIFACT_ROOT / "reports" / REPORT_NAME
selection_path = report_dir / "selection.json"
report_dir.mkdir(parents=True, exist_ok=True)
if selection_path.exists():
    assert json.loads(selection_path.read_text()) == selection, "Use a new report name for changed source selection."
else:
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
console_path = report_dir / "report-console.log"
with console_path.open("a") as console:
    result = subprocess.run([sys.executable, "-m", "pilot_eval", "report", "--selection", str(selection_path),
                             "--name", REPORT_NAME, "--output-root", str(ARTIFACT_ROOT)],
                            cwd=REPO_DIR, stdout=console, stderr=subprocess.STDOUT)
if result.returncode:
    print("\n".join(console_path.read_text().splitlines()[-15:]))
    raise RuntimeError("Combined report stopped; inspect its source runs and console log.")
from IPython.display import Markdown, display
display(Markdown((report_dir / "results/report.md").read_text()))
print("Combined JSON:", report_dir / "results/results.json")
```

The combined report is the convenient entry point; original response JSONL/config/status files remain in their original run directories. It reports different batch sizes explicitly. It does not include or average the partial five-shot batch-8 attempt. Keep the whole artifact tree when exporting.
