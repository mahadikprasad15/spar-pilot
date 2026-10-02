# Recover exact logit ties in your current Colab notebook

Add and run the following four cells after the failed recovery cell. No new notebook is needed. These correspond to sections 16–19 in the updated notebook.

## 16. Update code for logit ties

Run these four new cells in the **same notebook**, after the tied-logit error. Stop the old evaluation first. These cells preserve completed GSM8K (150), text 0-shot (1140), and text 5-shot (1140) runs. Do not rerun sections 9 or 14. On reconnect, restore Drive, the saved dependency environment, and section 6 helpers first. No dependency upgrade is required. This update checks local repository edits and fetches a pinned tested commit.

```python
from pathlib import Path
import json, os, subprocess, sys

REPO_DIR = Path("/content/spar-pilot")
ARTIFACT_ROOT = Path("/content/drive/MyDrive/SPAR/spar-pilot/artifacts")
SOURCE_PLAN = "baseline-batch8-v1"
RECOVERY_PLAN = "baseline-batch2-recovery-v1"
TIE_PLAN = "baseline-logits-v2"
TIE_COMMIT = "7ee0762cff41035520b28438b25141af42bdba0a"

assert (ARTIFACT_ROOT / "plans" / SOURCE_PLAN / "manifest.json").exists(), "Mount Drive and check SOURCE_PLAN."
assert "run_cli" in globals() and "run_folder" in globals(), "Restore the helper definitions from section 6 first."
dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip()
artifact_link = REPO_DIR / "artifacts"
if artifact_link.is_symlink() and artifact_link.resolve() == ARTIFACT_ROOT.resolve():
    dirty = "\n".join(line for line in dirty.splitlines() if line != "?? artifacts")
assert not dirty, f"Preserve these local repository edits before switching code:\n{dirty}"
subprocess.run(["git", "fetch", "origin", TIE_COMMIT], cwd=REPO_DIR, check=True)
subprocess.run(["git", "checkout", "--detach", TIE_COMMIT], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_DIR, check=True)
print("Recovery code ready; existing artifact files are unchanged.")
```

## 17. Freeze the revised scoring plan

The approved scorer `mmlu-logits-v2` records exact top ties as invalid and incorrect, keeps them in the denominator, and continues. It chooses no tied letter and gives no fractional credit. Nonfinite logits still stop. This command verifies source configs, inputs, saved scores and error batches, then freezes a cache of their raw logits in a new plan. It copies the same sample, prompts, model, batch size 2 and precision. Source files remain unchanged. The printed cache count includes the tied batch; the original logit 5-shot run has no full-run cache. Runtime provenance must match when replaying logits.

```python
subprocess.run([sys.executable, "-m", "pilot_eval", "revise-logits",
                "--source", RECOVERY_PLAN, "--plan", TIE_PLAN,
                "--output-root", str(ARTIFACT_ROOT)], cwd=REPO_DIR, check=True)
tie_manifest = json.loads((ARTIFACT_ROOT / "plans" / TIE_PLAN / "manifest.json").read_text())
TIE_CONFIG_PATHS = [ARTIFACT_ROOT / path for path in tie_manifest["configs"]]
SESSION_META = ARTIFACT_ROOT / "notebook" / TIE_PLAN
SESSION_META.mkdir(parents=True, exist_ok=True)
(SESSION_META / "code.json").write_text(json.dumps({"harness_commit": TIE_COMMIT,
    "source_plan": RECOVERY_PLAN, "plan": TIE_PLAN}, indent=2) + "\n")
for index in (3, 4):
    config = json.loads(TIE_CONFIG_PATHS[index].read_text())
    print(config["scorer"], config["shots"], "shot; batch", config["batch_size"],
          "cached items:", len(config["replayed_item_ids"]), "scorer:", config["scorer_version"])
```

## 18. Resume logit evaluations only

The new runs have separate directories and provenance links to cached source logits. Logit 0-shot reuses verified saved outputs and computes missing items; logit 5-shot starts its full 1140-item run. A cache-only batch does not load the model. Repeated execution verifies completed runs and resumes partial ones. Completed text and GSM8K evaluations are not invoked. A new scoring audit is unnecessary for this narrowly authorized change; prompts are identical to the reviewed audits.

```python
# Only logit settings: completed GSM8K and text results remain selected below.
# Cached raw logits are rescored; the model processes only missing item IDs.
# Repeat this cell after interruption to resume the new runs.
TIE_SUMMARIES = {index: run_cli(TIE_CONFIG_PATHS[index]) for index in (3, 4)}
```

## 19. Combine original text results with scorer v2

This report selects the original batch-8 GSM8K and text 0-shot runs, the completed batch-2 text 5-shot run, and the two revised logit runs. It verifies all sources, reports accuracy ranges and tie counts/rates, and records each source path. It retains the known system-message discrepancy and historical-gap caveats. Keep the entire artifact tree when exporting. Do not use the older section 15 selection for the revised logits.

```python
REPORT_NAME = "baseline-combined-logits-v2"
selection = {
    "gsm8k-0shot": SOURCE_PLAN,
    "mmlu_text-0shot": SOURCE_PLAN,
    "mmlu_text-5shot": RECOVERY_PLAN,
    "mmlu_logits-0shot": TIE_PLAN,
    "mmlu_logits-5shot": TIE_PLAN,
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
