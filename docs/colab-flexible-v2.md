# GSM8K flexible v2: CPU-only offline rescoring

Copy these two cells into Colab. They correspond to notebook sections 20–21. GPU evaluation is finished; these cells only read and rescore saved responses.

## 20. Load flexible v2 for CPU-only rescoring

Run these two new cells after the original five evaluations are complete and saved. A CPU runtime is sufficient. They work in the existing notebook or a fresh Colab notebook; no setup cells requiring a GPU are needed. Drive must contain the combined report and its source GSM8K files. This fetches a pinned tested code revision. Stop any active evaluation before switching code. No third-party Python packages, model downloads, or GPU inference are needed for the rescore command.

```python
from pathlib import Path
import json, subprocess, sys
from google.colab import drive

drive.mount("/content/drive")
REPO_DIR = Path("/content/spar-pilot")
ARTIFACT_ROOT = Path("/content/drive/MyDrive/SPAR/pilot1/artifacts")
V2_COMMIT = "be287098ce43f6c695d66b2d8a8555f033157aa4"
assert (ARTIFACT_ROOT / "reports/baseline-combined-logits-v2/results/results.json").is_file(), "Check Drive and artifact location."
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/mahadikprasad15/spar-pilot.git", str(REPO_DIR)], check=True)
dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip()
artifact_link = REPO_DIR / "artifacts"
if artifact_link.is_symlink() and artifact_link.resolve() == ARTIFACT_ROOT.resolve():
    dirty = "\n".join(line for line in dirty.splitlines() if line != "?? artifacts")
assert not dirty, f"Preserve local repository edits before switching code:\n{dirty}"
subprocess.run(["git", "fetch", "origin", V2_COMMIT], cwd=REPO_DIR, check=True)
subprocess.run(["git", "checkout", "--detach", V2_COMMIT], cwd=REPO_DIR, check=True)
print("Flexible v2 ready. No GPU or model download is needed.")
```

## 21. Rescore the saved GSM8K responses

Flexible v2 uses regex and exact numeric parsing without gold access during extraction. It accepts a single distinct numerical value in the final answer region, including prose, units, currency, boxed answers and inline markers. Multiple distinct quantities or conflicting answers remain invalid; caps stay incorrect. This is a post hoc scoring revision. Strict and flexible v1, all generated text, and the combined MMLU report remain unchanged. The new immutable report copies source files, stores source config/hashes, preserves all old/new scores and writes every changed extraction. Repeat with the same inputs/name to verify the existing report. Never choose a new grammar merely to reach the earlier 64% or diagnostic 46% figure. The source Qwen system-message discrepancy still applies.

```python
combined = json.loads((ARTIFACT_ROOT / "reports/baseline-combined-logits-v2/results/results.json").read_text())
gsm = next(cell for cell in combined["cells"] if cell["cell"] == "gsm8k-0shot")
source_run = ARTIFACT_ROOT / gsm["run_path"]
V2_REPORT = "baseline-gsm8k-flexible-v2"
subprocess.run([sys.executable, "-m", "pilot_eval", "rescore-gsm8k",
    "--responses", str(source_run / "results/responses.jsonl"),
    "--config", str(source_run / "config.json"),
    "--summary", str(source_run / "results/results.json"),
    "--name", V2_REPORT, "--output-root", str(ARTIFACT_ROOT)], cwd=REPO_DIR, check=True)
from IPython.display import Markdown, display
v2_dir = ARTIFACT_ROOT / "reports" / V2_REPORT
display(Markdown((v2_dir / "results/report.md").read_text()))
print("New scores:", v2_dir / "results/results.json")
print("Every changed extraction:", v2_dir / "results/changes.jsonl")
```
