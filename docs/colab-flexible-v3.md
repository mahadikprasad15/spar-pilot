# Pilot 2: audit and rescore saved responses with flexible v3

This is CPU-only. It does not load a model, train, or generate responses.
Keep the original v2 trajectory report. V3 is a post-hoc scoring revision;
apply it to the baseline and every checkpoint together.

## Saved source location

The Pilot 2 notebook uses the same Drive artifact root as Pilot 1:
`/content/drive/MyDrive/SPAR/pilot1/artifacts`.
The `pilot1` directory name does not mean it contains only Pilot 1.

The completed trajectory's source file is:
`reports/pilot2-sft-fp32-l4-batch8-v1-trajectory/results/paired-items.jsonl`
under that artifact root. `/content/spar-pilot2` is the temporary code checkout;
the notebook's output display is not where the responses are stored.

## Run after checking out code containing audit-sft-scores

Use a CPU runtime, mount Drive, and restore the notebook variables `REPO_DIR`
and `ARTIFACT_ROOT`. Update the checkout to the published scoring revision
before running the new command. Preserve any local code edits before switching
commits. The revised command runs in a fresh subprocess, so cached notebook
imports do not affect scoring. Do not rerun training with a changed code pin.

```python
import subprocess
import sys
from pathlib import Path

paired = ARTIFACT_ROOT / 'reports/pilot2-sft-fp32-l4-batch8-v1-trajectory/results/paired-items.jsonl'
assert paired.is_file(), f'Missing saved responses: {paired}'
name = 'pilot2-sft-fp32-l4-batch8-v1-flexible-v3-audit-v1'
subprocess.run([
    sys.executable, '-m', 'pilot_eval', 'audit-sft-scores',
    '--paired', str(paired), '--name', name,
    '--output-root', str(ARTIFACT_ROOT),
], cwd=REPO_DIR, check=True)
from IPython.display import Markdown, display
display(Markdown((ARTIFACT_ROOT / 'reports' / name / 'results/report.md').read_text()))
```

The new report saves the source bytes and their hash, all 750 rescored pairs,
the strict-correct/v2 disagreement cases, per-checkpoint counts and uncertainty
ranges, code hashes, config and completion status. Rerunning with identical inputs
is safe; changed inputs require a new report name. Legacy scores remain attached.

## Why the correction matters

V2 can reject a valid `#### 3` if the preceding reasoning paragraph contains
other numbers and no blank line separates the answer. SFT increasingly produces
this layout. V3 first accepts the existing strict final-line contract, including
its multiple-marker and explicit-answer conflict checks, then falls back to v2
for prose answers. Extraction never consults the gold answer.

The export audit verifies saved scores, item alignment, prompts and the zero
checkpoint. It cannot independently verify the original training configs or
run hashes from the paired file alone. Paired intervals describe this cohort
and one training seed. See `docs/adr/0005-gsm8k-flexible-v3.md` for the rule and
its limits.
