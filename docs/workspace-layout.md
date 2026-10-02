# One repository and one persistent workspace

```text
GitHub: spar-pilot/
  notebooks/
    pilot-1-colab.ipynb
    pilot-2-colab.ipynb
  pilot_eval/                 shared evaluation, training and scoring code
  scripts/                   profiling and auxiliary commands
  tests/
  docs/

Drive: MyDrive/SPAR/spar-pilot/
  artifacts/
    cohorts/                 frozen dataset selections
    plans/                   frozen configs and training inputs
    runs/                    responses, adapters, logs and status
    reports/                 original and revised analyses
    notebook/                notebook console logs and review records
```

The Drive folder `SPAR/pilot1` was renamed to `SPAR/spar-pilot` in place.
Its children and existing run names were retained. The temporary code checkout
in Colab is `/content/spar-pilot` for both notebooks. Changing it does not move
Drive data. Older `/content/spar-pilot2` checkouts can be left in an existing
session; the updated notebook uses the shared name.

## Completed Pilot 2: next step

Open the current `notebooks/pilot-2-colab.ipynb` from GitHub in Colab. On a CPU
runtime, **run only section 12**. It is self-contained and uses the saved
`reports/<plan>-trajectory/results/paired-items.jsonl` to write a new
`reports/<plan>-flexible-v3-audit-v1/` report under the shared Drive artifact root.
It explicitly fetches the revision containing the audit command; the earlier
training revision does not contain that command. An already-open Colab copy
does not automatically receive GitHub notebook edits: reopen the current link
or copy the new section into it.

## Historical paths and resume

Archived configs record the paths and runtime used at execution. Those strings
are not rewritten. In particular, Pilot 2 evaluation configs contain absolute
adapter paths under the old Drive root. Existing reports, response files and
checkpoints are retained, but do not rerun historical training/evaluation or
the old comparison command after this rename expecting automatic resume.
That requires a validated path migration and the recorded runtime. Use the
new CPU section to audit the completed run without invoking those commands.

Historical `git show <commit>:output/jupyter-notebook/...` instructions retain
their old paths because those files exist at that historical commit. Current
GitHub and Colab links use `notebooks/...`.
