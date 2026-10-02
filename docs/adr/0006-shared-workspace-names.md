# Shared workspace names

The user approved these names on 2026-10-02 to remove the implication that
Pilot 2 data belong to a separate repository or live outside the Drive workspace:

- GitHub notebooks: `notebooks/`, previously `output/jupyter-notebook/`.
- Shared Drive workspace: `SPAR/spar-pilot`, previously `SPAR/pilot1`.
- Both notebooks' temporary Colab code checkout: `/content/spar-pilot`.

Rename the existing Drive folder in place, retaining its ID, parents and contents.
Keep existing run names and internal artifact folders. Frozen configs and
historical Git revision paths remain unchanged as provenance. Notebook links,
current defaults and instructions use the new names.

Completed historical evaluation configs contain absolute adapter paths under
the old root. Renaming the workspace does not make those configs location
independent. Rebuilding or resuming those GPU runs needs a separately validated
path migration and their original runtime/code revision. CPU v3 export auditing
reads the paired file directly at its new location and requires no such rebuild.
