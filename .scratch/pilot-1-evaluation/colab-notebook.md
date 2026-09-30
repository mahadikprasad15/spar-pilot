# Guided Colab notebook

Status: resolved

Deliverable: `output/jupyter-notebook/pilot-1-colab.ipynb`, with 27 cells (11 executable Python cells).

The notebook covers project context and B0/module connections, baseline definitions and historical reference limits, scoring/decoding explanations, GPU setup, Drive persistence, pinned repository/dependency versions, frozen preparation, prompt previews, audit execution and manual review, full evaluation/resume, uncertainty ranges, notes, export, and troubleshooting.

Verification: all Python cells parse; outputs are cleared; a local fake-boundary execution exercised preparation, audit inspection, refusal of an unapproved full run, approved full execution of all five cells, results display, and completed-run resume. The harness suite remains 62 passing CPU tests.

Limit: the notebook has not been executed in a real Colab runtime. Drive authorization, dependency installation and GPU inference must be verified there. It opens against the pinned harness commit `55be867d9b0757dcd3c18b23352f644efeb062f3`.
