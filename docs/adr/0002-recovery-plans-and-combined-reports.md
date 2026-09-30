# Preserve source runs when reducing evaluation batch size

The user authorized recovery of unfinished MMLU settings at a smaller batch size and one combined report. Existing configs and raw records remain immutable. A derived plan copies original revisions, selected IDs, dev examples and rendered prompts, changing only batch size, execution IDs and input locations; it records its parent plan. It does not resolve new Hub revisions or import partial response records.

The combined report explicitly selects one completed source per pilot cell. It validates records and aggregates, compatible scientific settings and frozen cohorts, shared text/logit prompts, and runtime provenance. Batch size may differ across selected cells and is shown alongside source paths and hashes. This is a collection of separately configured runs, not a claim that they had identical execution settings.

The legacy system-message discrepancy is preserved and flagged. Completion means complete verified source records, not compliance with the no-system-message protocol. Partial sources, conflicting source settings or corrupt records cannot produce a final report. Fixing the prompt protocol or expanding scoring remains separate work.
