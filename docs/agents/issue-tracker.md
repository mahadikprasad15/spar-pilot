# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Implementation issues are separate files at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Triage state is a `Status:` line near the top of each issue, using `docs/agents/triage-labels.md`.
- Append conversation history under a `## Comments` heading.

## Skill operations

- To publish a spec or issue, create its file under `.scratch/<feature-slug>/`.
- To fetch a ticket, read the referenced issue file. The user will normally provide its path or issue number.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`, with Notes, Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`. Its `Type:` is `research`, `prototype`, `grilling`, or `task`; its `Status:` is `claimed` or `resolved`.
- Blocking: `Blocked by: NN, NN` near the top. A ticket is unblocked when every listed ticket is `resolved`.
- Frontier: the lowest-numbered open, unblocked, unclaimed ticket.
- Claim: set `Status: claimed` and save before work.
- Resolve: add the answer under `## Answer`, set `Status: resolved`, and add a context pointer to the map's Decisions-so-far.
