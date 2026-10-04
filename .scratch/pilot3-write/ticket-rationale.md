# Why these implementation tickets

## The reasoning process

1. Extract the promises from the spec: fixed inputs, correct measurements,
   verified resume, understandable results and a usable Colab workflow.
2. Trace each promise through the design: what evidence enters, what processing
   occurs, what evidence is saved and what a user can inspect.
3. Find the smallest complete behavior we can build and verify. A ticket should
   produce a usable result, not merely a group of functions or one software layer.
4. Put risky assumptions early: source identity, token ownership and real hooks
   precede the expensive full-cohort workflow.
5. Add only genuine blocking edges: the next slice must require a capability or
   verified artifact that the preceding slice creates.
6. Include tests and failure behavior in every slice; do not postpone correctness
   to a final testing ticket.

## The six slices and why their order matters

| Slice | Inspectable result | Risk retired | Why the next stage needs it |
|---|---|---|---|
| Frozen inputs | Source manifest, complete token sequences, masks and audit | Wrong sources, hidden prompt changes, silent truncation | Instrument checks need known matched inputs |
| One validated batch | Known-correct block/module summaries for five checkpoints | Hooks or contribution math measuring the wrong quantity | Profiling must time the real instrument |
| Reviewed batching | Throughput/memory/agreement evidence and immutable batch manifest | Long-input OOM and batch-dependent numerical differences | Production must use a declared passing plan |
| Full resumable measurement | Verified complete scientific shards | Lost progress, duplicate accumulation, corrupt completion | Scientific reporting needs complete valid evidence |
| CPU report | Reconstructed values, intervals and plots | Misleading aggregation, null handling or labels | The notebook needs stable user-visible outputs |
| Guided Colab | Explained execution and recovery path | Setup/version/path confusion | The researcher can run the complete workflow safely |

The chain is conservative for single-agent implementation. It does not mean
every line of the notebook must wait until the report exists: explanatory drafts
can be written earlier, but the completed notebook ticket requires integration.

## Mental models

- **Tracer bullet:** a narrow working path through the system provides evidence
  that the architecture can connect. It is retained production code, rather than
  a disposable mock demonstration.
- **Risk retirement:** choose early slices that test assumptions capable of
  invalidating later work. Here the instrument matters more than cosmetic polish.
- **Dependency graph:** blockers describe necessary inputs, not preferences or
  arbitrary ordering. Do not invent parallelism where shared contracts remain
  unsettled; do not invent blockers merely because two tickets share a file.
- **Contracts:** callers know inputs, outputs and failure rules. Private function
  structure can change while those promises remain stable.
- **Layered verification:** focused math tests diagnose errors, workflow tests
  check connections, real-library CPU tests check integration, and GPU preflight
  checks the actual execution environment.
- **Feedback loop:** implement one behavior, run its test, inspect what it taught
  us, then proceed. Avoid writing all imagined tests followed by all imagined code.

## Useful sources

These are influential explanations, not a ranking of the world's best programmers.

- David Thomas and Andrew Hunt's [The Pragmatic Programmer](https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/)
  includes tracer bullets, design by contract, decoupling and reversibility.
- Martin Fowler's [Test Pyramid](https://martinfowler.com/bliki/TestPyramid.html)
  describes a balanced test portfolio rather than relying on expensive broad
  GUI tests alone.
- Ham Vocke's [The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
  explains practical test levels and user-journey coverage.

Our application of these principles is specific to a research instrument:
correctly executing software is necessary, but does not prove a causal scientific
claim. The spec, tests and report must preserve that distinction.
