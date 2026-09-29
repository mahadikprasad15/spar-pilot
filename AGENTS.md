# AGENTS.md

## Project
Evaluation and fine-tuning pilots for SPAR Team 2 (selective fine-tuning).
The code is a measurement instrument: correctness of scoring matters more than speed.

## Rules
- At the start of every session, run the tests and report the result.
- Use red/green TDD: write the test, run it, confirm it fails, then implement.
- Unit tests must run on CPU with no model download (mock the model).
- Never change an acceptance number or tolerance in spec.md to make a test pass.
  If a gate fails, stop and report.
- Every run writes a config file: model, adapter, prompt template, prompt indices,
  decoding settings, scorer, seed. Every run saves all responses as JSONL.
- Commit after each green step, with a short message.
- If a decision is not in spec.md or docs/adr/, ask. Do not choose silently.

## Commands
- Tests: `pytest -q`
