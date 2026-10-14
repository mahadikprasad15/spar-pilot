# 10: Prepare blinded grading and add validated grade evidence

**What to build:** Prepare an opaque-ID bundle and separate mapping, validate imported LLM/human grades, compute reviewed agreement and attach a source-bound graded supplement when available. Human work does not gate the automatic report.

**Blocked by:** 08: Decode saved means and reveal controlled token lists.

**Status:** ready-for-agent

- [ ] Public bundle/import operations exclude condition labels, enforce declared rubric/task/model/cost settings and reject duplicate/misaligned/incomplete grade inventories before unblinding.
- [ ] No external submission occurs without explicit authorization; imported grades record provenance and human sample selection.
- [ ] Agreement metrics use independent worked examples; unblinding is recorded after grade validation.
- [ ] A validated supplement can be attached to the report through the predeclared grade contract; incomplete grades remain pending and do not overwrite automatic evidence.
