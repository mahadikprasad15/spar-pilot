# 04: Validation set sampler

**Status:** ready-for-agent
**Type:** task
**Blocked by:** — (real IDs need 01)
**Group:** A

Seeded 128-question GSM8K train sampler with an exclusion record (D11). Build and test on fixtures now; freeze the real set once ticket 01 supplies the `sequences-v2` IDs.

- [ ] Red/green: spec test 12, including a malformed exclusion line.
- [ ] Exclusions from `catalog.find` and the `neuron_audit_sequences` manifest reader.
- [ ] Items built by the gold builder (same tokens, labels and mask).

## Comments

- 2026-11-24 (ticket 01): real set frozen with PR 13's code at `3ef8e43` in
  `artifacts/validation/sparse-rewiring-ticket04-v1/`: 128 IDs, `ids_sha256` `5ea47bc5…`, `items_sha256` `cc170735…`;
  837 excluded (gold 512 + `sequences-v2` 325). See `../source-records.md`.

## Answer
