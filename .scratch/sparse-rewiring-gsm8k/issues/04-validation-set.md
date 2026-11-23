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

## Answer
