# 02: Rewire module

**Status:** ready-for-agent
**Type:** task

Clean-room `MaskedReRead`-style module for the pinned Qwen2 MLP (`spec.md` §Module), ranks 0/1/k, gate+up parts, off switch, per-neuron removal of new reads. Do not copy Max's code (D16); cite `max@a6dc314` and Louizos et al. 2018 in the module docstring.

- [ ] Red/green: spec tests 1–4.
- [ ] Confirm hidden 1,536, intermediate 8,960, 28 layers and SiLU from the pinned config in a test that reads a saved config fixture, not a download.

## Comments

## Answer
