# 02: Rewire module

**Status:** ready-for-agent
**Type:** task
**Blocked by:** —
**Group:** A

Clean-room masked read-change module for the pinned Qwen2 MLP (`spec.md` §Module): rank 1 and rank 0 (rank k supported), gate+up, off switch, per-neuron removal and transplant hooks for later checks. Do not copy Max's code (D16); cite `max@a6dc314` and Louizos et al. 2018.

- [ ] Red/green: spec tests 1–4 and 10b.
- [ ] Shapes read from a saved config fixture, never a download.
- [ ] Device placement: spec test 23.

## Comments

## Answer
