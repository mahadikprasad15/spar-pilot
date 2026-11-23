# Tickets

Each ticket is sized for one implementation thread. Groups are waves: tickets in the same group can run in parallel once their dependencies are done.

| # | Ticket | Status | Depends on | Group |
|---|---|---|---|---|
| 01 | Source records and artifact locations | needs-info | — | A |
| 02 | Rewire module | ready-for-agent | — | A |
| 03 | Hard-concrete gate, shared mask and survivor export | ready-for-agent | 02 | B |
| 04 | Validation set sampler | ready-for-agent | — (real IDs need 01) | A |
| 05 | Plan and launch-record schema | ready-for-agent | — | A |
| 06 | Dynamics logs and settle diagnostics | ready-for-agent | 03 | C |
| 07 | Training loop | ready-for-agent | 03, 05, 06 | D |
| 08 | Checkpoints and restart | ready-for-agent | 07 | E |
| 09 | Evaluation wiring | ready-for-agent | 03 | C |
| 10 | Attribution source manifest and comparison | ready-for-agent | — (real inputs need 01) | A |
| 11 | Rank 1 vs rank 0 comparison | ready-for-agent | 03, 10 | C |
| 12 | Standalone analysis of each rank | ready-for-agent | 03 | C |
| 13 | Training-dynamics figure | ready-for-agent | 06 | D |
| 14 | Pod chain integration and real-path CPU test | ready-for-agent | 08, 09 | F |
| 15 | GPU qualification | ready-for-human | 14, 20 | G |
| 16 | Launch record | needs-info | 01, 04, 15 | H |
| 17 | Production run | ready-for-human | 16 | I |
| 18 | Report and catalog registration | ready-for-agent | 10, 11, 12, 13, 17 | J |
| 19 | Causal checks (proposed) | needs-info | 17 | J |
| 20 | ADR: sparse arms on A100-80GB | ready-for-agent | — | A |
