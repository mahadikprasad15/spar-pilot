# Sampling SFT journal prototype

This is an isolated CPU-only prototype. It is not connected to the running
generation instrument and does not authorize resuming an existing plan with
changed code. Never run the benchmark on the active Pod volume: writes and
syncs can contend with the experiment.

The current instrument appends and syncs each scientific row. The prototype
serializes rows using the same JSON settings, writes them in order, and syncs
once when the caller calls `sync`. The caller must only seal a block after a
successful sync, must preserve/discard and replay an interrupted partial block,
and must never append to a sealed file. Any write, serialization, or sync
failure makes the journal unusable. Close does not imply sync or seal. This is
not a complete checkpoint implementation: a production integration must also
prove state, directory rename and directory durability, error recovery, and
all existing seal and provenance invariants.

The byte threshold bounds the pending buffer to approximately the threshold,
except that one encoded row may exceed it. Encoding and write temporary copies
also consume memory. The benchmark itself materializes its synthetic input and
is not a measurement of peak production memory.

Run a local benchmark into a new canonical directory:

```sh
python -m pilot_eval.sampling_sft_journal_benchmark \
  --output-dir artifacts/runs/sampling-sft/local-journal-benchmark-NEW \
  --rows 2500 --token-count 256 --max-bytes 1048576
```

The old benchmark path models the existing writer's operations; the CPU test
compares the prototype bytes directly with the actual existing writer. It
records encoding, write, flush, sync and a simple checkpoint time, counts,
size and hashes. This simple checkpoint is not the actual full block seal.
Mac timings do not establish Runpod global-volume performance. Use an independent
volume or wait until the live workload is finished before measuring that volume.
A real integration needs separate provenance, review and real-CUDA exact replay
qualification; it cannot simply reuse the existing frozen plan.

## Versioned CPU fixture and migration boundary

`pilot_eval.sampling_sft_journal_v2` supplies an explicitly non-production
fixture plan and block writer. Its plan binds hashes of the sampler and journal
implementation, and its block receipt binds the plan, records and state. Tests
exercise the actual sampler, verify byte identity with the old record writer,
inject errors, kill a child before and after a seal, and reject tampering. This
fixture does not implement the production lifecycle, previous-block chain,
directory-durability protocol, or GPU qualification.

Old generation blocks bind the old plan hash and previous seal; their RNG and
state are meaningful only after validating that chain and the original worker
evidence. The gate, selection, and handoff presently require all three arms
under one verified generation plan. Consequently neither copying old blocks
into the new plan, changing their seals, nor combining k0/k2 with a new k5 is
currently accepted. Supporting either requires a reviewed, versioned bridge
that retains original seals, verifies all source inventories and full runtime,
algorithm, cohort, retry, and code identity, and explicitly represents each
arm's provenance to the gate and handoff. A writer-only change still needs
real-CUDA exact equivalence and interruption/replay qualification. No such
bridge or new GPU-ready generation plan is supplied by this fixture.

For a future decision, compare remaining current-run wall time `R` with
`Q + N + B`, where `Q` is qualification and migration/setup time, `N` is the
new k5 time including retries and `B` is any backup/recovery overhead. Neither
is currently measured on the relevant storage, so no break-even claim follows
from the Mac benchmark. Running a storage probe on the live global volume can
compete with sampling; wait for completion or use an independent volume, and
label any different-volume result accordingly.
