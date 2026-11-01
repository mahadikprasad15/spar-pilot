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
