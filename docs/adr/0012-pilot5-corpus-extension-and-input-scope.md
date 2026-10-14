# Pilot 5 raw corpus extension and verification scope

Status: accepted selection procedure by the user, 2026-10-14.

Replay the same pinned FineWeb revision/config and first 2,000-document pool used in Pilot 3. Recover the original 150 exactly. Exclude their document IDs and full-text identities, as well as duplicates among candidates. Require at least 128 content tokens. Select 50 with a separate local RNG seed 42, preserving source order afterward. Keep legacy/new stratum labels. Selection uses no model outcomes or lens outputs; prior exposure is recorded explicitly and unknown history is not called unseen.

Raw encoding uses the pinned tokenizer JSON directly on CPU, with no inserted special tokens. Record actual special-token IDs and offsets. Preserve the legacy 128-content-token prefix exactly; a mismatch blocks preparation instead of replacing the document.

Verification has distinct scopes. Source audit validates trained adapter/checkpoint parents once explicitly. Corpus preparation checks the frozen parent input seal, every declared input payload and row identity, and binds its parent seal in the new corpus. It does not consume optimizer or adapter tensors. Restore checks selected config fingerprints; corpus verification checks all saved corpus/input hashes without network access. Later model capture must independently qualify adapter sources and numerical runtime. An input verification result cannot substitute for that qualification.

This avoids repeating unrelated training-state hashes during CPU data preparation while retaining the actual dependencies of each operation. No existing Pilot 4 validators or acceptance thresholds are changed.
