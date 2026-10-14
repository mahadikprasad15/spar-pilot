# Pilot 5 primary vector and durable retained evidence

Status: accepted by the user, 2026-10-14.

The primary readout uses the pooled mean of content positions 0–4 at decoder block 14, rather than five separate primary position tests. Individual positions remain secondary; multiple readouts and arms still require an explicit primary testing family. This choice changes the source draft's position-specific primary interpretation and must stay visible in reports.

Retained per-example activation shards must reach verified durable storage before GPU release. Temporary-only storage reduces transfers but makes subsequent bootstrap and sample-size questions depend on reruns; the user chose durable evidence. Use the community Jacobian lens only after compatibility/qualification checks; own-lens fitting is a separately approved fallback rather than an automatic expense.

The user subsequently approved bounded zero-adapter qualification for both arms. Its frozen diagnostic cohort limits the coverage claim; no full zero-condition capture is required.

The user then approved raw-only input for the first execution, matching the reference extraction configuration. User/assistant forms are deferred to a separate extension; this run cannot test role/template dependence.

Assistant-form implementation is subsequently approved as an optional separately invoked notebook script, disabled by default. Share capture/readout code and document membership, but freeze and persist separate formatted inputs and outputs; raw base activations are not interchangeable with assistant-form baselines. User form remains deferred.
