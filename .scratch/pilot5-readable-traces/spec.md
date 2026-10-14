# Pilot 5: Readable traces of rank-1 SFT and GRPO

Status: ready-for-agent

Test seam: approved by the user on 14 October 2026. Scientific execution settings remain subject to the explicit review gates below. Ready-for-agent authorizes implementation of this workflow, not invention of unresolved scientific conventions or paid runs.

## Problem Statement

The researcher wants to understand whether the training task is readable from the internal changes left by existing rank-1 SFT and GRPO adapters on unrelated text. The existing pilots supply adapters and validated paired-activation infrastructure, but their position-pooled summaries cannot answer the proposed early-position question or support the required sample-size analysis.

The researcher and colleagues also need a readable notebook they can run without repeated repair cells, ambiguous paths or expensive recomputation. Pilot 4 exposed dependency conflicts, fresh-runtime state loss, hardware changes, batch-sensitive numerical measurements, slow repeated parent verification and logs that showed liveness without completed work. Pilot 5 must address those failure modes as part of its workflow.

## Solution

Provide a separately named readable-trace experiment that reuses existing trained adapters and a fixed unrelated-text corpus. The default experiment uses raw text; an optional assistant-form script is exposed in a separate notebook section and remains disabled by default. Capture position-wise mean differences at every named activation site and retain per-example differences at selected blocks. Decode declared mean vectors with the logit lens and a qualified community Jacobian lens, save their ranked tokens and scores, compare them with controls, and produce source-backed research outputs.

A frozen plan determines sources, inputs, conditions, numerical settings, analysis definitions and retention requirements. Every stage either reuses verified evidence or computes only its missing sealed units. A fresh session restores saved selections and shows the next pending stage without requiring previous experiment cells to execute. Before paid computation, the workflow reports its actual matrix, bounded qualification cost, profile-based estimate and storage needs. Required evidence is durably verified before GPU release.

## User Stories

1. As a researcher, I want a separate Pilot 5 run identity, so that its evidence is distinguishable from Pilot 4.
2. As a researcher, I want to reuse existing SFT and GRPO adapters, so that I can study their traces without retraining.
3. As a researcher, I want to verify model, tokenizer and adapter identities, so that the comparison uses the intended sources.
4. As a researcher, I want the random adapter's construction and parent norms recorded, so that I know what it controls.
5. As a researcher, I want to see missing source evidence before loading a GPU model, so that I do not pay for a doomed run.
6. As a researcher, I want 200 fixed unrelated documents, including the original 150 where eligible, so that results connect to earlier experiments.
7. As a researcher, I want the added 50 documents selected by a declared procedure, so that corpus membership is reproducible.
8. As a researcher, I want exactly 128 counted content tokens per sequence, so that every content position has the same example count.
9. As a researcher, I want actual special-token behaviour recorded, so that position zero is not falsely called BOS.
10. As a researcher, I want content-relative positions distinguished from model positions, so that optional formatting does not silently change the measured location.
11. As a researcher, I want raw text to be the only default input form, so that the first execution remains focused.
12. As a researcher, I want an optional assistant-form script using the same documents, so that I can later examine context dependence independently.
13. As a researcher, I want assistant-form sources and outputs separately named, so that they do not replace raw results.
14. As a researcher, I want the capture checkpoint list displayed and reviewed, so that the full trajectory is not run accidentally.
15. As a researcher, I want bounded checks of both zero adapters, so that instrument correctness is checked without full zero-condition production capture.
16. As a researcher, I want one shared disabled-adapter reference within each compatible batch, so that base computation is not needlessly repeated.
17. As a researcher, I want all decoder block outputs and embedding outputs observed in one pass, so that capture does not multiply by the number of layers.
18. As a researcher, I want position-wise sums for all 128 positions, so that later readout windows do not require another model run.
19. As a researcher, I want retained per-example token differences at the declared six blocks, so that bootstrap and sample-size analysis remain possible.
20. As a researcher, I want differences computed before retention conversion and sums accumulated in high precision, so that storage choices are explicit.
21. As a researcher, I want retained-storage precision qualified, so that approximate stored evidence is not called exact.
22. As a researcher, I want relative mean shift and coherence alongside readouts, so that readability is interpreted with signal magnitude and cancellation.
23. As a researcher, I want pooled positions zero through four at block 14 to be the primary vector, so that the main hypothesis is fixed before outcomes.
24. As a researcher, I want secondary position and layer grids declared separately, so that searching for attractive lists does not redefine success.
25. As a researcher, I want direct lens token lists and scores saved, so that I can inspect the measurements before grading them.
26. As a researcher, I want positive and negative readouts distinguished, so that their interpretation is not conflated.
27. As a researcher, I want zero vectors marked uninformative, so that tied arbitrary tokens are not presented as traces.
28. As a researcher, I want the real final norm and unembedding conventions used, so that the logit lens measures the declared quantity.
29. As a researcher, I want community-lens provenance and dimensions verified, so that a file name does not substitute for compatibility.
30. As a researcher, I want a lens qualification failure to remain visible, so that an incompatible readout cannot support a scientific conclusion.
31. As a researcher, I want own-lens fitting separately approved, so that a fallback does not silently add expensive work.
32. As a researcher, I want random-adapter, random-direction and mean-activation controls kept distinct, so that each comparison has a clear meaning.
33. As a researcher, I want a frozen task-enriched token vocabulary, so that relevance scoring does not depend on observed outputs.
34. As a researcher, I want automatic vocabulary overlap described accurately, so that it is not mistaken for semantic ground truth.
35. As a researcher, I want the primary testing family and null procedure frozen, so that multiple readouts do not inflate apparent success.
36. As a researcher, I want direct SFT–GRPO contrasts distinguished from separate threshold tests, so that one passing and one failing is not treated as proof of a difference.
37. As a researcher, I want sample-size analysis to resample whole examples, so that token correlations are retained.
38. As a researcher, I want SNR definitions and undefined cases explicit, so that a matched-noise comparison is reproducible.
39. As a researcher, I want blinded grading bundles separated from their condition mapping, so that graders cannot inspect adapter labels.
40. As a researcher, I want grading configuration and cost reviewed before external calls, so that a large readout grid does not trigger accidental spending.
41. As a researcher, I want human-review selection and agreement saved, so that semantic grading can be audited.
42. As a researcher, I want prediction confidences supplied rather than copied from a draft, so that preregistration reflects actual beliefs.
43. As a colleague, I want readable code with explicit imports, named settings and stage objects, so that I can understand notebook execution.
44. As a colleague, I want each cell to state its hardware, inputs and saved outputs, so that I can follow the experiment without this chat.
45. As a researcher, I want CPU restoration to work without GPU-library imports, so that I can inspect and report results cheaply.
46. As a researcher, I want actual model-library imports checked after dependency installation, so that package version numbers alone do not imply readiness.
47. As a researcher, I want GPU and numerical runtime differences reported before inference, so that reconnection cannot silently mix evidence.
48. As a researcher, I want a second writer or GPU child blocked, so that concurrent sessions do not corrupt a stage.
49. As a researcher, I want interrupts to terminate and wait for their children, so that stopping a cell does not leave expensive hidden work.
50. As a researcher, I want completed sealed units reused and incomplete units redone, so that a disconnect does not discard the entire run.
51. As a researcher, I want stale status/PIDs distinguished from verified completion, so that apparent progress is not trusted as evidence.
52. As a researcher, I want progress by phase, condition, batch and durable unit, so that elapsed-time messages do not masquerade as percentage completion.
53. As a researcher, I want structured numerical failures naming the offending comparison and coordinates, so that another diagnostic rerun is not the first response.
54. As a researcher, I want hashing and verification time visible, so that expensive I/O is included in budget estimates.
55. As a researcher, I want bounded qualification declared before it starts, so that a small check cannot unexpectedly become a large experiment.
56. As a researcher, I want required shards durably verified before release, so that GPU-local storage loss cannot remove inference evidence.
57. As a researcher, I want corrupted or missing artifacts rejected without touching sources, so that repair preserves valid evidence.
58. As a researcher, I want a research narrative with plots, exact values and limitations, so that collaborators can interpret results without reading execution logs.
59. As a researcher, I want null results and unavailable readouts reported honestly, so that qualification or measurement failures are not recast as scientific absence.
60. As a researcher, I want export inventory and storage duplication disclosed, so that packaging does not unexpectedly exhaust durable storage.

## Implementation Decisions

### Scope and identities

- The default form is raw unrelated text; assistant-form support is a separately invoked optional extension sharing the corpus and scientific code. User-form execution is out of scope.
- The implementation supports an explicit selected checkpoint list. Final step 64 anchors the primary claim; endpoint-only scope is recommended but not yet approved. Missing scope approval blocks execution-plan freeze, rather than silently launching every checkpoint.
- Existing model/tokenizer revisions, learned adapters and random construction are immutable source inputs. Pilot 4 supplemental coefficient/KL completion is not required.
- All capture sites have explicit names: embedding output and decoder blocks zero through 27. The primary block is block index 14, not the fifteenth entry of an ambiguously indexed array.
- Model calls capture all sites together. Execution counts distinguish sequence forwards, batched model calls and additional qualification work.
- Input identity includes document revision/ID, content and full-document hashes, token IDs, actual special tokens, content-position map, form and any prompt/template hashes.
- The new 50-document extension is outcome-blind. Preserve legacy/new stratum identities and record reviewer exposure to earlier pilots, distinguishing scalar outcomes from token lists and known exposure from unknown history. The original 150 documents are recovered from saved identities and checked for new-builder eligibility. A discrepancy blocks the shared-cohort claim; no replacement or retokenization mismatch is hidden.

### Components and state

- Build a source-inventory component, corpus/form builder, per-position capture instrument, lens adapter, controlled readout/analysis component, blinded-grade import component, research reporter and notebook-session facade.
- These expose named public stage operations for inspect, prepare, qualify, review/freeze, capture/resume, durable verify, readout, analyze, grade import/unblind, report and export. Notebook and command-line callers use the same operations.
- Prefer additive reuse of existing locks, atomic writers, safe paths, source validators, one-model ownership and integrity boundaries. Wide refactors of earlier pilots are not part of this work.
- Frozen parent identities and execution plans determine compatibility. Mutable notebook selections are preferences, not scientific approval or completion.
- Stage states distinguish missing, prepared, qualified, awaiting review, frozen, running, failed, durably complete and reported. Lens qualification also records incompatible/insufficient-provenance status; successful lens qualification is distinct from completing its inspection operation. Readiness inventory restores these states without starting model work.
- Publish a sealed batch only after its scientific checks and integrity boundary pass. Failed or partially staged units remain diagnostics and are not counted in accepted aggregates.
- Partition work so interruption repeats bounded incomplete units. Record readout/analysis units too where they have meaningful cost; do not make a large downstream stage all-or-nothing without explanation.
- Concurrent notebooks may perform independent stages with distinct writers. Shared selections must be merged atomically or written through a single explicit selection owner. No two writers may publish the same unit.

### Capture and storage

- The core quantity is adapted minus base activation for identical supplied inputs. Use FP32 model activations and FP64 accumulated sums.
- Capture-plan freeze solely owns sites, positions, pooling, conditions, precision, retention, the primary identity and predictions. Persist per-position sums for all 128 content positions at all 29 named sites. Persist retained per-example deltas at blocks 14 and 17–21, plus the baseline evidence required by the analysis plan.
- Additionally persist FP64 per-example pooled primary-window delta/base vectors and required norm/count scalars at all 29 sites. Together with the six-block position-wise evidence, this is a superset of the allowed whole-example analysis family; unsupported per-position analyses at other sites require a new capture extension. Reveal-policy choices cannot expand the frozen byte contract. At the model width 1,536, pooled evidence costs about 67.97 MiB per condition, separately from the 450 MiB six-block token evidence. The source proposes FP16 retained deltas. Treat retained precision as a reviewed storage setting requiring a quantization/ranking/bootstrap sensitivity check; full-precision accumulation does not make converted shards exact.
- Reuse a disabled-adapter reference within a compatible frozen batch only while base identity and disabled-adapter invariance hold.
- Freeze example ordering, batch membership, numerical settings and capture pass order. Exact repeat behaviour and cross-batch/hardware sensitivity are separate claims.
- Bounded zero checks cover both learned zero adapters and a declared diagnostic cohort. Cohort size, selection and exposure must be reviewed before qualification; no full production zero-condition capture is required.
- Local disk is staging. Required retained shards, summaries, identities and diagnostics must exist in verified durable storage before the GPU-release message is allowed.
- Profile free space, staging/durable transfer volume and readout workspace. Resume sync without re-copying valid units; incomplete destinations are never mistaken for complete evidence.

### Lenses and analysis

- The logit lens preserves the pinned final norm gain and unembedding convention. Save ranked positive/negative tokens and raw scores with vector identity.
- Lens inspection starts from the stable source/site contract without requiring production capture. Its status and any capture-site requirements are reviewed before capture freeze. Failed qualification does not silently authorize own fitting or a logit-only primary experiment; either is an explicit separate approval. The Jacobian artifact is community-first and must pass revision/hash, dimensions, source-site, target-representation and bounded independent qualification checks. Own fitting requires explicit separate approval and measured budget. Unsupported sites are unavailable, not fabricated.
- The primary vector averages content positions zero through four per example at block 14, then averages examples; raw form and final step 64 are fixed. Secondary windows, individual positions, other sites, intermediate checkpoints and ablations are labelled separately.
- Random adapter, Gaussian-vector null, base activation and adapted activation readouts are distinct controls. One random adapter is not a significance distribution.
- Automatic relevance is the fraction of top-20 token IDs in a frozen 200-token task-enriched vocabulary. The vocabulary's count definition, smoothing, exclusions and tie handling are reviewed before readouts.
- The capture plan alone owns the single primary-vector definition and all capture-affecting commitments. Vocabulary/inference freeze references that identity and chooses only rules supported by the retained-data family. Corpus/capture freeze is separate from vocabulary/inference freeze: capture does not consume the vocabulary. Freeze the reference vocabulary and primary multiple-testing family, null statistic/correction, random seeds, null counts, bootstrap settings, declared sample sizes and SNR estimator before revealing any learned lens lists, including diagnostic previews. Numeric/toy/base-model qualification may proceed without learned task-token reveal. Missing choices block analysis-plan freeze. Results remain descriptive until these choices are approved.
- Predeclare a primary robustness result for legacy 150 versus new 50 documents, with subset-size and noise/uncertainty context. No equal-power claim follows from comparing differently sized strata. Preserve whole-example pairing for bootstrap and sample-size analysis; individual positions within an example are not independent samples. State the precision/source used for each derived result.
- Freeze grader task candidates, rubric, model/version/settings, sample selection, cost limit and human review rules before external submission. Keep condition labels out of the grading bundle; unblind only a validated imported grade set. Do not auto-submit grader calls during a raw capture run.
- Qualified readout tokens are direct experimental outputs. Semantic grading and automatic overlap assess these outputs; neither replaces or generates the lens lists.
- The notebook reveal barrier prevents accidental exposure through supported operations; it is not an access-control guarantee against opening raw data. Record known bypasses and prior exposure. Automatic reporting does not depend on human/LLM grade completion; a predeclared grade schema supports a later validated supplement. Required raw release validation is independent of optional assistant execution. Missing qualification/evidence, pending human grading and genuine null findings are distinct report states. Never call a failed instrument a reportable scientific null.

### Notebook and operational repair requirements

- Provide a dedicated notebook with readable normal Python, explicit imports, named settings/objects, and concise purpose/hardware/input/output notes. Do not expose executable expressions as editable string fields. No pasted patch cells should be required for supported operation.
- Separate CPU preparation/reporting from GPU qualification/capture. Restore from a fresh kernel with the selected source and run identities; setup does not regenerate earlier pilot results.
- Pin tested harness/dependency releases and show what code is actually running. Check imports in a fresh child process before inference; handle optional vision/audio package conflicts explicitly while preserving unrelated user state.
- Compare source identity, scientific numerical settings and runtime compatibility separately. A changed GPU, dependency version or code identity is not automatically scientific corruption, but it must not silently reuse incompatible execution evidence. A separately qualified bridge/new execution is an explicit reviewed action, never a forced rebuild of previous pilots.
- Profile freeze explicitly enumerates base, learned checkpoint selections, random-64 and bounded zero checks, binding the existing random-control construction and parent norms. Freeze user/team predictions before production. A failed correctness gate stops capture; an unaffordable profile stops spend and offers a declared reduction ladder with separate scientific approval instead of changing the primary cohort/sites/retention under pressure. Qualify only the declared bounded workload. Profile qualification, model loading, forward work, hashes, saves and transfers separately and include their costs in total estimates. Do not promise the source document's under-one-hour budget before measurements.
- Prasad owns vocabulary/inference, lens-fallback and budget approvals; timeout is not approval. The predeclared lens-failure action is save diagnostics/status, stop paid model work, release safely and ask about logit-only scope or separately approved fitting. Budget rejection similarly stops production while CPU review continues. Vocabulary review blocks reveal but not qualified capture. A paid GPU must not be held for long CPU-only stages by design. Surface the expected runtime grouping and durable stop/reconnect boundaries before launch.
- Verification has a visible scope, phase and measured cost. Preserve hash checking and integrity guarantees while avoiding repeated scans of large immutable parents without an explicit verification strategy. A cached verified record must bind its parent seals, verification scope and trust assumptions; invalidated parents force re-verification.
- Stream combined output and persist status/logs. Show phase, completed durable units/total, current condition/form/checkpoint, batch size, throughput and an evidence-based ETA when available. Label heartbeat-only output as liveness.
- Failures save the comparison type, source identity, token/site coordinates, observed/reference/allowance and bounded worst cases where applicable. Add a bounded replay path; retain original thresholds unless the user approves an amendment. The approved Pilot 4 coefficient policy is not blindly copied onto different Pilot 5 quantities.
- Wait for actual child exit; interrupt terminates/waits for the owned child and preserves valid evidence. Lock contention, stale PIDs and nonzero exits get specific actionable messages.

## Testing Decisions

- The user approved one highest-level saved-workflow seam: the public operations used by the notebook, covering preparation through qualification/freeze, interrupted/resumed capture, readouts, analysis and report. Test externally visible artifacts, scientific outputs and failure behaviour; avoid private-helper snapshots or assertions that merely mirror implementation.
- Controlled model/dependency boundaries allow real orchestration, locks, writers, seals and recovery to run offline on CPU. They do not establish target-GPU numerical correctness.
- Independent tiny-model and hand-worked math oracles cover paired differences, named sites, position mapping, means, primary pooling, retained precision, final norm gain, lens ranking and relevance definitions. Wrong signs, wrong maps and corrupt sources must be rejected.
- Prior art includes the existing public activation-measurement interruption/resume tests, supplemental full-workflow source-preservation tests, coefficient batch-sensitivity/strict-gate tests, and fresh-kernel/active-child notebook-session tests.
- Extend failure coverage to dependency conflicts, missing CUDA, changed runtime, ambiguous compatible source selection, stale process/status, interruption during staged save or durable sync, disk-full/missing durable shards, corrupt seals, and independent concurrent sessions. No failure should start an extra model or overwrite a valid parent.
- The initial grade contract is only status and schema version; detailed fields are deferred to actual grading/report consumers. Label readable-code review as a human judgment, separate from automatic cell parsing/environment/recovery tests. Run a minimal recorded fresh-Colab install/import/durable-write/reconnect smoke before major dependent slices, then a final raw-path deployment regression. Reuse writer locks with one bounded conflict/source-preservation check rather than an extensive standalone contention framework. A simulated environment/bootstrap check validates orchestration offline. A recorded real fresh Colab check separately validates installation, imports, actual CUDA/runtime, bounded qualification, disconnect/reconnect and reuse without source regeneration.
- Include the optional assistant extension at the same public seam with a small declared fixture: correct content/header mapping, distinct formatted base reference, separate identity/output, disabled default and interrupted recovery.
- Do not widen spec tolerances to pass tests. Scientific parameters not approved in this conversation stay explicit review inputs. CPU tests make no pretrained model downloads or paid grader calls.

## Out of Scope

- New SFT/GRPO training, additional training seeds, regeneration of existing behavioural scores, or requiring Pilot 4's optional coefficient/KL branch.
- User-turn implementation for this first delivery, automatic assistant extension execution, and automatic full-checkpoint capture.
- Steering, Patchscope, SAE readouts and interpretability agents.
- Automatic own-lens fitting, silent lens substitution, automatic external grading expenditure and confidence values invented for the researcher.
- Claims of broad capability retention, exact cross-device equality, causal mechanism, absence of all possible traces, or a null distribution from one random adapter.
- Large architectural rewrites of previous pilots and destructive cleanup/replacement of their artifacts.

## Further Notes

The main input form, pooled primary vector, bounded zero checks, community-first lens and durable retention are approved. Endpoint-only scope, the secondary readout grid, final statistical/grade settings and some numerical qualification definitions are not yet approved. Their review gates are part of the implementation contract; readiness to implement is distinct from readiness to execute a frozen scientific run.

A source-and-decision map and a more detailed design document accompany this spec. The reference paper and code were inspected, but exact community-lens binary metadata still require verification. The optional assistant implementation is a contextual extension, not a claim that Minder's default extraction requires chat formatting.

Existing-data first looks are exploratory. Freeze primary definitions and the reference vocabulary before learned outcome lists are exposed, including bounded previews; disclose any earlier exposure. Sample-size and SNR analyses depend on retained per-example evidence and must respect its stored precision.
