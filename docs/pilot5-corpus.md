# Pilot 5 corpus preparation

Use the shared Pilot 5 notebook on CPU. No model weights or GPU computation are needed.

| Cell | Operation | Saved output |
| --- | --- | --- |
| 1–2 | Mount Drive, load pinned code, restore selected source paths | Restored state; no repeated checkpoint audit |
| 7 | Record approved method and honest prior exposure | Immutable `notebook/<experiment>/<corpus>/review.json` |
| 8 | Recover 150 legacy documents, select 50 disjoint documents from the pinned first-2,000 pool, tokenize and seal | `plans/<corpus>/traces.corpus.json`; input documents, pool audit, tokenizer files, manifest and completion seal |
| 9 | Verify saved inputs without network or retokenization | Public verification output; selected corpus retained across fresh runtimes |

For cell 7, keep the declared procedure and seed 42. Set `reviewed` to true only after checking the method. Notes can state: “Approved same pinned FineWeb first-2,000 pool; preserve all 150 legacy identities and their exact raw prefix; exclude duplicate/legacy document and text identities; select 50 eligible documents with local seed 42 and preserve source order. No model outcomes or lens outputs used for selection.”

Record prior exposure honestly: whether the reviewer saw earlier scalar results, token lists, both, or neither, and what is unknown. The added documents are called new to this cohort, not automatically unseen. Do not invent a clean exposure history.

Initial duration depends on streaming/download speed and is not benchmarked yet. Cell 8 prints actual pool progress every 250 documents. Identical completed runs verify and reuse saved inputs without accessing the tokenizer or dataset. Interrupted unsealed preparation may repeat the small data scan; no GPU work is lost. After completion, fresh cells 1–2 and 9 suffice to verify the saved cohort.

This stage fixes the common text and token positions used to compare SFT, GRPO and random changes later. It produces no activations, lens tokens or grading. Adapter source qualification, lens compatibility and capture settings remain later gates.
