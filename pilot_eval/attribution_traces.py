"""Ticket 01: base-correct trace preparation for adapter attribution (spec §Data and provenance, §Metrics).

Frozen ``sequences-v2`` math prompts -> untuned greedy generation under the pilot protocol at a frozen batch
manifest -> raw responses sealed *before* any processing -> flexible-v3 scoring -> teacher-forcing rows
(prompt IDs + generated IDs exactly; nothing is re-tokenized) with masks -> accept/reject records, counts and
the minimum check after every mask exclusion.

- Prompt IDs are the frozen sequence's prefix through the assistant header (``input_ids[:reply_start + 1]``);
  ``build_example`` already refused any token crossing that boundary.
- Stop IDs are sealed from the decoder configuration actually passed to generation (pilot backend:
  ``eos_token_id`` only, i.e. {151645}; 151643 is padding). Any other metric stop set is refused unless it
  is labelled as a decoding change.
- Masks are target positions (``attribution_metrics`` reads logits one step earlier):
  ``trace_targets`` = every generated content token (first included; the final ``<|im_end|>`` excluded);
  ``stop_targets`` = content tokens that end at or before the start of the last non-empty line (character
  offsets from incremental decoding; a token crossing that newline belongs to the last line). An empty
  stop mask is recorded with a reason; the row stays in trace analyses but not in the post-exclusion count.
- Visibility: calibration ``visible``; discovery ``attribution-only``; heldout (confirmation)
  ``blind-until-freeze``: ``load_rows`` refuses it until a sealed selection freeze exists.
"""
import hashlib
import json
from pathlib import Path

from pilot_eval import attribution_units as au
from pilot_eval.scoring import score_gsm8k_flexible_v3

SPLITS = {'calibration': 'visible', 'discovery': 'attribution-only', 'heldout': 'blind-until-freeze'}
TRACES_DIR = 'attribution/traces'


class TraceRejected(ValueError):
    pass


class MinimumUnmet(RuntimeError):
    pass


# --- stop IDs ----------------------------------------------------------------------------------

def seal_stop_ids(decoder, token_text):
    """The stop IDs passed to the decoder (``eos_token_id``), with their strings and the pad ID."""
    eos = decoder.get('eos_token_id')
    ids = [eos] if isinstance(eos, int) else list(eos or [])
    if not ids or any(not isinstance(i, int) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('decoder eos_token_id must be an int or a nonempty list of distinct ints')
    return dict(stop_ids=ids, strings={str(i): token_text(i) for i in ids}, pad_token_id=decoder.get('pad_token_id'),
                source='decoder eos_token_id as passed to generate')


def metric_stop_ids(sealed, requested=None, *, decoding_change=None):
    if requested is None or list(requested) == sealed['stop_ids']:
        return list(sealed['stop_ids'])
    if not decoding_change:
        raise ValueError(f"stop set {list(requested)} differs from the decoder's {sealed['stop_ids']}; "
                         'refused unless labelled a decoding change')
    return list(requested)


# --- masks -------------------------------------------------------------------------------------

def content_offsets(content_ids, decode):
    """Character end offset of each content token by incremental decoding; refuse non-prefix decodes."""
    ends, previous = [], ''
    for k in range(len(content_ids)):
        text = decode(content_ids[:k + 1])
        if not text.startswith(previous):
            raise TraceRejected('nonmonotonic decoding: a prefix decode is not a prefix of the next')
        ends.append(len(text))
        previous = text
    return ends, previous


def build_masks(*, prompt_length, content_ids, decode):
    if not content_ids:
        raise TraceRejected('empty-content')
    ends, text = content_offsets(content_ids, decode)
    stripped = text.rstrip()
    last_line_start = stripped.rfind('\n') + 1  # 0 for a single-line response
    stop = [prompt_length + k for k, end in enumerate(ends) if end <= last_line_start]
    return dict(trace_targets=[prompt_length + k for k in range(len(content_ids))], stop_targets=stop,
                length=prompt_length + len(content_ids) + 1, last_line_start_char=last_line_start,
                stop_empty_reason=None if stop else 'single-line-response')


# --- prompts and generation --------------------------------------------------------------------

def prompt_rows(sequences, split):
    if split not in SPLITS:
        raise ValueError(f'unknown sequences-v2 split {split!r} (use {sorted(SPLITS)})')
    out = []
    for row in sequences:
        if row.get('group') != 'math' or row.get('split') != split:
            continue
        header = row['masks']['reply_start']
        if len(header) != 1:
            raise ValueError(f"{row['id']}: expected one reply-start position")
        out.append(dict(id=row['id'], split=split, source_ref=row['source_ref'],
                        prompt_ids=list(row['input_ids'][:header[0] + 1])))
    return out


def generate(generator, rows, batches):
    """Generate in the frozen manifest's batch order; every row exactly once."""
    by_id = {r['id']: r for r in rows}
    flat = [i for b in batches for i in b]
    if len(flat) != len(set(flat)) or set(flat) != set(by_id):
        raise ValueError('batch manifest must cover every row exactly once')
    out = []
    for index, batch in enumerate(batches):
        prompts = [by_id[i]['prompt_ids'] for i in batch]
        results = generator.generate(prompts)
        if len(results) != len(batch):
            raise ValueError('generator returned a different number of responses')
        for id_, prompt, result in zip(batch, prompts, results):
            out.append(dict(id=id_, batch=index, prompt_ids=list(prompt), generated_ids=list(result['generated_ids']),
                            stop_reason=result['stop_reason'], text=result['text']))
    return out


class HFGenerator:
    """Greedy FP32 generation with left padding and the decoder's EOS; returns generated IDs incl. EOS."""

    def __init__(self, model, *, pad_token_id, eos_token_id, max_new_tokens, decode):
        import torch
        self.torch, self.model, self.decode = torch, model, decode
        self.pad, self.eos, self.max_new_tokens = pad_token_id, eos_token_id, max_new_tokens
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('FP32 weights required')

    def generate(self, prompts):
        torch = self.torch
        width = max(map(len, prompts))
        device = next(self.model.parameters()).device
        ids = torch.tensor([[self.pad] * (width - len(p)) + list(p) for p in prompts], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
        self.model.eval()
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            generated = self.model.generate(input_ids=ids, attention_mask=mask, do_sample=False, num_beams=1,
                                            repetition_penalty=1.0, no_repeat_ngram_size=0,
                                            max_new_tokens=self.max_new_tokens, pad_token_id=self.pad,
                                            eos_token_id=self.eos, use_cache=True)
        out = []
        for continuation in generated[:, width:].tolist():
            end = next((i for i, t in enumerate(continuation) if t == self.eos), None)
            raw = continuation if end is None else continuation[:end + 1]
            content = raw if end is None else raw[:-1]
            out.append(dict(generated_ids=raw, stop_reason='cap' if end is None else 'eos',
                            text=self.decode(content)))
        return out


# --- acceptance ---------------------------------------------------------------------------------

def accept(responses, gold, *, decode, eos_id, split):
    rows, records = [], []
    for r in responses:
        generated = r['generated_ids']
        record = dict(id=r['id'], split=split, question_tokens=len(r['prompt_ids']), response_tokens=len(generated),
                      stop_reason=r['stop_reason'], reason=None, stop_empty=None, score=None)
        content = generated[:-1]
        if r['stop_reason'] != 'eos':
            record['reason'] = 'capped'
        elif not generated or generated[-1] != eos_id or eos_id in content:
            record['reason'] = 'eos-not-final'
        elif r['id'] not in gold:
            record['reason'] = 'gold-missing'
        else:
            score = score_gsm8k_flexible_v3(r['text'], gold[r['id']], capped=False)
            record['score'] = dict(status=score['status'], extracted=score.get('extracted'), correct=score['correct'])
            if score['status'] != 'valid':
                record['reason'] = 'invalid'
            elif not score['correct']:
                record['reason'] = 'incorrect'
            else:
                try:
                    masks = build_masks(prompt_length=len(r['prompt_ids']), content_ids=content, decode=decode)
                except TraceRejected as exc:
                    record['reason'] = str(exc)
                else:
                    record['stop_empty'] = not masks['stop_targets']
                    rows.append(dict(id=r['id'], split=split, input_ids=list(r['prompt_ids']) + list(generated),
                                     prompt_length=len(r['prompt_ids']), masks=masks))
        records.append(record)
    return rows, records


def counts(records):
    rejected = {}
    for r in records:
        if r['reason'] is not None:
            rejected[r['reason']] = rejected.get(r['reason'], 0) + 1
    accepted = [r for r in records if r['reason'] is None]
    return dict(generated=len(records), accepted=len(accepted),
                after_mask_exclusions=sum(1 for r in accepted if not r['stop_empty']),
                rejected=dict(sorted(rejected.items())), stop_mask_empty=sum(1 for r in accepted if r['stop_empty']))


def check_minimum(split, split_counts, *, minimum):
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        raise ValueError('the minimum must be a fixed integer from the launch record')
    if split_counts['after_mask_exclusions'] < minimum:
        raise MinimumUnmet(f"{split}: {split_counts['after_mask_exclusions']} rows after mask exclusions "
                           f'< minimum {minimum}; stop and decide')


# --- sources, sealing, reading -----------------------------------------------------------------

def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(inventory_dir, sequences_dir, artifact_root, *, inventory_verify=None, sequences_verify=None):
    """Refuse unless inventory-v2 and sequences-v2 verify; record their seals and per-arm provenance."""
    if inventory_verify is None:
        from pilot_eval.neuron_audit_inventory import verify as inventory_verify
    if sequences_verify is None:
        from pilot_eval.neuron_audit_sequences import verify as sequences_verify
    inventory = inventory_verify(inventory_dir, artifact_root)
    if not inventory['ok']:
        raise ValueError(f"inventory-v2 failed verification: {inventory['problems']}")
    sequences = sequences_verify(sequences_dir)
    if not sequences['ok']:
        raise ValueError(f"sequences-v2 failed verification: {sequences['problems']}")
    leakage = json.loads((Path(sequences_dir) / 'leakage_report.json').read_text())
    return dict(inventory_dir=str(inventory_dir), sequences_dir=str(sequences_dir),
                inventory_seal_sha256=_sha_file(Path(inventory_dir) / 'complete.json'),
                sequences_seal_sha256=_sha_file(Path(sequences_dir) / 'complete.json'),
                provenance=leakage['arms'], provenance_statement='no overlap with available sources',
                leakage_cleanliness=leakage.get('cleanliness'))


def load_gold(sequences_dir, artifact_root, rows):
    """Gold answers for math rows from the GSM8K train parquet pinned (by hash) in the sequences manifest."""
    from pilot_eval.neuron_audit_sequences import _inside, _read_parquet
    manifest = json.loads((Path(sequences_dir) / 'manifest.json').read_text())
    path = _inside(Path(artifact_root).resolve(), manifest['sources']['gsm8k']['train_parquet'])
    if _sha_file(path) != manifest['inputs']['files']['gsm8k_train']:
        raise ValueError('GSM8K train parquet hash differs from the sequences-v2 manifest')
    table = _read_parquet(path)
    return {r['id']: table[int(r['source_ref'])]['answer'] for r in rows}


def _jsonl(rows):
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def split_dir(root, split):
    if split not in SPLITS:
        raise ValueError(f'unknown split {split!r}')
    return Path(root) / TRACES_DIR / split


def seal_responses(root, split, responses, *, config, sources, stop):
    """All raw responses and the run config, sealed before scoring or masking."""
    directory = split_dir(root, split) / 'generation'
    au.seal_run(directory, {'responses.jsonl': _jsonl(responses),
                            'config.json': dict(config, split=split, stop=stop, sources=sources,
                                                visibility=SPLITS[split], scorer='gsm8k_flexible_v3')})
    return directory


def seal_rows(root, split, *, gold, decode, minimum, responses_dir=None):
    """Score, mask and seal rows + records + manifest; then enforce the minimum (records stay sealed)."""
    responses_dir = Path(responses_dir) if responses_dir is not None else split_dir(root, split) / 'generation'
    au.verify_run(responses_dir)
    config = json.loads((responses_dir / 'config.json').read_text())
    responses = _read_jsonl(responses_dir / 'responses.jsonl')
    eos = config['stop']['stop_ids']
    if len(eos) != 1:
        raise ValueError('trace acceptance expects the single pilot stop ID')
    rows, records = accept(responses, gold, decode=decode, eos_id=eos[0], split=split)
    split_counts = counts(records)
    manifest = dict(split=split, visibility=SPLITS[split], counts=split_counts, minimum=minimum,
                    stop_ids=eos, stop=config['stop'], provenance=config['sources']['provenance'],
                    provenance_statement=config['sources']['provenance_statement'],
                    sources={k: v for k, v in config['sources'].items() if k != 'provenance'},
                    responses_sha256=_sha_file(responses_dir / 'responses.jsonl'),
                    scope='questions the untuned model solves under flexible v3 and pilot decoding, '
                          'read as its own generated tokens',
                    mask_semantics='target positions; logits read at t-1 (attribution_metrics)')
    directory = split_dir(root, split) / 'rows'
    au.seal_run(directory, {'rows.jsonl': _jsonl(rows), 'records.jsonl': _jsonl(records), 'manifest.json': manifest})
    check_minimum(split, split_counts, minimum=minimum)
    return dict(directory=directory, manifest=manifest, rows=len(rows))


def load_rows(root, split, *, freeze_dir=None):
    """Sealed teacher-forcing rows; confirmation (heldout) only after a sealed selection freeze."""
    if SPLITS.get(split) == 'blind-until-freeze':
        if freeze_dir is None:
            raise PermissionError('confirmation rows are blind until the selection and analysis freeze')
        au.verify_run(freeze_dir)
    directory = split_dir(root, split) / 'rows'
    au.verify_run(directory)
    return _read_jsonl(directory / 'rows.jsonl')
