"""C-attribution rows from saved GRPO behaviour records (ticket 02).

Reads a behaviour ``responses.jsonl`` and its run ``config.json``, selects the step-64 sampled rows,
checks them against the bound provenance, and builds instrument rows plus a manifest.

Two steps, so the provenance checks run without a tokenizer:

    loaded = load(source)                       # hashes, provenance, refusals, counts (CPU, no model)
    rows, manifest = instrument_rows(loaded, encode)
    seal(out_dir, rows, manifest)

``source`` binds every input (normally taken from the launch record)::

    dict(arm='online-grpo', catalog_id='online-grpo.s64.sampled-x8.test150',
         responses=dict(path=..., sha256=...), config=dict(path=..., sha256=...),
         adapter_sha256=..., select=dict(step=64, mode='sampled'))

Nothing is filtered: wrong, invalid, capped, all-correct and all-wrong rows are all kept. Any failed
check raises ``RowsError``; there is no partial result.
"""
import collections
import hashlib
import json
from pathlib import Path

from pilot_eval.scoring import score_gsm8k_flexible_v3
from pilot_eval.workflow import _hash

SCHEMA = 'c-attribution-rows-v1'
REPO = Path(__file__).resolve().parents[1]
EOS = 151645
SCORER = 'gsm8k-flexible-v3'
SCORER_SHA256 = 'c958d2de1a6212b49fcfa2f7ac1fd215117ca8d3dbfe3ace805576ce6937eb6c'
SELECT = {'step': 64, 'mode': 'sampled'}
# The sampled policy both GRPO behaviour runs record for step 64 (spec: Data and provenance).
POLICY = {'batch_size': 4, 'completion_limit': 1024, 'do_sample': True, 'draws': 8, 'mode': 'sampled',
          'num_beams': 1, 'step': 64, 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'use_cache': True}
# Not in the run configs; grpo_gpu._generate passes them explicitly. grpo_gpu.py is not hashed by the runs.
INFERRED = {
    'repetition_penalty': 1.0,
    'no_repeat_ngram_size': 0,
    'evidence': ('grpo_gpu._generate passes repetition_penalty=1.0 and no_repeat_ngram_size=0 explicitly; '
                 'the run configs do not hash grpo_gpu.py; git shows its last change 3c859a6 on 2026-10-09, '
                 'before both behaviour runs'),
}
UNVERIFIED = {'resolved_logits_processors': 'not verified until ticket 05 passes'}


class RowsError(ValueError):
    pass


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def _bound(ref, name):
    try:
        path, expected = Path(ref['path']), ref['sha256']
    except (KeyError, TypeError) as exc:
        raise RowsError(f'{name}: path and sha256 are required') from exc
    actual = file_sha256(path)
    if actual != expected:
        raise RowsError(f'{name}: sha256 {actual} != bound {expected}')
    return path


def _provenance(config, select, policy, adapter, repo, eos=EOS):
    implementation = config.get('implementation') or {}
    engine = file_sha256(repo / 'pilot_eval' / 'grpo_eval_engine.py')
    if implementation.get('engine') != engine:
        raise RowsError(f"implementation.engine {implementation.get('engine')} != sha256(grpo_eval_engine.py) {engine}")
    scorer = file_sha256(repo / 'pilot_eval' / 'scoring.py')
    if scorer != SCORER_SHA256 or implementation.get('scorer') != SCORER_SHA256:
        raise RowsError(f"scorer hash: code {scorer}, run {implementation.get('scorer')}, required {SCORER_SHA256}")
    if config.get('scorer') != SCORER:
        raise RowsError(f"run scorer {config.get('scorer')} != {SCORER}")
    matches = [p for p in config.get('policies', []) if p.get('mode') == select['mode'] and p.get('step') == select['step']]
    if len(matches) != 1:
        raise RowsError(f'expected one {select} policy in the run config, found {len(matches)}')
    if matches[0] != policy:
        diff = sorted(k for k in set(policy) | set(matches[0]) if policy.get(k) != matches[0].get(k))
        raise RowsError(f'sampled policy differs from the bound policy in {diff}')
    checkpoint = (config.get('checkpoints') or {}).get(str(select['step'])) or {}
    if checkpoint.get('adapter_sha256') != adapter:
        raise RowsError(f"checkpoint adapter {checkpoint.get('adapter_sha256')} != bound {adapter}")
    template = config.get('prompt_template') or {}
    runtime = config.get('runtime') or {}
    if runtime.get('eos_token_id') != eos:
        raise RowsError(f"runtime eos_token_id {runtime.get('eos_token_id')} != {eos}")
    return {
        'recorded': {
            'model': config.get('model'),
            'model_revision': config.get('model_revision'),
            'tokenizer_revision': config.get('tokenizer_revision'),
            'dataset_revision': template.get('dataset_revision'),
            'chat_template_sha256': template.get('chat_template_sha256'),
            'prompt_template': template.get('prompt_template'),
            'policy': dict(matches[0]),
            'eos_token_id': eos,
            'seed_rule': config.get('seed_rule'),
            'runtime': {k: runtime.get(k) for k in ('versions', 'device', 'dtype', 'adapter_dtype', 'tf32',
                                                    'deterministic_algorithms', 'attention_implementation',
                                                    'generation_use_cache', 'cublas_workspace_config')},
            'implementation.engine': engine,
            'implementation.scorer': SCORER_SHA256,
            'checkpoint': dict(checkpoint, step=select['step']),
        },
        'inferred': dict(INFERRED, grpo_gpu_sha256_now=file_sha256(repo / 'pilot_eval' / 'grpo_gpu.py')),
        'unverified': dict(UNVERIFIED),
    }


def _check_row(row, adapter, units, items, *, cap=POLICY['completion_limit'], eos=EOS):
    rid = row['draw_id']
    if row.get('adapter_sha256') != adapter:
        raise RowsError(f"{rid}: adapter_sha256 {row.get('adapter_sha256')} != {adapter}")
    tokens = row.get('generated_token_ids')
    if not isinstance(tokens, list) or not tokens:
        raise RowsError(f'{rid}: empty generated sequence')
    capped, stop = row.get('capped'), row.get('stop_reason')
    if capped is True:
        if stop != 'cap' or len(tokens) != cap or eos in tokens:
            raise RowsError(f'{rid}: a capped row needs stop_reason cap, exactly {cap} tokens and no EOS')
    elif capped is False:
        if stop != 'eos' or tokens[-1] != eos or eos in tokens[:-1] or len(tokens) > cap:
            raise RowsError(f'{rid}: an EOS row must end with its only {eos} within {cap} tokens')
    else:
        raise RowsError(f'{rid}: capped must be a boolean')
    if row.get('scorer') != SCORER:
        raise RowsError(f"{rid}: scorer {row.get('scorer')} != {SCORER}")
    score = score_gsm8k_flexible_v3(row['text'], row['gold'], capped=capped)
    reward = 0 if capped else int(score['correct'])
    if row.get('reward') != reward:
        raise RowsError(f"{rid}: stored reward {row.get('reward')} != re-scored {reward}")
    item = items.get(row['group_id'])
    if item is None or item['prompt'] != row['prompt'] or item['gold'] != row['gold']:
        raise RowsError(f'{rid}: prompt or gold differs from the run config item')
    unit = units.get(row.get('unit_key'))
    if unit is None or row['group_id'] not in unit['ids'] or unit['seed'] != row.get('seed'):
        raise RowsError(f"{rid}: unit {row.get('unit_key')} does not hold this question with seed {row.get('seed')}")
    return score['status'] != 'valid'


def load(source, *, repo=REPO):
    """Validate one arm's sampled rows against its bound inputs; return questions and the manifest."""
    select = dict(source.get('select') or SELECT)
    policy = dict(source.get('policy') or POLICY)
    adapter = source['adapter_sha256']
    eos = int(source.get('eos_token_id', EOS))  # the real runs: 151645; the CPU chain world: its own EOS
    responses = _bound(source['responses'], 'responses')
    config_path = _bound(source['config'], 'config')
    config = json.loads(config_path.read_text())
    provenance = _provenance(config, select, policy, adapter, Path(repo), eos)
    units = {u['key']: u for u in config.get('units', []) if u.get('policy') == policy}
    items = {i['id']: i for i in config.get('items', [])}
    order = [i['id'] for i in config.get('items', [])]

    groups, seen, total = collections.defaultdict(list), set(), 0
    with open(responses) as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            if any(row.get(k) != v for k, v in select.items()):
                continue
            if row['draw_id'] in seen:
                raise RowsError(f"duplicate draw_id {row['draw_id']}")
            seen.add(row['draw_id'])
            row['_invalid'] = _check_row(row, adapter, units, items, cap=policy['completion_limit'], eos=eos)
            groups[row['group_id']].append(row)

    questions = []
    for qid in [q for q in order if q in groups] + sorted(set(groups) - set(order)):
        draws = sorted(groups[qid], key=lambda r: r['draw_index'])
        if len(draws) < 2:
            raise RowsError(f'{qid}: {len(draws)} draw(s); leave-one-out needs at least 2')
        if len(draws) != policy['draws'] or [r['draw_index'] for r in draws] != list(range(policy['draws'])):
            raise RowsError(f"{qid}: draw indices {[r['draw_index'] for r in draws]} != 0..{policy['draws'] - 1}")
        questions.append({'question_id': qid, 'draws': draws})
    if not questions:
        raise RowsError(f'no rows match {select}')

    rows = [r for q in questions for r in q['draws']]
    types = collections.Counter(
        'all-correct' if all(r['reward'] for r in q['draws']) else
        'all-wrong' if not any(r['reward'] for r in q['draws']) else 'mixed' for q in questions)
    manifest = {
        'schema': SCHEMA,
        'arm': source.get('arm'),
        'catalog_id': source.get('catalog_id'),
        'inputs': {
            'responses': {'path': str(source['responses']['path']), 'sha256': source['responses']['sha256'],
                          'rows_total': total, 'select': select},
            'config': {'path': str(source['config']['path']), 'sha256': source['config']['sha256']},
            'adapter_sha256': adapter,
        },
        'provenance': provenance,
        'counts': {
            'rows': len(rows),
            'questions': len(questions),
            'groups': {k: types.get(k, 0) for k in ('all-correct', 'all-wrong', 'mixed')},
            'capped': sum(r['capped'] for r in rows),
            'invalid': sum(r['_invalid'] for r in rows),
            'mean_reward': sum(r['reward'] for r in rows) / len(rows),
        },
        'rows': [{'id': r['draw_id'], 'group_id': r['group_id'], 'draw_index': r['draw_index'],
                  'reward': r['reward'], 'capped': r['capped'], 'invalid': r['_invalid'], 'seed': r['seed'],
                  'unit_key': r['unit_key'], 'record_id': r.get('record_id'),
                  'generated_tokens': len(r['generated_token_ids'])} for r in rows],
    }
    return {'questions': questions, 'items': items, 'manifest': manifest}


def seq_targets(prompt_length, total_length):
    """Target positions of every generated token in input_ids: prompt_length .. total_length - 1."""
    if prompt_length < 1 or total_length <= prompt_length:
        raise RowsError('a row needs a nonempty prompt and a nonempty generated sequence')
    return list(range(prompt_length, total_length))


def instrument_rows(loaded, encode, *, tokenizer_identity=None):
    """Build instrument rows, grouped by question (all draws of a question are adjacent).

    ``encode(prompt) -> list[int]`` must be the pinned tokenizer with ``add_special_tokens=False``.
    ``tokenizer_identity`` (``revision``, ``chat_template_sha256``), if given, must match the run.
    """
    manifest = json.loads(json.dumps(loaded['manifest']))
    recorded = manifest['provenance']['recorded']
    if tokenizer_identity is not None:
        expected = {'revision': recorded['tokenizer_revision'], 'chat_template_sha256': recorded['chat_template_sha256']}
        if dict(tokenizer_identity) != expected:
            raise RowsError(f'tokenizer identity {dict(tokenizer_identity)} != run {expected}')
    rows, cache = [], {}
    for question in loaded['questions']:
        qid = question['question_id']
        if qid not in cache:
            ids = [int(t) for t in encode(loaded['items'][qid]['prompt'])]
            want = loaded['items'][qid].get('prompt_tokens')
            if want is not None and len(ids) != want:
                raise RowsError(f'{qid}: prompt encodes to {len(ids)} tokens, the run recorded {want}')
            cache[qid] = ids
        prompt = cache[qid]
        for r in question['draws']:
            input_ids = prompt + [int(t) for t in r['generated_token_ids']]
            rows.append({'id': r['draw_id'], 'question_id': qid, 'draw_index': r['draw_index'],
                         'reward': r['reward'], 'input_ids': input_ids,
                         'masks': {'seq_targets': seq_targets(len(prompt), len(input_ids)), 'length': len(input_ids)}})
    lengths = [len(r['input_ids']) for r in rows]
    manifest['tokenization'] = {
        'add_special_tokens': False,
        'tokenizer_identity': dict(tokenizer_identity) if tokenizer_identity is not None else None,
        'prompt_tokens': {qid: len(ids) for qid, ids in cache.items()},
        'input_tokens_total': sum(lengths),
        'input_tokens_max': max(lengths),
    }
    manifest['rows_sha256'] = rows_sha256(rows)
    return rows, manifest


def rows_sha256(rows):
    return _hash([{k: r[k] for k in ('id', 'input_ids', 'masks')} for r in rows])


def seal(out_dir, rows, manifest):
    """Write rows.jsonl and manifest.json; an existing seal must be identical (resume), never overwritten."""
    out = Path(out_dir)
    if manifest.get('rows_sha256') != rows_sha256(rows):
        raise RowsError('manifest rows_sha256 does not match the rows')
    body = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
    sealed = dict(manifest, rows_file_sha256=hashlib.sha256(body.encode()).hexdigest())
    text = json.dumps(sealed, indent=2, sort_keys=True) + '\n'
    rows_path, manifest_path = out / 'rows.jsonl', out / 'manifest.json'
    if manifest_path.exists() or rows_path.exists():
        if not (manifest_path.exists() and rows_path.exists()) or manifest_path.read_text() != text \
                or file_sha256(rows_path) != sealed['rows_file_sha256']:
            raise RowsError(f'{out}: an existing, different row seal is present; refusing to overwrite')
        return sealed
    out.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(body)
    manifest_path.write_text(text)
    return sealed


def verify(out_dir):
    """Re-hash a sealed row set; return its manifest."""
    out = Path(out_dir)
    manifest = json.loads((out / 'manifest.json').read_text())
    if file_sha256(out / 'rows.jsonl') != manifest['rows_file_sha256']:
        raise RowsError(f'{out}: rows.jsonl does not match its seal')
    rows = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines()]
    if rows_sha256(rows) != manifest['rows_sha256']:
        raise RowsError(f'{out}: rows do not match rows_sha256')
    return manifest
