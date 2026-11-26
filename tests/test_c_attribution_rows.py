"""Ticket 02: rows from saved behaviour records."""
import copy
import json
import os
from pathlib import Path

import pytest

from pilot_eval import c_attribution_rows as cr

FIXTURES = Path(__file__).parent / 'fixtures' / 'c_attribution_rows'
ARTIFACTS = Path(os.environ.get('PILOT_ARTIFACTS', '/Users/prasadmahadik/spar/pilot1/artifacts'))
ADAPTERS = {'online': '70d2d56cd4636933020b107d6fb344ed451bc879b4a694e725c8608181f07226',
            'offline': '6601551812ab754ccf35c0fbd5f6cf181786636683e9367b24c4f92451a1c93f'}
# Expected counts for the two real groups copied per arm (see the fixture files).
FIXTURE_COUNTS = {
    'online': {'rows': 16, 'questions': 2, 'groups': {'all-correct': 0, 'all-wrong': 1, 'mixed': 1},
               'capped': 0, 'invalid': 1, 'mean_reward': 7 / 16},
    'offline': {'rows': 16, 'questions': 2, 'groups': {'all-correct': 0, 'all-wrong': 0, 'mixed': 2},
                'capped': 1, 'invalid': 3, 'mean_reward': 6 / 16},
}


def _source(directory, arm, *, responses='responses.jsonl'):
    directory = Path(directory)
    return {'arm': f'{arm}-grpo', 'catalog_id': f'{arm}-grpo.s64.sampled-x8.test150',
            'responses': {'path': directory / responses, 'sha256': cr.file_sha256(directory / responses)},
            'config': {'path': directory / 'config.json', 'sha256': cr.file_sha256(directory / 'config.json')},
            'adapter_sha256': ADAPTERS[arm], 'select': {'step': 64, 'mode': 'sampled'}}


def _copy(tmp_path, arm, *, rows=None, config=None):
    """Copy an arm's fixture, optionally mutating rows (list of dicts) or the config (dict)."""
    out = tmp_path / arm
    out.mkdir()
    lines = [json.loads(line) for line in (FIXTURES / arm / 'responses.jsonl').read_text().splitlines()]
    cfg = json.loads((FIXTURES / arm / 'config.json').read_text())
    if rows is not None:
        lines = rows(lines)
    if config is not None:
        cfg = config(cfg)
    (out / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in lines))
    (out / 'config.json').write_text(json.dumps(cfg))
    return _source(out, arm)


def _fake_encode(items):
    """Encode a prompt as the recorded number of distinct placeholder tokens."""
    by_prompt = {i['prompt']: i['prompt_tokens'] for i in items.values()}
    return lambda prompt: list(range(1000, 1000 + by_prompt[prompt]))


def _sampled(lines):
    return [r for r in lines if r['mode'] == 'sampled' and r['step'] == 64]


# --- real-format fixtures ---------------------------------------------------------------------

@pytest.mark.parametrize('arm', ['online', 'offline'])
def test_fixture_groups_reproduce_their_counts_and_keep_every_row(arm):
    loaded = cr.load(_source(FIXTURES / arm, arm))
    manifest = loaded['manifest']
    assert manifest['counts'] == FIXTURE_COUNTS[arm]
    assert manifest['inputs']['responses']['rows_total'] == 17  # one greedy row is not selected
    assert [len(q['draws']) for q in loaded['questions']] == [8, 8]
    for q in loaded['questions']:
        assert [r['draw_index'] for r in q['draws']] == list(range(8))
    fields = set(manifest['rows'][0])
    assert {'reward', 'group_id', 'draw_index', 'seed', 'unit_key'} <= fields


@pytest.mark.parametrize('arm', ['online', 'offline'])
def test_provenance_block_separates_recorded_inferred_and_unverified(arm):
    prov = cr.load(_source(FIXTURES / arm, arm))['manifest']['provenance']
    recorded = prov['recorded']
    assert recorded['policy'] == cr.POLICY
    assert recorded['model_revision'] == recorded['tokenizer_revision'] == '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    assert recorded['chat_template_sha256'].startswith('2e2d2512')
    assert recorded['implementation.engine'] == cr.file_sha256(cr.REPO / 'pilot_eval' / 'grpo_eval_engine.py')
    assert recorded['runtime']['versions']['torch'] == '2.9.0'
    assert recorded['checkpoint']['adapter_sha256'] == ADAPTERS[arm]
    assert prov['inferred']['repetition_penalty'] == 1.0 and prov['inferred']['no_repeat_ngram_size'] == 0
    assert '3c859a6' in prov['inferred']['evidence']
    assert 'resolved_logits_processors' in prov['unverified']


@pytest.mark.parametrize('arm', ['online', 'offline'])
def test_instrument_rows_are_grouped_by_question_with_generated_targets(arm):
    loaded = cr.load(_source(FIXTURES / arm, arm))
    rows, manifest = cr.instrument_rows(loaded, _fake_encode(loaded['items']))
    assert len(rows) == 16
    assert [r['question_id'] for r in rows] == [q['question_id'] for q in loaded['questions'] for _ in range(8)]
    for row, q_row in zip(rows, [r for q in loaded['questions'] for r in q['draws']]):
        prompt = loaded['items'][row['question_id']]['prompt_tokens']
        generated = q_row['generated_token_ids']
        assert row['input_ids'][prompt:] == generated
        assert row['masks']['seq_targets'] == list(range(prompt, prompt + len(generated)))
        assert row['masks']['length'] == len(row['input_ids'])
        assert row['input_ids'][row['masks']['seq_targets'][-1]] == generated[-1]
    assert manifest['rows_sha256'] == cr.rows_sha256(rows)
    if arm == 'offline':
        capped = [r for r, q in zip(rows, [x for q in loaded['questions'] for x in q['draws']]) if q['capped']]
        assert len(capped) == 1 and len(capped[0]['masks']['seq_targets']) == 1024
        assert cr.EOS not in capped[0]['input_ids'][-1024:]


# --- exact target masks -----------------------------------------------------------------------

def test_target_masks_for_eos_eos_only_and_capped_responses():
    prompt = 5
    eos = cr.seq_targets(prompt, prompt + 3)            # two content tokens then the sampled EOS
    assert eos == [5, 6, 7] and eos[-1] == 7
    eos_only = cr.seq_targets(prompt, prompt + 1)       # a response that is only the EOS
    assert eos_only == [5]
    capped = cr.seq_targets(prompt, prompt + 1024)
    assert len(capped) == 1024 and capped[0] == 5 and capped[-1] == 1028
    with pytest.raises(cr.RowsError):
        cr.seq_targets(0, 4)
    with pytest.raises(cr.RowsError):
        cr.seq_targets(4, 4)


def test_eos_only_row_builds_a_single_target(tmp_path):
    def mutate(lines):
        row = _sampled(lines)[0]
        row['generated_token_ids'] = [cr.EOS]
        row['text'] = ''
        row['reward'] = 0
        return lines
    loaded = cr.load(_copy(tmp_path, 'online', rows=mutate))
    rows, _ = cr.instrument_rows(loaded, _fake_encode(loaded['items']))
    first = next(r for r in rows if r['input_ids'][-1] == cr.EOS and len(r['masks']['seq_targets']) == 1)
    assert first['masks']['seq_targets'] == [len(first['input_ids']) - 1]


# --- refusals ---------------------------------------------------------------------------------

def _first(lines, predicate=lambda r: True):
    return next(r for r in _sampled(lines) if predicate(r))


ROW_MUTATIONS = {
    'duplicate draw_id': lambda ls: ls + [copy.deepcopy(_first(ls))],
    'adapter differs': lambda ls: [_first(ls).__setitem__('adapter_sha256', '0' * 64), ls][1],
    'EOS row not ending in EOS': lambda ls: [_first(ls).__setitem__('generated_token_ids',
                                                                     _first(ls)['generated_token_ids'][:-1] + [11]), ls][1],
    'EOS earlier in an EOS row': lambda ls: [_first(ls).__setitem__('generated_token_ids',
                                                                     [cr.EOS] + _first(ls)['generated_token_ids']), ls][1],
    'reward disagrees with re-score': lambda ls: [_first(ls).__setitem__('reward', 1 - _first(ls)['reward']), ls][1],
    'scorer name differs': lambda ls: [_first(ls).__setitem__('scorer', 'gsm8k-flexible-v2'), ls][1],
    'fewer than 2 draws': lambda ls: [r for r in ls if not (r['mode'] == 'sampled' and r['group_id'] == _first(ls)['group_id']
                                                          and r['draw_index'] > 0)],
    'missing draw': lambda ls: [r for r in ls if not (r['mode'] == 'sampled' and r['draw_index'] == 7
                                                      and r['group_id'] == _first(ls)['group_id'])],
    'seed differs from unit': lambda ls: [_first(ls).__setitem__('seed', 999), ls][1],
    'prompt differs from config': lambda ls: [_first(ls).__setitem__('prompt', 'x'), ls][1],
    'empty generation': lambda ls: [_first(ls).__setitem__('generated_token_ids', []), ls][1],
}


@pytest.mark.parametrize('name', sorted(ROW_MUTATIONS))
def test_row_refusals(tmp_path, name):
    with pytest.raises(cr.RowsError):
        cr.load(_copy(tmp_path, 'online', rows=ROW_MUTATIONS[name]))


def _capped(ls):
    return _first(ls, lambda r: r['capped'])


CAPPED_MUTATIONS = {
    'capped row with 1,023 tokens': lambda ls: [_capped(ls).__setitem__('generated_token_ids',
                                                                         _capped(ls)['generated_token_ids'][:-1]), ls][1],
    'capped row containing EOS': lambda ls: [_capped(ls)['generated_token_ids'].__setitem__(10, cr.EOS), ls][1],
    'capped row with eos stop reason': lambda ls: [_capped(ls).__setitem__('stop_reason', 'eos'), ls][1],
}


@pytest.mark.parametrize('name', sorted(CAPPED_MUTATIONS))
def test_capped_refusals(tmp_path, name):
    with pytest.raises(cr.RowsError):
        cr.load(_copy(tmp_path, 'offline', rows=CAPPED_MUTATIONS[name]))


def _policy(field, value):
    def mutate(cfg):
        for p in cfg['policies']:
            if p['mode'] == 'sampled' and p['step'] == 64:
                p[field] = value
        return cfg
    return mutate


CONFIG_MUTATIONS = {
    'engine hash': lambda c: [c['implementation'].__setitem__('engine', '0' * 64), c][1],
    'scorer hash': lambda c: [c['implementation'].__setitem__('scorer', '0' * 64), c][1],
    'checkpoint adapter': lambda c: [c['checkpoints']['64'].__setitem__('adapter_sha256', '0' * 64), c][1],
    'policy temperature': _policy('temperature', 0.7),
    'policy top_k': _policy('top_k', 50),
    'policy batch_size': _policy('batch_size', 8),
    'policy draws': _policy('draws', 4),
}


@pytest.mark.parametrize('name', sorted(CONFIG_MUTATIONS))
def test_provenance_refusals(tmp_path, name):
    with pytest.raises(cr.RowsError):
        cr.load(_copy(tmp_path, 'online', config=CONFIG_MUTATIONS[name]))


def test_bound_file_hash_mismatch_is_refused(tmp_path):
    source = _source(FIXTURES / 'online', 'online')
    source['responses']['sha256'] = '0' * 64
    with pytest.raises(cr.RowsError):
        cr.load(source)
    source = _source(FIXTURES / 'online', 'online')
    source['config']['sha256'] = '0' * 64
    with pytest.raises(cr.RowsError):
        cr.load(source)
    source = _source(FIXTURES / 'online', 'online')
    source['adapter_sha256'] = '0' * 64
    with pytest.raises(cr.RowsError):
        cr.load(source)


def test_scorer_code_hash_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, 'SCORER_SHA256', '0' * 64)
    with pytest.raises(cr.RowsError):
        cr.load(_source(FIXTURES / 'online', 'online'))


def test_prompt_token_count_and_tokenizer_identity_are_checked():
    loaded = cr.load(_source(FIXTURES / 'online', 'online'))
    with pytest.raises(cr.RowsError):
        cr.instrument_rows(loaded, lambda prompt: [1, 2, 3])
    with pytest.raises(cr.RowsError):
        cr.instrument_rows(loaded, _fake_encode(loaded['items']),
                           tokenizer_identity={'revision': 'other', 'chat_template_sha256': 'x'})


# --- sealing ----------------------------------------------------------------------------------

def test_seal_is_idempotent_and_refuses_a_different_existing_seal(tmp_path):
    loaded = cr.load(_source(FIXTURES / 'online', 'online'))
    rows, manifest = cr.instrument_rows(loaded, _fake_encode(loaded['items']))
    out = tmp_path / 'rows'
    sealed = cr.seal(out, rows, manifest)
    assert cr.seal(out, rows, manifest) == sealed
    assert cr.verify(out)['rows_sha256'] == manifest['rows_sha256']
    rows[0]['input_ids'][0] += 1
    with pytest.raises(cr.RowsError):
        cr.seal(out, rows, manifest)
    with pytest.raises(cr.RowsError):
        cr.seal(out, rows, dict(manifest, rows_sha256=cr.rows_sha256(rows)))
    (out / 'rows.jsonl').write_text((out / 'rows.jsonl').read_text().replace('1000', '1001', 1))
    with pytest.raises(cr.RowsError):
        cr.verify(out)


# --- the pinned tokenizer (local artifacts), else a cached Qwen2.5 tokenizer with the same template ------

def test_cached_qwen_tokenizer_reproduces_recorded_prompt_lengths():
    transformers = pytest.importorskip('transformers')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    tokenizer = None
    pinned = ARTIFACTS / 'inputs/models/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    for name in (str(pinned), 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'):
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
            break
        except Exception:
            continue
    if tokenizer is None:
        pytest.skip('no cached Qwen2.5 tokenizer')
    from pilot_eval.workflow import _hash
    identity = {'revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', 'chat_template_sha256': _hash(tokenizer.chat_template)}
    for arm in ('online', 'offline'):
        loaded = cr.load(_source(FIXTURES / arm, arm))
        rows, manifest = cr.instrument_rows(loaded, lambda p: tokenizer(p, add_special_tokens=False)['input_ids'],
                                            tokenizer_identity=identity)
        assert len(rows) == 16 and manifest['tokenization']['tokenizer_identity'] == identity


# --- the real local files ---------------------------------------------------------------------

REAL = {
    'online': ('imports/pilot4-behaviour-breakdown-2026-10-22/grpo', 'responses.jsonl',
               {'rows': 1200, 'questions': 150, 'groups': {'all-correct': 49, 'all-wrong': 12, 'mixed': 89},
                'capped': 0, 'invalid': 46, 'mean_reward': 0.675}),
    'offline': ('imports/offline-grpo-v1/runs/offline-grpo/offline-grpo-v1-behaviour/behaviour', 'results/responses.jsonl',
                {'rows': 1200, 'questions': 150, 'groups': {'all-correct': 57, 'all-wrong': 13, 'mixed': 80},
                 'capped': 1, 'invalid': 39, 'mean_reward': 832 / 1200}),
}
REAL_SHA = {'online': '4af7926a3def742d9eef128f3b7306b783aada413f785624054dacb16ee971ec',
            'offline': 'a1cd79aa24e9cf7a87d0d8db3bd15588b97e830229db662892038f4b7437be94'}


@pytest.mark.parametrize('arm', ['online', 'offline'])
def test_real_local_files_reproduce_the_data_table(arm):
    directory, responses, counts = REAL[arm]
    path = ARTIFACTS / directory
    if not (path / responses).exists():
        pytest.skip(f'real {arm} behaviour records are not local')
    source = _source(path, arm, responses=responses)
    assert source['responses']['sha256'] == REAL_SHA[arm]
    manifest = cr.load(source)['manifest']
    assert manifest['counts'] == counts
    assert manifest['inputs']['responses']['rows_total'] == 3150
