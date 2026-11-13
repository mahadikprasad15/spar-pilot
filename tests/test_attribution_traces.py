"""Ticket 01: base-correct trace preparation (stop IDs, masks, acceptance, sealing, visibility). CPU only."""
import hashlib
import json

import numpy as np
import pytest

from pilot_eval import attribution_traces as at
from pilot_eval import attribution_units as au

EOS, PAD = 151645, 151643
# Character-level fixture vocabulary for content tokens.
VOCAB = {10: 'Th', 11: 'e ', 12: 'sum', 13: '\n', 14: '#### ', 15: '42', 16: ' is', 17: '.\n', 18: 'The answer is 42.',
         19: '\n\n', 20: 'x', 21: 'y\nz', 22: '7', EOS: '<|im_end|>', PAD: '<|endoftext|>'}


def decode(ids):
    return ''.join(VOCAB[i] for i in ids if i not in (EOS, PAD))


def token_text(i):
    return VOCAB[i]


def test_stop_ids_sealed_from_the_decoder_config_single_id():
    sealed = at.seal_stop_ids(dict(eos_token_id=EOS, pad_token_id=PAD), token_text)
    assert sealed['stop_ids'] == [EOS] and sealed['strings'] == {str(EOS): '<|im_end|>'}
    assert sealed['pad_token_id'] == PAD and PAD not in sealed['stop_ids']
    assert at.metric_stop_ids(sealed) == [EOS]
    assert at.metric_stop_ids(sealed, [EOS]) == [EOS]


def test_mismatched_stop_set_refused_unless_labelled_a_decoding_change():
    sealed = at.seal_stop_ids(dict(eos_token_id=EOS, pad_token_id=PAD), token_text)
    with pytest.raises(ValueError, match='decoder'):
        at.metric_stop_ids(sealed, [EOS, PAD])
    labelled = at.metric_stop_ids(sealed, [EOS, PAD], decoding_change='two-ID stop set, qualified separately')
    assert labelled == [EOS, PAD]
    with pytest.raises(ValueError):
        at.seal_stop_ids(dict(eos_token_id=None, pad_token_id=PAD), token_text)


def masks(content):
    return at.build_masks(prompt_length=4, content_ids=content, decode=decode)


def test_masks_multi_line_with_hash_answer():
    # 'The sum\n#### 42' -> last non-empty line starts after the '\n' token
    m = masks([10, 11, 12, 13, 14, 15])
    assert m['trace_targets'] == [4, 5, 6, 7, 8, 9]  # all content incl. first; EOS excluded
    assert m['stop_targets'] == [4, 5, 6, 7]  # tokens ending at or before the last line's start
    assert m['length'] == 4 + 6 + 1


def test_masks_single_line_has_empty_stop_mask():
    m = masks([18])
    assert m['trace_targets'] == [4] and m['stop_targets'] == [] and m['stop_empty_reason'] == 'single-line-response'


def test_masks_prose_answer_and_trailing_blank_lines():
    # 'The sum is 42.\n7\n\n' : last non-empty line is '7'
    m = masks([10, 11, 12, 16, 15, 17, 22, 19])
    assert m['stop_targets'] == [4 + k for k in range(6)]


def test_token_crossing_the_newline_belongs_to_the_last_line():
    # 'x' + 'y\nz' : the boundary-crossing token is in the last line
    m = masks([20, 21])
    assert m['stop_targets'] == [4]


def test_empty_content_rejected():
    with pytest.raises(at.TraceRejected):
        masks([])


def test_nonmonotonic_decoding_rejected():
    def odd(ids):
        return 'ab' if len(ids) == 1 else 'xyz'
    with pytest.raises(at.TraceRejected, match='decod'):
        at.build_masks(prompt_length=1, content_ids=[1, 2], decode=odd)


def response(id_, content, stop='eos'):
    return dict(id=id_, prompt_ids=[1, 2, 3, 4], generated_ids=content + ([EOS] if stop == 'eos' else []),
                stop_reason=stop, text=decode(content))


def test_acceptance_keeps_correct_uncapped_and_preserves_generated_ids():
    gold = {'a': 'work\n#### 42', 'b': 'work\n#### 41', 'c': 'work\n#### 42', 'd': 'w\n#### 42'}
    rs = [response('a', [10, 11, 12, 13, 14, 15]), response('b', [10, 11, 12, 13, 14, 15]),
          response('c', [10, 11, 12, 13, 14, 15], stop='cap'), response('d', [18])]
    rows, records = at.accept(rs, gold, decode=decode, eos_id=EOS, split='discovery')
    assert [r['id'] for r in rows] == ['a', 'd']
    assert rows[0]['input_ids'] == [1, 2, 3, 4, 10, 11, 12, 13, 14, 15, EOS]  # prompt + generated, no re-tokenizing
    reasons = {r['id']: r['reason'] for r in records}
    assert reasons == {'a': None, 'b': 'incorrect', 'c': 'capped', 'd': None}
    assert all('question_tokens' in r and 'response_tokens' in r for r in records)


def test_eos_must_end_an_accepted_trace():
    bad = dict(response('a', [10, 15]), generated_ids=[10, EOS, 15])
    rows, records = at.accept([bad], {'a': '#### 42'}, decode=decode, eos_id=EOS, split='discovery')
    assert rows == [] and records[0]['reason'] == 'eos-not-final'


def test_minimum_checked_after_mask_exclusions():
    counts = at.counts([dict(id='a', reason=None, stop_empty=False), dict(id='b', reason=None, stop_empty=True),
                        dict(id='c', reason='incorrect', stop_empty=None)])
    assert counts == dict(generated=3, accepted=2, after_mask_exclusions=1, rejected={'incorrect': 1},
                          stop_mask_empty=1)
    at.check_minimum('discovery', counts, minimum=1)
    with pytest.raises(at.MinimumUnmet):
        at.check_minimum('discovery', counts, minimum=2)
    with pytest.raises(ValueError):  # the minimum comes from the launch record, never a default
        at.check_minimum('discovery', counts, minimum=None)


def test_prompt_rows_are_the_frozen_prefix_through_the_header():
    seq = [dict(id='gsm8k:train:5', group='math', split='discovery', source_ref=5,
                input_ids=[1, 2, 3, 4, 50, 51, EOS, 13], masks=dict(reply_start=[3], response_content=[4, 5])),
           dict(id='ultrachat:x', group='general', split='discovery', input_ids=[1], masks=dict(reply_start=[0])),
           dict(id='gsm8k:train:6', group='math', split='heldout', source_ref=6, input_ids=[1, 2, 9],
                masks=dict(reply_start=[1], response_content=[2]))]
    rows = at.prompt_rows(seq, 'discovery')
    assert rows == [dict(id='gsm8k:train:5', split='discovery', source_ref=5, prompt_ids=[1, 2, 3, 4])]
    with pytest.raises(ValueError):
        at.prompt_rows(seq, 'confirmation')  # sequences-v2 names the split 'heldout'


def test_batches_must_cover_rows_exactly_once_and_order_is_reproduced():
    rows = [dict(id=i, prompt_ids=[1] * n) for i, n in [('a', 3), ('b', 1), ('c', 2)]]
    seen = []

    class Gen:
        def generate(self, prompts):
            seen.append([len(p) for p in prompts])
            return [dict(generated_ids=[EOS], stop_reason='eos', text='') for _ in prompts]

    out = at.generate(Gen(), rows, [['c', 'a'], ['b']])
    assert seen == [[2, 3], [1]] and [o['id'] for o in out] == ['c', 'a', 'b'] and out[0]['batch'] == 0
    for bad in ([['a', 'b']], [['a', 'b', 'c', 'a']], [['a', 'b', 'c', 'z']]):
        with pytest.raises(ValueError):
            at.generate(Gen(), rows, bad)


def fake_sources(tmp_path):
    inv, seq = tmp_path / 'inv', tmp_path / 'seq'
    for d in (inv, seq):
        d.mkdir()
        (d / 'complete.json').write_text(json.dumps(dict(files={})))
    (seq / 'leakage_report.json').write_text(json.dumps(dict(
        arms={'online-grpo': dict(status='partial-some-sources-missing', missing_sources=['plan.yaml']),
              'gold-sft': dict(status='available', missing_sources=[])},
        cleanliness='no-overlap-with-available-sources')))
    return inv, seq


def test_sources_must_verify_and_are_recorded(tmp_path):
    inv, seq = fake_sources(tmp_path)
    ok = lambda *a: dict(ok=True, problems=[])  # noqa: E731
    record = at.verify_sources(inv, seq, tmp_path, inventory_verify=ok, sequences_verify=ok)
    assert record['inventory_seal_sha256'] == hashlib.sha256((inv / 'complete.json').read_bytes()).hexdigest()
    assert record['provenance']['online-grpo']['status'].startswith('partial')
    assert record['provenance_statement'] == 'no overlap with available sources'
    bad = lambda *a: dict(ok=False, problems=['sequences.jsonl integrity'])  # noqa: E731
    with pytest.raises(ValueError, match='sequences'):
        at.verify_sources(inv, seq, tmp_path, inventory_verify=ok, sequences_verify=bad)
    with pytest.raises(ValueError, match='inventory'):
        at.verify_sources(inv, seq, tmp_path, inventory_verify=bad, sequences_verify=ok)


def test_seal_split_writes_raw_responses_first_then_rows_and_manifest(tmp_path):
    inv, seq = fake_sources(tmp_path)
    ok = lambda *a: dict(ok=True, problems=[])  # noqa: E731
    sources = at.verify_sources(inv, seq, tmp_path, inventory_verify=ok, sequences_verify=ok)
    stop = at.seal_stop_ids(dict(eos_token_id=EOS, pad_token_id=PAD), token_text)
    rs = [response('a', [10, 11, 12, 13, 14, 15]), response('b', [18])]
    config = dict(model='qwen', adapter=None, decoding=dict(mode='greedy', max_new_tokens=1024))
    gen_dir = at.seal_responses(tmp_path / 'root', 'discovery', rs, config=config, sources=sources, stop=stop)
    assert (gen_dir / 'responses.jsonl').read_text().count('\n') == 2 and au.verify_run(gen_dir)
    result = at.seal_rows(tmp_path / 'root', 'discovery', gold={'a': '#### 42', 'b': '#### 42'}, decode=decode,
                          minimum=1)
    manifest = result['manifest']
    assert manifest['counts']['after_mask_exclusions'] == 1 and manifest['visibility'] == 'attribution-only'
    assert manifest['stop_ids'] == [EOS] and manifest['provenance']['gold-sft']['status'] == 'available'
    assert manifest['responses_sha256'] == hashlib.sha256((gen_dir / 'responses.jsonl').read_bytes()).hexdigest()
    with pytest.raises(at.MinimumUnmet):
        at.seal_rows(tmp_path / 'root2', 'discovery', gold={}, decode=decode, minimum=1,
                     responses_dir=gen_dir)


def test_confirmation_rows_refused_before_the_freeze(tmp_path):
    inv, seq = fake_sources(tmp_path)
    ok = lambda *a: dict(ok=True, problems=[])  # noqa: E731
    sources = at.verify_sources(inv, seq, tmp_path, inventory_verify=ok, sequences_verify=ok)
    stop = at.seal_stop_ids(dict(eos_token_id=EOS, pad_token_id=PAD), token_text)
    root = tmp_path / 'root'
    at.seal_responses(root, 'heldout', [response('a', [10, 11, 12, 13, 14, 15])], config={}, sources=sources, stop=stop)
    at.seal_rows(root, 'heldout', gold={'a': '#### 42'}, decode=decode, minimum=1)
    with pytest.raises(PermissionError):
        at.load_rows(root, 'heldout')
    with pytest.raises(ValueError):
        at.load_rows(root, 'heldout', freeze_dir=root / 'attribution/freeze')  # not sealed
    au.seal_run(root / 'attribution/freeze', {'freeze.json': dict(frozen=True)})
    assert [r['id'] for r in at.load_rows(root, 'heldout', freeze_dir=root / 'attribution/freeze')] == ['a']
    assert at.load_rows(root, 'heldout', freeze_dir=root / 'attribution/freeze')[0]['masks']['trace_targets']


def test_hf_generator_preserves_generated_ids_left_padded(tmp_path):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                         attn_implementation='eager')).eval()
    gen = at.HFGenerator(model, pad_token_id=0, eos_token_id=31, max_new_tokens=5, decode=lambda ids: str(ids))
    out = gen.generate([[3, 4, 5], [6]])
    single = gen.generate([[6]])
    assert all(o['stop_reason'] in ('eos', 'cap') for o in out)
    assert out[1]['generated_ids'] == single[0]['generated_ids']  # left padding: batch-invariant on this fixture
    for o in out:
        if o['stop_reason'] == 'cap':
            assert len(o['generated_ids']) == 5 and 31 not in o['generated_ids']
        else:
            assert o['generated_ids'][-1] == 31


def test_gold_answers_read_from_the_pinned_parquet_checked_by_hash(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    root, seq = tmp_path / 'art', tmp_path / 'seq'
    (root / 'data').mkdir(parents=True)
    seq.mkdir()
    path = root / 'data/train.parquet'
    pq.write_table(pa.table(dict(question=['q0', 'q1'], answer=['a\n#### 1', 'b\n#### 2'])), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(inputs=dict(files=dict(gsm8k_train=digest)), sources=dict(gsm8k=dict(train_parquet='data/train.parquet')))
    (seq / 'manifest.json').write_text(json.dumps(manifest))
    rows = [dict(id='gsm8k:train:1', source_ref=1)]
    assert at.load_gold(seq, root, rows) == {'gsm8k:train:1': 'b\n#### 2'}
    manifest['inputs']['files']['gsm8k_train'] = '0' * 64
    (seq / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash'):
        at.load_gold(seq, root, rows)
