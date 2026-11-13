"""Ticket 02 (joint neuron/adapter audit): frozen sequences, masks and leakage on CPU fixtures."""
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pilot_eval import neuron_audit_sequences as seq

REAL_TOKENIZER = Path('/Users/prasadmahadik/spar/pilot1/artifacts/inputs/models/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
GOLD_ITEMS = Path('/Users/prasadmahadik/spar/pilot1/artifacts/plans/pilot2-sft-fp32-l4-batch8-v1/training.items.json')


class FakeTokenizer:
    """Qwen-shaped template with word-level tokens; ``merge_newline`` fuses '\\n' with a following capital."""
    eos_token = '<|im_end|>'
    eos_token_id = 2

    def __init__(self, merge_newline=False):
        self.merge_newline = merge_newline
        self.vocab = {'<|im_start|>': 1, '<|im_end|>': 2}
        self.pattern = re.compile((r'\n[A-Z]|' if merge_newline else '') + r'<\|im_start\|>|<\|im_end\|>|\n|[^\s<]+|\s+|<')

    def render(self, messages, add_generation_prompt):
        text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ('<|im_start|>assistant\n' if add_generation_prompt else '')

    def encode(self, text):
        ids, offsets = [], []
        for match in self.pattern.finditer(text):
            ids.append(self.vocab.setdefault(match.group(), len(self.vocab) + 1))
            offsets.append(match.span())
        return ids, offsets

    def token_text(self, token_id):
        return {v: k for k, v in self.vocab.items()}[token_id]

    def identity(self):
        return dict(kind='fake', merge_newline=self.merge_newline)


def msgs(prompt, response, *extra):
    out = [dict(role='user', content=prompt), dict(role='assistant', content=response)]
    for i, text in enumerate(extra):
        out.append(dict(role='user' if i % 2 == 0 else 'assistant', content=text))
    return out


# --- masks and boundaries -------------------------------------------------------------------

def test_masks_select_content_reply_start_and_termination():
    tok = FakeTokenizer()
    ex = seq.build_example(tok, 'x', msgs('Hi there', 'Fine thanks'), max_tokens=100, overlength='exclude')
    assert ex['status'] == 'ok'
    pieces = [tok.token_text(i) for i in ex['input_ids']]
    assert [pieces[i] for i in ex['masks']['response_content']] == ['Fine', ' ', 'thanks']
    assert pieces[ex['masks']['reply_start'][0]] == '\n'
    assert pieces[ex['masks']['reply_start'][0] - 1] == 'assistant'
    assert [pieces[i] for i in ex['masks']['termination']] == ['thanks', '<|im_end|>']
    assert pieces[-1] == '\n'  # trailing newline is kept in the IDs but in no mask
    last = len(pieces) - 1
    assert all(last not in m for m in ex['masks'].values())
    assert ex['lengths'] == dict(total=len(pieces), response_content=3)


def test_boundary_crossing_token_is_rejected_not_assumed():
    ex = seq.build_example(FakeTokenizer(merge_newline=True), 'x', msgs('Hi', 'Fine'), max_tokens=100,
                           overlength='exclude')
    assert ex['status'] == 'rejected' and 'crosses' in ex['reason']


def test_empty_response_rejected():
    ex = seq.build_example(FakeTokenizer(), 'x', msgs('Hi', ''), max_tokens=100, overlength='exclude')
    assert ex['status'] == 'rejected' and 'empty' in ex['reason']


def test_overlength_excluded_or_truncated_with_record():
    tok = FakeTokenizer()
    long = ' '.join(['word'] * 30)
    assert seq.build_example(tok, 'x', msgs('Hi', long), max_tokens=20, overlength='exclude')['status'] == 'excluded-overlength'
    ex = seq.build_example(tok, 'x', msgs('Hi', long), max_tokens=20, overlength='truncate')
    assert ex['status'] == 'ok' and ex['truncated'] is True and len(ex['input_ids']) == 20
    assert ex['masks']['termination'] == []
    assert max(ex['masks']['response_content']) == 19
    assert ex['truncation'] == dict(original_total=ex['truncation']['original_total'], kept=20)


def test_truncation_that_leaves_no_content_is_rejected():
    ex = seq.build_example(FakeTokenizer(), 'x', msgs('a b c d e f', 'g'), max_tokens=8, overlength='truncate')
    assert ex['status'] == 'rejected' and 'empty' in ex['reason']


# --- first turn and filters -------------------------------------------------------------------

def test_first_turn_extracts_only_first_pair_from_multi_turn_record():
    turn = seq.first_turn(msgs('Q1', 'A1', 'Q2', 'A2'))
    assert turn == dict(prompt='Q1', response='A1', n_messages=4)
    with pytest.raises(ValueError, match='roles'):
        seq.first_turn([dict(role='assistant', content='x'), dict(role='user', content='y')])
    with pytest.raises(ValueError, match='empty'):
        seq.first_turn(msgs('  ', 'A'))


@pytest.mark.parametrize('prompt,response,label', [
    ('Tell me about rivers', 'Rivers flow to the sea.', 'general'),
    ('When was it built?', 'It was finished in 1889.', 'near_miss'),
    ('How many apples?', 'There are many.', 'math'),
    ('Add these', '3 + 4 is seven', 'math'),
    ('Show code', 'Use ```print(1)```', 'code'),
    ('Write html', 'Use <div class="x"> here', 'code'),
    ('List 3 rivers', 'Nile, Amazon, Yangtze.', 'digit-prompt-only'),
])
def test_background_classifier(prompt, response, label):
    assert seq.classify_background(prompt, response) == label


def test_normalized_question_hash_ignores_whitespace_and_case():
    assert seq.question_hash('A  cat\nsat.') == seq.question_hash('a cat sat.')
    assert seq.question_hash('a cat sat.') != seq.question_hash('a dog sat.')


# --- end-to-end preparation -----------------------------------------------------------------

def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def world(tmp_path):
    root = tmp_path / 'artifacts'
    gsm = [dict(question=f'Question number {chr(65 + i % 26)}{i} about apples?', answer=f'Work.\n#### {i}')
           for i in range(60)]
    gsm[59]['question'] = gsm[3]['question']  # duplicate text of a protected question
    write_parquet(root / 'data/gsm8k-train.parquet', gsm)
    write_parquet(root / 'data/gsm8k-test.parquet', [dict(question=gsm[10]['question'], answer='x\n#### 1')])
    bg = []
    for i in range(40):
        name = chr(97 + i // 26) + chr(97 + i % 26)
        bg.append(dict(prompt_id=f'g{i}', prompt='x',
                       messages=msgs(f'Tell me about rivers {name}', 'Rivers flow to the sea.', 'More?', 'Yes.')))
    for i in range(20):
        bg.append(dict(prompt_id=f'n{i}', prompt='x', messages=msgs(f'When was bridge {i} built', 'In 1889.')))
    bg.append(dict(prompt_id='m0', prompt='x', messages=msgs('How many cats?', 'Five.')))
    bg.append(dict(prompt_id='dup', prompt='x', messages=msgs('Tell me about  rivers AA', 'Rivers flow.')))
    write_parquet(root / 'data/ultrachat.parquet', bg)
    (root / 'lineage').mkdir()
    (root / 'lineage/gold.json').write_text(json.dumps(
        [dict(id='gsm8k:train:3', question=gsm[3]['question']), dict(id='gsm8k:train:4')]))
    (root / 'lineage/grpo.jsonl').write_text(json.dumps(dict(item='gsm8k:train:5')) + '\n')
    (root / 'lineage/eval.json').write_text(json.dumps(dict(ids=['gsm8k:test:0'])))
    inventory = root / 'inventory'
    inventory.mkdir()
    (inventory / 'inventory.json').write_text(json.dumps(dict(arms={'gold': {}, 'grpo': {}, 'offline': {}})))
    (inventory / 'complete.json').write_text(json.dumps(dict(
        inventory_sha256=hashlib.sha256((inventory / 'inventory.json').read_bytes()).hexdigest())))
    return root


def plan(root, **overrides):
    value = dict(
        schema=seq.PLAN_SCHEMA, seed=42, max_tokens=200, overlength='exclude',
        inventory_dir='inventory',
        tokenizer=dict(kind='fake'),
        gsm8k=dict(dataset='openai/gsm8k', revision='r' * 40, license='mit',
                   train_parquet='data/gsm8k-train.parquet', test_parquet='data/gsm8k-test.parquet',
                   prompt='protocol.gsm8k_messages+gold-answer'),
        background=dict(dataset='HuggingFaceH4/ultrachat_200k', revision='u' * 40, license='mit',
                        split='test_sft', parquet='data/ultrachat.parquet'),
        counts=dict(math=dict(discovery=10, heldout=5, calibration=3),
                    general=dict(discovery=10, heldout=5, calibration=3),
                    near_miss=dict(discovery=5, heldout=3)),
        lineage=[dict(arm='gold', role='training', path='lineage/gold.json'),
                 dict(arm='grpo', role='rollout', path='lineage/grpo.jsonl'),
                 dict(arm='gold', role='evaluation', path='lineage/eval.json'),
                 dict(arm='grpo', role='selection', path='lineage/missing.json')],
        review_sample=dict(general=4, near_miss=2, math=1, code=0))
    value.update(overrides)
    path = root / 'plan.json'
    path.write_text(json.dumps(value))
    return path


def prepare(root, tmp_path, **kw):
    return seq.prepare(plan(root, **kw), root, tmp_path / 'out', 'fixture-v1',
                       tokenizer_factory=lambda spec, root: FakeTokenizer())


def load(run, name):
    return json.loads((run / name).read_text())


def rows(run):
    return [json.loads(line) for line in (run / 'sequences.jsonl').read_text().splitlines()]


def test_prepare_selects_disjoint_clean_splits_and_seals(world, tmp_path):
    manifest = prepare(world, tmp_path)
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    data = rows(run)
    by = {}
    for r in data:
        by.setdefault((r['group'], r['split']), []).append(r)
    assert {k: len(v) for k, v in by.items()} == {
        ('math', 'discovery'): 10, ('math', 'heldout'): 5, ('math', 'calibration'): 3,
        ('general', 'discovery'): 10, ('general', 'heldout'): 5, ('general', 'calibration'): 3,
        ('near_miss', 'discovery'): 5, ('near_miss', 'heldout'): 3}
    ids = [r['id'] for r in data]
    assert len(ids) == len(set(ids))
    math_refs = {r['source_ref'] for r in data if r['group'] == 'math'}
    # Protected IDs 3,4,5; 59 duplicates protected 3; 10 matches a test question.
    assert not math_refs & {3, 4, 5, 59, 10}
    convs = [r['source_ref'] for r in data if r['group'] != 'math']
    assert len(convs) == len(set(convs)) and 'm0' not in convs
    hashes = [r['text_sha256'] for r in data]
    assert len(hashes) == len(set(hashes))
    assert manifest['counts'] == {f'{g}/{s}': len(v) for (g, s), v in sorted(by.items())}
    seal = load(run, 'complete.json')
    for name, digest in seal['files'].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    assert load(run, 'meta/status.json')['state'] == 'complete'
    assert manifest['background_review'] == 'pending'


def test_leakage_report_lists_sources_unknowns_and_never_claims_clean(world, tmp_path):
    prepare(world, tmp_path)
    report = load(tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1', 'leakage_report.json')
    assert report['protected_train_ids'] == [3, 4, 5]
    assert report['excluded']['duplicate_text_of_protected'] == [59]
    assert report['excluded']['matches_test_question'] == [10]
    assert report['arms']['grpo']['missing_sources'] == ['lineage/missing.json']
    assert report['arms']['offline']['status'] == 'unknown-no-lineage-listed'
    assert report['cleanliness'] == 'no-overlap-with-available-sources'
    assert 'offline' in report['arms_with_unknown_provenance']


def test_multi_turn_records_use_first_turn_only(world, tmp_path):
    tok = FakeTokenizer()
    seq.prepare(plan(world), world, tmp_path / 'out', 'fixture-v1', tokenizer_factory=lambda spec, root: tok)
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    general = [r for r in rows(run) if r['group'] == 'general']
    assert all(r['n_messages'] == 4 for r in general)
    text = ''.join(tok.token_text(i) for i in general[0]['input_ids'])
    assert 'More?' not in text and text.count('<|im_start|>assistant') == 1


def test_reproducible_from_seed_and_ids(world, tmp_path):
    a = prepare(world, tmp_path)
    b = seq.prepare(plan(world), world, tmp_path / 'out2', 'fixture-v1',
                    tokenizer_factory=lambda spec, root: FakeTokenizer())
    ra = (tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1' / 'sequences.jsonl').read_bytes()
    rb = (tmp_path / 'out2' / seq.RUN_SUBDIR / 'fixture-v1' / 'sequences.jsonl').read_bytes()
    assert ra == rb and a['sequences_sha256'] == b['sequences_sha256']
    c = seq.prepare(plan(world, seed=7), world, tmp_path / 'out3', 'fixture-v1',
                    tokenizer_factory=lambda spec, root: FakeTokenizer())
    assert c['sequences_sha256'] != a['sequences_sha256']


def test_insufficient_pool_stops_and_reports(world, tmp_path):
    counts = dict(math=dict(discovery=50, heldout=5, calibration=3),
                  general=dict(discovery=10, heldout=5, calibration=3), near_miss=dict(discovery=5, heldout=3))
    with pytest.raises(ValueError, match='Insufficient'):
        prepare(world, tmp_path, counts=counts)
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    status = load(run, 'meta/status.json')
    assert status['state'] == 'failed' and 'math' in status['error']
    assert not (run / 'complete.json').exists()


def test_unsealed_inventory_stops_preparation(world, tmp_path):
    (world / 'inventory/complete.json').unlink()
    with pytest.raises(ValueError, match='inventory'):
        prepare(world, tmp_path)


def test_review_and_decode_artifacts_are_written(world, tmp_path):
    prepare(world, tmp_path)
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    sample = [json.loads(l) for l in (run / 'review/background_sample.jsonl').read_text().splitlines()]
    assert {s['label'] for s in sample} == {'general', 'near_miss', 'math'}
    decodes = (run / 'review/decodes.md').read_text()
    assert decodes.count('## ') >= 3 and 'response_content' in decodes and 'truncated' in decodes


def test_review_record_binds_to_sample_and_cannot_be_silently_replaced(world, tmp_path):
    prepare(world, tmp_path)
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    record = seq.record_review(run, reviewer='tester', decision='accept', notes='ok')
    assert record['sample_sha256'] == hashlib.sha256((run / 'review/background_sample.jsonl').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='already'):
        seq.record_review(run, reviewer='tester', decision='reject', notes='')
    assert seq.verify(run)['background_review'] == 'accept'


def test_resume_reuses_and_changed_input_refused(world, tmp_path):
    first = prepare(world, tmp_path)
    assert prepare(world, tmp_path) == first
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    assert seq.verify(run)['ok'] is True
    (world / 'lineage/grpo.jsonl').write_text(json.dumps(dict(item='gsm8k:train:6')) + '\n')
    with pytest.raises(ValueError, match='changed'):
        prepare(world, tmp_path)


def test_corrupt_sealed_output_detected(world, tmp_path):
    prepare(world, tmp_path)
    run = tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1'
    (run / 'sequences.jsonl').write_text('')
    assert seq.verify(run)['ok'] is False
    with pytest.raises(ValueError, match='integrity'):
        prepare(world, tmp_path)


@pytest.mark.skipif(not (REAL_TOKENIZER / 'tokenizer.json').exists() or not GOLD_ITEMS.exists(),
                    reason='local pinned tokenizer/Gold items not present')
def test_real_tokenizer_reproduces_transformers_rendering_on_gold_items():
    tok = seq.ChatTokenizer(REAL_TOKENIZER)
    assert (tok.pad_token, tok.pad_token_id, tok.eos_token_id) == ('<|endoftext|>', 151643, 151645)
    items = json.loads(GOLD_ITEMS.read_text())[:32]
    for item in items:
        messages = seq.math_messages(item['question'], item['gold'])
        assert tok.render(messages, add_generation_prompt=False) == item['rendered']
        ids, _ = tok.encode(item['rendered'])
        assert ids[:len(item['input_ids'])] == item['input_ids']
        ex = seq.build_example(tok, item['id'], messages, max_tokens=4096, overlength='exclude')
        assert ex['status'] == 'ok'
        assert tok.token_text(ex['input_ids'][ex['masks']['reply_start'][0]]) == '\n'


def test_malformed_lineage_line_is_reported_never_dropped(tmp_path):
    source = tmp_path / 'rollout.jsonl'
    source.write_text(json.dumps(dict(question='What is two plus two?')) + '\n{not json\n')
    ids, questions, errors = seq._lineage_file(source)
    assert len(questions) == 1 and errors and errors[0]['line'] == 2  # the valid line still counts


def test_unparseable_lineage_marks_the_arm_unknown(world, tmp_path):
    (world / 'lineage/grpo.jsonl').write_text(json.dumps(dict(item='gsm8k:train:5')) + '\n{broken\n')
    (world / 'lineage/missing.json').write_text('{}')
    prepare(world, tmp_path)
    report = load(tmp_path / 'out' / seq.RUN_SUBDIR / 'fixture-v1', 'leakage_report.json')
    assert report['arms']['grpo']['status'] == 'partial-some-sources-unparseable'
    assert report['arms']['grpo']['unparseable_sources'] == ['lineage/grpo.jsonl']
    assert 'grpo' in report['arms_with_unknown_provenance']
