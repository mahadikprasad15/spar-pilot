"""Sparse-rewiring validation set sampler (ticket 04, spec test 12). CPU, fixtures, no download."""
import hashlib
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

from pilot_eval import catalog as cat  # noqa: E402
from pilot_eval import sparse_validation as sv  # noqa: E402
from pilot_eval.sft import gold_item, gold_training_indices, load_sft, prepare_sft  # noqa: E402
from test_sft_prepare import TrainingData, source_plan  # noqa: E402

N_TRAIN = 1000
SEQ = 'runs/joint-neuron-adapter-audit/sequences/sequences-v2'


class BigTrain(TrainingData):
    """1,000 train rows: enough left after the gold 512 for a 128 sample. Row 7 repeats gold row text."""

    def load_dataset(self, path, name, revision):
        data = super().load_dataset(path, name, revision)
        if name == 'main':
            data['train'] = [dict(question=f'Train {i}', answer='2+1=<<2+1=3>>3\n#### 3') for i in range(N_TRAIN)]
            first_gold = gold_training_indices(N_TRAIN)[0]
            data['train'][DUPLICATE]['question'] = f'  train {first_gold} '  # same normalized text
        return data


GOLD = set(gold_training_indices(N_TRAIN))
DUPLICATE = next(i for i in range(N_TRAIN) if i not in GOLD)
SEQ_TRAIN = sorted(i for i in range(N_TRAIN) if i not in GOLD)[1:41]        # 40 sequences-v2 math IDs
RECORD_TRAIN = sorted(i for i in range(N_TRAIN) if i not in GOLD)[41:51]     # 10 IDs in a run record
EXTRA_COHORT = sorted(i for i in range(N_TRAIN) if i not in GOLD)[51:56]     # another recorded train cohort
PRIOR_USE = sorted(i for i in range(N_TRAIN) if i not in GOLD)[56]           # named only in prior_uses text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def write_sequences(root, ids):
    run = root / SEQ
    run.mkdir(parents=True, exist_ok=True)
    rows = [dict(id=i, group='math' if i.startswith('gsm8k') else 'general', split='discovery') for i in ids]
    (run / 'sequences.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    manifest = dict(schema='joint-neuron-adapter-audit/sequences-v1', run_id='sequences-v2', order=ids,
                    sequences_sha256=sha(run / 'sequences.jsonl'))
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'complete.json').write_text(json.dumps(dict(files={
        n: sha(run / n) for n in ['sequences.jsonl', 'manifest.json']})))
    return run


@pytest.fixture
def world(tmp_path):
    root = tmp_path / 'artifacts'
    root.mkdir()
    deps = BigTrain()
    sft_config = prepare_sft(source_plan(root, deps), root, 'sft-v1', dependencies=deps)
    gold_ids = [f'gsm8k:train:{i}' for i in sorted(GOLD)]
    (root / 'gold.json').write_text(json.dumps({'gold': [dict(problem_id=i) for i in gold_ids]}))
    (root / 'extra.jsonl').write_text(''.join(json.dumps(dict(id=f'gsm8k:train:{i}')) + '\n' for i in EXTRA_COHORT))
    write_sequences(root, [f'gsm8k:train:{i}' for i in SEQ_TRAIN] + ['ultrachat:train_sft:abc'])
    record = root / 'runs/gold/steps.jsonl'
    record.parent.mkdir(parents=True)
    record.write_text(''.join(json.dumps(dict(step=s, ids=[f'gsm8k:train:{i}'])) + '\n'
                              for s, i in enumerate(RECORD_TRAIN)))
    cdir = tmp_path / 'catalog'
    dump(cdir / 'locations.yaml', {'locations': {'local': {'kind': 'local-artifacts', 'description': 'root'},
                                                 'hf': {'kind': 'hf-dataset', 'description': 'backup'}}})
    dump(cdir / 'arms.yaml', {'arms': {}})
    dump(cdir / 'cohorts.yaml', {'cohorts': {
        'gsm8k-test-cohort150': {'split': 'test', 'size': 150, 'ids_sha256': 'x' * 64,
                                 'source': {'location': 'local', 'path': 'missing/test.json'},
                                 'prior_uses': ['greedy evaluation', f'C rehearsal on gsm8k:train:{PRIOR_USE}']},
        'gsm8k-train-512': {'split': 'train', 'size': 512, 'ids_sha256': cat.ids_digest(gold_ids),
                            'source': {'location': 'local', 'path': 'gold.json', 'json_path': ['gold'],
                                       'id_field': 'problem_id'}},
        'extra-train': {'split': 'train', 'size': len(EXTRA_COHORT),
                        'ids_sha256': cat.ids_digest([f'gsm8k:train:{i}' for i in EXTRA_COHORT]),
                        'source': {'location': 'local', 'path': 'extra.jsonl', 'id_field': 'id'}}}})
    (cdir / 'responses').mkdir()
    return dict(root=root, deps=deps, sft_config=sft_config, catalog=cdir, records=['runs/gold/steps.jsonl'])


def build(world, **overrides):
    kwargs = dict(catalog_dir=world['catalog'], sequences_dir=SEQ, run_records=world['records'],
                  dependencies=world['deps'])
    kwargs.update(overrides)
    return sv.build(world['sft_config'], world['root'], **kwargs)


def train_index(item_id):
    return int(item_id.rsplit(':', 1)[1])


def test_samples_128_seeded_train_ids_outside_every_exclusion_source(world):
    items, record = build(world)
    ids = [i['id'] for i in items]
    assert len(ids) == len(set(ids)) == sv.SIZE == 128
    assert ids == sorted(ids, key=train_index)
    picked = {train_index(i) for i in ids}
    for excluded in [GOLD, SEQ_TRAIN, RECORD_TRAIN, EXTRA_COHORT, {PRIOR_USE, DUPLICATE}]:
        assert not picked & set(excluded)
    assert record['seed'] == sv.SEED == 20261122
    assert record['ids_sha256'] == cat.ids_digest(ids)
    kinds = {s['kind'] for s in record['sources']}
    assert kinds == {'gold-plan', 'catalog-cohort', 'catalog-prior-uses', 'sequences', 'run-record'}
    assert all(s['sha256'] for s in record['sources'])
    assert record['excluded']['duplicate_text_of_protected'] == [f'gsm8k:train:{DUPLICATE}']
    n_excluded = len(GOLD | set(SEQ_TRAIN) | set(RECORD_TRAIN) | set(EXTRA_COHORT) | {PRIOR_USE})
    assert record['excluded']['by_id'] == n_excluded
    assert record['eligible'] == N_TRAIN - n_excluded - 1
    # Reproducible: the same inputs give the same set.
    assert build(world)[1]['ids'] == record['ids']


def test_items_are_built_by_the_gold_builder(world):
    """A gold index through the shared builder reproduces gold's own frozen row exactly."""
    config, gold_rows, _ = load_sft(world['sft_config'], world['root'])
    source = config['source_config']
    tokenizer, context = world['deps'].load_tokenizer(source['model'], source['tokenizer_revision'])
    data = world['deps'].load_dataset(source['dataset_path'], source['dataset_config'], source['dataset_revision'])
    row = gold_rows[3]
    rebuilt = gold_item(tokenizer, data['train'][row['source_index']], row['source_index'],
                        dataset_revision=source['dataset_revision'],
                        context_limit=min(context, source['context_limit']))
    assert rebuilt == row
    items, record = build(world)
    item = items[0]
    assert set(item) == set(row)
    assert item['labels'] == [-100] * item['prompt_tokens'] + item['input_ids'][item['prompt_tokens']:]
    assert item['labels'][-1] == tokenizer.eos_token_id
    assert record['items_sha256'] == sv._hash(items)


def test_freeze_writes_items_and_exclusion_record_and_refuses_a_changed_set(world, tmp_path):
    items, record = build(world)
    out = tmp_path / 'run/inputs'
    paths = sv.freeze(out, items, record)
    assert json.loads(paths['items'].read_text()) == items
    assert json.loads(paths['record'].read_text())['items_sha256'] == record['items_sha256']
    sv.freeze(out, items, record)  # identical: reused
    with pytest.raises(ValueError, match='frozen artifact mismatch'):
        sv.freeze(out, items[1:], dict(record, items_sha256=sv._hash(items[1:]), ids=record['ids'][1:]))
    with pytest.raises(ValueError, match='items_sha256'):
        sv.freeze(tmp_path / 'other', items[1:], record)


@pytest.mark.parametrize('seed', [0, 1, 7, 42, 20261122, 99991])
def test_never_returns_an_excluded_id_for_any_seed(world, seed):
    _, record = build(world, seed=seed)
    picked = {train_index(i) for i in record['ids']}
    assert not picked & (GOLD | set(SEQ_TRAIN) | set(RECORD_TRAIN) | set(EXTRA_COHORT) | {PRIOR_USE, DUPLICATE})


# --- refusals: an unreadable or malformed source fails, never "no overlap" -------------------

def test_refuses_a_malformed_exclusion_jsonl_line(world):
    path = world['root'] / world['records'][0]
    path.write_text(path.read_text() + '{"step": 99, "ids": ["gsm8k:train:3"\n')
    with pytest.raises(ValueError, match='malformed.*line 11'):
        build(world)


def test_refuses_a_missing_run_record(world):
    with pytest.raises(ValueError, match='unreadable'):
        build(world, run_records=world['records'] + ['runs/gold/absent.jsonl'])


def test_refuses_a_run_record_with_no_gsm8k_ids(world):
    (world['root'] / 'empty.jsonl').write_text('{"step": 1}\n')
    with pytest.raises(ValueError, match='no GSM8K'):
        build(world, run_records=['empty.jsonl'])


def test_refuses_a_run_record_id_outside_the_train_split(world):
    (world['root'] / 'big.jsonl').write_text(json.dumps(dict(id=f'gsm8k:train:{N_TRAIN + 5}')) + '\n')
    with pytest.raises(ValueError, match='outside'):
        build(world, run_records=['big.jsonl'])


def test_refuses_an_unsealed_or_tampered_sequences_run(world):
    run = world['root'] / SEQ
    (run / 'sequences.jsonl').write_text((run / 'sequences.jsonl').read_text() + '\n')
    with pytest.raises(ValueError, match='sequences'):
        build(world)
    (run / 'complete.json').unlink()
    with pytest.raises(ValueError, match='not sealed'):
        build(world)
    with pytest.raises(ValueError, match='sequences'):
        build(world, sequences_dir='runs/nowhere')


def test_refuses_a_sequences_run_with_no_gsm8k_train_ids(world):
    write_sequences(world['root'], ['ultrachat:train_sft:abc'])
    with pytest.raises(ValueError, match='no GSM8K train'):
        build(world)


def test_refuses_an_unreadable_catalog_train_cohort(world):
    (world['root'] / 'extra.jsonl').unlink()
    with pytest.raises(ValueError, match='extra-train.*unreadable'):
        build(world)


def test_refuses_a_malformed_catalog_cohort_file(world):
    path = world['root'] / 'extra.jsonl'
    path.write_text(path.read_text() + '{"id": \n')
    with pytest.raises(ValueError, match='extra-train'):
        build(world)


def test_refuses_a_catalog_cohort_whose_ids_differ_from_its_digest(world):
    path = world['root'] / 'extra.jsonl'
    path.write_text(path.read_text().replace(f'gsm8k:train:{EXTRA_COHORT[0]}', 'gsm8k:train:0'))
    with pytest.raises(ValueError, match='extra-train.*ids_sha256'):
        build(world)


def test_gold_cohort_off_machine_is_resolved_only_by_a_matching_gold_plan(world):
    (world['root'] / 'gold.json').unlink()
    _, record = build(world)
    gold = next(s for s in record['sources'] if s.get('name') == 'gsm8k-train-512')
    assert gold['resolved_by'] == 'gold-plan'
    cohorts = yaml.safe_load((world['catalog'] / 'cohorts.yaml').read_text())
    cohorts['cohorts']['gsm8k-train-512']['ids_sha256'] = 'f' * 64
    dump(world['catalog'] / 'cohorts.yaml', cohorts)
    with pytest.raises(ValueError, match='gsm8k-train-512'):
        build(world)


def test_refuses_too_few_eligible_questions(world):
    with pytest.raises(ValueError, match='eligible'):
        build(world, size=N_TRAIN)
