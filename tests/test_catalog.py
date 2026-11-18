"""Catalog of arms, cohorts, storage locations and response sets (CPU, no model)."""
import hashlib
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

from pilot_eval import catalog as cat  # noqa: E402

ADAPTER = 'a' * 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


@pytest.fixture
def world(tmp_path):
    """A tiny artifact root and a catalog that describes it correctly."""
    root = tmp_path / 'artifacts'
    ckpt = root / 'runs/arm/checkpoint-2/adapter_model.safetensors'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'weights')
    rows = [dict(item_id=f'q{q}', step=2, mode='sampled', draw_index=d, reward=int(d == 0),
                 adapter_sha256=sha(ckpt)) for q in range(3) for d in range(2)]
    rows += [dict(item_id=f'q{q}', step=2, mode='greedy', draw_index=0, reward=1, adapter_sha256=sha(ckpt))
             for q in range(3)]
    responses = write_jsonl(root / 'runs/arm/responses.jsonl', rows)
    ids = write_jsonl(root / 'cohort/items.jsonl', [dict(id=f'q{q}') for q in range(3)])

    cdir = tmp_path / 'catalog'
    dump(cdir / 'locations.yaml', {'locations': {
        'local': {'kind': 'local-artifacts', 'description': 'artifact root'},
        'hf': {'kind': 'hf-dataset', 'repo': 'someone/private', 'description': 'backup'}}})
    dump(cdir / 'arms.yaml', {'arms': {'arm-a': {
        'aliases': ['old-name'], 'description': 'test arm',
        'checkpoints': {2: {'location': 'local', 'path': 'runs/arm/checkpoint-2/adapter_model.safetensors',
                            'sha256': sha(ckpt)}}}}})
    dump(cdir / 'cohorts.yaml', {'cohorts': {'c3': {
        'description': 'three questions', 'split': 'test', 'size': 3,
        'source': {'location': 'local', 'path': 'cohort/items.jsonl', 'id_field': 'id'},
        'ids_sha256': cat.ids_digest([f'q{q}' for q in range(3)]), 'prior_uses': ['unit test']}}})
    base = {
        'kind': 'evaluation', 'arm': 'arm-a', 'checkpoint_step': 2, 'cohort': 'c3',
        'policy': {'mode': 'sampled', 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'repetition_penalty': 1.0,
                   'max_new_tokens': 8, 'eos_token_id': 1},
        'draws_per_question': 2, 'question_field': 'item_id',
        'scorer': {'name': 'toy'},
        'selection': {'correctness_filtered': False, 'gold_access': False, 'all_groups_retained': True},
        'c_suitability': 'on-policy',
        'files': [{'location': 'local', 'path': 'runs/arm/responses.jsonl', 'sha256': sha(responses),
                   'rows_total': 9, 'select': {'step': 2, 'mode': 'sampled'}, 'rows_selected': 6}],
        'evidence': {'status': 'verified', 'checked': ['fixture']}}
    dump(cdir / 'responses/arm-a.sampled.yaml', dict(base, id='arm-a.sampled'))
    return dict(root=root, cdir=cdir, base=base, responses=responses, ids=ids, ckpt=ckpt)


def verify(world):
    return cat.verify(cat.load(world['cdir']), roots={'local': world['root']})


def failures(report):
    return [c for c in report['checks'] if c['status'] == 'fail']


def test_correct_catalog_passes(world):
    report = verify(world)
    assert failures(report) == []
    assert report['ok'] is True
    assert {c['entry'] for c in report['checks']} >= {'arm:arm-a', 'cohort:c3', 'responses:arm-a.sampled'}


def test_changed_file_bytes_fail(world):
    world['responses'].write_text(world['responses'].read_text() + '\n')
    assert any('sha256' in c['detail'] for c in failures(verify(world)))


def test_wrong_selected_row_count_fails(world):
    entry = dict(world['base'], id='arm-a.sampled')
    entry['files'] = [dict(entry['files'][0], rows_selected=5)]
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', entry)
    assert any('rows_selected' in c['detail'] for c in failures(verify(world)))


def test_draws_per_question_checked(world):
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', dict(world['base'], id='arm-a.sampled', draws_per_question=8))
    assert any('draws' in c['detail'] for c in failures(verify(world)))


def test_question_ids_must_equal_cohort(world):
    rows = [json.loads(line) for line in world['responses'].read_text().splitlines()]
    rows[0]['item_id'] = rows[1]['item_id'] = 'q99'
    write_jsonl(world['responses'], rows)
    entry = dict(world['base'], id='arm-a.sampled')
    entry['files'] = [dict(entry['files'][0], sha256=sha(world['responses']))]
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', entry)
    assert any('cohort' in c['detail'] for c in failures(verify(world)))


def test_row_adapter_hash_must_match_checkpoint(world):
    rows = [json.loads(line) for line in world['responses'].read_text().splitlines()]
    rows[0]['adapter_sha256'] = 'f' * 64  # one draw from a different adapter
    write_jsonl(world['responses'], rows)
    entry = dict(world['base'], id='arm-a.sampled')
    entry['files'] = [dict(entry['files'][0], sha256=sha(world['responses']))]
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', entry)
    report = verify(world)
    details = ' '.join(c['detail'] for c in failures(report))
    assert 'adapter_sha256' in details


def test_cohort_ids_digest_checked(world):
    write_jsonl(world['ids'], [dict(id=f'q{q}') for q in range(4)])
    assert any(c['entry'] == 'cohort:c3' for c in failures(verify(world)))


def test_non_local_files_are_reported_not_failed(world):
    entry = dict(world['base'], id='arm-a.remote')
    entry['files'] = [{'location': 'hf', 'path': 'x/bundle.tar.gz', 'sha256': 'b' * 64}]
    dump(world['cdir'] / 'responses/arm-a.remote.yaml', entry)
    report = verify(world)
    assert report['ok'] is True
    assert [c['status'] for c in report['checks'] if c['entry'] == 'responses:arm-a.remote'] == ['not-local']


def test_missing_local_file_fails(world):
    world['responses'].unlink()
    assert any('missing' in c['detail'] for c in failures(verify(world)))


@pytest.mark.parametrize('field, value, message', [
    ('arm', 'nobody', 'unknown arm'),
    ('cohort', 'nothing', 'unknown cohort'),
    ('replay_of', 'ghost', 'unknown replay_of'),
    ('c_suitability', 'maybe', 'c_suitability'),
])
def test_references_and_enums_validated(world, field, value, message):
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', dict(world['base'], id='arm-a.sampled', **{field: value}))
    with pytest.raises(cat.CatalogError, match=message):
        cat.load(world['cdir'])


def test_required_fields_and_unique_ids(world):
    entry = dict(world['base'], id='arm-a.sampled')
    del entry['evidence']
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', entry)
    with pytest.raises(cat.CatalogError, match='evidence'):
        cat.load(world['cdir'])
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml', dict(world['base'], id='arm-a.sampled'))
    dump(world['cdir'] / 'responses/copy.yaml', dict(world['base'], id='arm-a.sampled'))
    with pytest.raises(cat.CatalogError, match='duplicate'):
        cat.load(world['cdir'])


def test_unverified_evidence_needs_a_note(world):
    dump(world['cdir'] / 'responses/arm-a.sampled.yaml',
         dict(world['base'], id='arm-a.sampled', evidence={'status': 'inferred', 'checked': []}))
    with pytest.raises(cat.CatalogError, match='note'):
        cat.load(world['cdir'])


def test_aliases_resolve_to_canonical_arm(world):
    catalog = cat.load(world['cdir'])
    assert cat.resolve_arm(catalog, 'old-name') == 'arm-a'
    assert cat.resolve_arm(catalog, 'arm-a') == 'arm-a'
    with pytest.raises(cat.CatalogError):
        cat.resolve_arm(catalog, 'nope')


def test_find_filters_response_sets(world):
    dump(world['cdir'] / 'responses/arm-a.greedy.yaml', dict(
        world['base'], id='arm-a.greedy', draws_per_question=1, c_suitability='greedy-only',
        policy=dict(world['base']['policy'], mode='greedy'),
        files=[dict(world['base']['files'][0], select={'step': 2, 'mode': 'greedy'}, rows_selected=3)]))
    catalog = cat.load(world['cdir'])
    assert [e['id'] for e in cat.find(catalog, arm='old-name', mode='sampled')] == ['arm-a.sampled']
    assert [e['id'] for e in cat.find(catalog, c_suitability='greedy-only')] == ['arm-a.greedy']
    assert failures(verify(world)) == []


def test_cli_verify_exit_code(world, capsys):
    args = ['verify', '--catalog', str(world['cdir']), '--root', f"local={world['root']}"]
    assert cat.main(args) == 0
    world['responses'].write_text('')
    assert cat.main(args) == 1
    assert 'FAIL' in capsys.readouterr().out
