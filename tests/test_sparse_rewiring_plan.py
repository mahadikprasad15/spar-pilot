"""Ticket 05: sparse-rewiring launch record and frozen plan (spec test 8, typed YAML, D24 storage)."""
import json
import random
from pathlib import Path

import pytest
import yaml

from pilot_eval import c_attribution_chain as cc
from pilot_eval import sparse_rewiring_plan as sp
from pilot_eval.attribution_chain import LaunchRefused

DRAFT = Path(__file__).resolve().parents[1] / '.scratch/sparse-rewiring-gsm8k/launch-record-draft.yaml'
UNSIGNED = ['D1', 'D2', 'D3', 'D4', 'D7', 'D8', 'D9', 'D10', 'D11', 'D15']
FILL = {'inputs.artifact_root': '/w/inputs', 'inputs.bundle': '/w/b.json', 'inputs.model_dir': '/w/model',
        'inputs.work_dir': '/w'}


def _signed(**changes):
    """The draft with every decision signed (as the researcher would), plus dotted overrides."""
    record = yaml.safe_load(DRAFT.read_text())
    for key, text in record['decisions'].items():
        record['decisions'][key] = text.replace('[proposed]', '[user test]')
    for dotted, value in changes.items():
        parent, keys = record, dotted.split('.')
        for k in keys[:-1]:
            parent = parent[k]
        parent[keys[-1]] = value
    return record


def _write(tmp_path, record, name='draft.yaml'):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True))
    return path


def _seal(tmp_path, record):
    return sp.finalize_draft(_write(tmp_path, record), tmp_path / 'launch_record.json', code_commit='c0ffee',
                             pod_id='pod1', fill=FILL)


def _record(**changes):
    record = _signed(**changes)
    record.update(code_commit='c0ffee', pod_id='pod1')
    record['inputs'].update({k.split('.')[1]: v for k, v in FILL.items()})
    return record


def _gold():
    ids = [f'gsm8k:train:{i}' for i in sorted(random.Random(7).sample(range(7473), 512))]
    order = list(range(512))
    random.Random(42).shuffle(order)   # what sft_backend does to row indices
    return ids, [ids[i] for i in order]


# --- launch record -----------------------------------------------------------------------------------

def test_draft_refused_until_every_proposed_decision_is_signed(tmp_path):
    problems = sp.draft_problems(DRAFT)
    assert sorted(p.split(' ')[0] for p in problems) == sorted(f'decisions.{d}' for d in UNSIGNED)
    with pytest.raises(LaunchRefused, match='unresolved'):
        sp.finalize_draft(DRAFT, tmp_path / 'r.json', code_commit='c0ffee', pod_id='pod1', fill=FILL)
    assert sp.main(['check', str(DRAFT)]) == 1


def test_signed_draft_seals_and_loads(tmp_path):
    path = _seal(tmp_path, _signed())
    record = sp.load_launch_record(path, code_commit='c0ffee')
    assert record['objective']['l0_lambda'] == 6e-4
    assert {k: (v['rank'], v['read_lr']) for k, v in record['arms'].items()} == {
        'sparse-gold-sft-r1': (1, 3e-3), 'sparse-gold-sft-r0': (0, 3e-4)}
    with pytest.raises(LaunchRefused, match='code commit'):
        sp.load_launch_record(path, code_commit='other')
    tampered = json.loads(path.read_text())
    tampered['objective']['l0_lambda'] = 7e-4
    path.write_text(json.dumps(tampered))
    with pytest.raises(LaunchRefused, match='seal'):
        sp.load_launch_record(path, code_commit='c0ffee')


def test_c_attribution_loader_refuses_a_sparse_record(tmp_path):
    with pytest.raises(LaunchRefused):
        cc.load_launch_record(_seal(tmp_path, _signed()), code_commit='c0ffee')


@pytest.mark.parametrize('line, typo', [
    ('disk: {min_free_gb: 60,', 'disk: {min_free_gb: 19.5e12,'),       # lesson 2026-11-16: read as text
    ('l0_lambda: 6.0e-4', 'l0_lambda: 6e-4'),
    ('steps: 1024 ', 'steps: 1024.0 '),
    ('gpu: A100-80GB, count: 1', 'gpu: A100-80GB, count: true'),
])
def test_mistyped_yaml_value_is_refused(tmp_path, line, typo):
    source = DRAFT.read_text().replace("'[proposed]", "'[user test]").replace('"[proposed]', '"[user test]')
    assert line in source
    assert sp.draft_problems(_write(tmp_path, yaml.safe_load(source))) == []
    bad = tmp_path / 'bad.yaml'
    bad.write_text(source.replace(line, typo))
    assert sp.draft_problems(bad)
    with pytest.raises(LaunchRefused, match='launch record problems'):
        sp.finalize_draft(bad, tmp_path / 'r.json', code_commit='c0ffee', pod_id='pod1', fill=FILL)


@pytest.mark.parametrize('dotted, value, match', [
    ('objective.l0_lambda', 7e-4, 'l0_lambda'),
    ('training.freeze_after_step', 900, 'freeze_after_step'),
    ('training.steps', 512, 'steps'),
    ('arms.sparse-gold-sft-r0.read_lr', 3e-3, 'arms'),
    ('data.examples_per_mask', 8, 'examples_per_mask'),
    ('hardware.gpu', 'L4', 'hardware'),
    ('evaluation.cohort150.steps', [64, 512, 1024], 'cohort150'),
    ('checkpoints', [0, 8, 16, 32, 64, 128, 256, 512, 960, 1024], 'checkpoints'),
])
def test_production_holds_the_decided_values(dotted, value, match):
    with pytest.raises(LaunchRefused, match=match):
        sp.validate(_record(**{dotted: value}), code_commit='c0ffee')


@pytest.mark.parametrize('dotted, value, match', [
    ('storage.checkpoint_contents', ['read_factors', 'log_alpha', 'optimizer', 'seal'], 'checkpoint_contents'),
    ('storage.interruption', 'resume', 'interruption'),
    ('runtime.precision', 'tf32', 'precision'),
    ('data.microbatch', 4, 'microbatch'),
    ('optimizer.masks.clip', '1.0', 'clip'),
    ('arms.sparse-gold-sft-r1.rank', 4, 'rank'),
    ('arms.sparse-gold-sft-r0.run_id', 'sparse-gold-sft-r1-1', 'run_id'),
])
def test_rules_hold_in_every_mode(dotted, value, match):
    rehearsal = dict(mode='rehearsal', label='rehearsal-tiny')
    for mode_changes in ({}, rehearsal):
        with pytest.raises(LaunchRefused, match=match):
            sp.validate(_record(**mode_changes, **{dotted: value}), code_commit='c0ffee')


def test_rehearsal_may_shorten_the_run_but_stays_consistent():
    short = {'mode': 'rehearsal', 'label': 'rehearsal-qualification', 'training.steps': 6,
             'training.warmup_steps': 2, 'training.freeze_after_step': 4, 'data.gold_order_steps': 6,
             'checkpoints': [0, 4, 6], 'evaluation.cohort150.steps': [6], 'cadences.validation_ce_every': 2,
             'cadences.gate_snapshot_every': 2, 'cadences.settle.first': 2, 'cadences.settle.every': 2}
    record = sp.validate(_record(**short), code_commit='c0ffee')
    assert sp.validation_steps(record) == [2, 4, 6] and sp.settle_steps(record) == [2, 4]
    with pytest.raises(LaunchRefused, match='freeze'):
        sp.validate(_record(**{**short, 'checkpoints': [0, 6]}), code_commit='c0ffee')
    with pytest.raises(LaunchRefused, match='warmup'):
        sp.validate(_record(**{**short, 'training.warmup_steps': 5}), code_commit='c0ffee')


def test_disk_block_must_cover_the_d24_storage_plan():
    record = _record()
    planned = sp.planned_disk_bytes(record) / 1e9
    assert 33 < planned < 35                     # spec: about 34 GB for rank 0, rank 1 negligible
    with pytest.raises(LaunchRefused, match='D24'):
        sp.validate(_record(**{'disk.container_total_gb': 80}), code_commit='c0ffee')
    assert sp.validate(_record(**{'disk.container_total_gb': 95}), code_commit='c0ffee')


def test_storage_arithmetic_matches_the_decision_log():
    assert sp.read_parameters(1) == 587_776
    assert sp.read_parameters(0) == 770_703_360
    assert sp.read_parameters(4) == 4 * 587_776
    assert 3.0e9 < sp.checkpoint_bytes(0) < 3.2e9      # about 3.1 GB per rank-0 checkpoint
    assert 3.0e6 < sp.checkpoint_bytes(1) < 3.6e6      # about 3.5 MB per rank-1 checkpoint


def test_missing_decision_is_refused():
    record = _record()
    del record['decisions']['D14']
    with pytest.raises(LaunchRefused, match='D14'):
        sp.validate(record, code_commit='c0ffee')


# --- plan (spec test 8) ------------------------------------------------------------------------------

def _plan(record=None, arm='sparse-gold-sft-r1'):
    ids, gold = _gold()
    record = record or sp.validate(_record(), code_commit='c0ffee')
    return record, ids, gold, sp.build_plan(record, arm, ids, gold, inputs_sha256={'training_items_sha256': 'x'})


def test_first_64_steps_are_golds_order_and_later_epochs_follow_d7():
    record, ids, gold, plan = _plan()
    steps = plan['steps']
    assert [s['step'] for s in steps] == list(range(1, 1025))
    assert all(len(s['ids']) == 8 for s in steps)
    assert [i for s in steps[:64] for i in s['ids']] == gold
    for epoch in range(2, 17):
        expected = list(ids)
        random.Random(42 + epoch - 1).shuffle(expected)
        consumed = [i for s in steps[(epoch - 1) * 64:epoch * 64] for i in s['ids']]
        assert consumed == expected
        assert {s['epoch'] for s in steps[(epoch - 1) * 64:epoch * 64]} == {epoch}
    assert [i for s in steps[64:128] for i in s['ids']] != gold   # not a periodic repeat of gold's batches
    for step in (1, 64, 65, 700, 1024):
        assert sp.step_ids(plan, step) == steps[step - 1]['ids']
    assert plan['inputs']['gold_order_reconstruction'] == 'matches'


def test_plan_schedule_masks_and_cadences():
    _, _, _, plan = _plan()
    steps = plan['steps']
    assert all(s['mask_groups'] == [[0, 4], [4, 8]] for s in steps)                  # D14: 2 draws per step
    assert steps[0]['lr_multiplier'] == pytest.approx(0.05)
    assert steps[19]['lr_multiplier'] == 1.0 and steps[-1]['lr_multiplier'] == pytest.approx(0.0, abs=1e-15)
    after = [s['lr_multiplier'] for s in steps[19:]]
    assert all(a >= b for a, b in zip(after, after[1:]))
    assert [s['masks'] for s in steps].count('trainable') == 960
    assert all(s['masks'] == 'frozen-binary' for s in steps[960:])
    assert plan['checkpoints'] == [0, 8, 16, 32, 64, 128, 256, 512, 768, 960, 1024]
    assert plan['validation_steps'] == list(range(16, 1025, 16)) and len(plan['validation_steps']) == 64
    assert plan['settle_steps'][0] == 256 and plan['settle_steps'][-1] == 960
    assert plan['cohort150_steps'] == [64, 1024]
    assert plan['read_init_seeds'] == list(range(42, 70))
    assert plan['optimizer']['reads']['lr'] == 3e-3 and plan['optimizer']['masks']['lr'] == 0.1


def test_arms_share_one_plan_except_rank_and_read_lr():
    record, ids, gold, r1 = _plan()
    r0 = sp.build_plan(record, 'sparse-gold-sft-r0', ids, gold, inputs_sha256={'training_items_sha256': 'x'})
    assert r1['steps'] == r0['steps'] and r1['settings'] == r0['settings']
    assert (r0['rank'], r0['optimizer']['reads']['lr']) == (0, 3e-4)
    assert r0['trainable_parameters'] == {'reads': 770_703_360, 'log_alpha': 250_880}
    assert r1['trainable_parameters'] == {'reads': 587_776, 'log_alpha': 250_880}
    assert r1['plan_sha256'] != r0['plan_sha256']


def test_plan_round_trip_and_refusals(tmp_path):
    record, _, _, plan = _plan()
    path = sp.write_plan(plan, tmp_path / 'inputs/plan.json')
    assert sp.load_plan(path, record, 'sparse-gold-sft-r1')['plan_sha256'] == plan['plan_sha256']
    with pytest.raises(LaunchRefused, match='another arm'):
        sp.load_plan(path, record, 'sparse-gold-sft-r0')
    other = sp.validate(_record(**{'seeds.mask_noise': 7}), code_commit='c0ffee')
    with pytest.raises(LaunchRefused, match='another arm or launch record'):
        sp.load_plan(path, other, 'sparse-gold-sft-r1')
    tampered = json.loads(path.read_text())
    tampered['steps'][100]['ids'][0], tampered['steps'][100]['ids'][1] = (tampered['steps'][100]['ids'][1],
                                                                        tampered['steps'][100]['ids'][0])
    path.write_text(json.dumps(tampered))
    with pytest.raises(LaunchRefused, match='hash'):
        sp.load_plan(path, record, 'sparse-gold-sft-r1')
    tampered['plan_sha256'] = sp.plan_sha256(tampered)   # rehashed, but not what the record builds
    path.write_text(json.dumps(tampered))
    with pytest.raises(LaunchRefused, match='differs'):
        sp.load_plan(path, record, 'sparse-gold-sft-r1')


def test_plan_from_gold_files_and_cli(tmp_path):
    ids, gold = _gold()
    items = tmp_path / 'training.items.json'
    items.write_text(json.dumps([{'id': i, 'input_ids': [1], 'labels': [1]} for i in ids]))
    order = tmp_path / 'order.json'
    order.write_text(json.dumps(gold))
    sealed = _seal(tmp_path, _signed())
    out = tmp_path / 'plan.json'
    assert sp.main(['plan', '--record', str(sealed), '--code-commit', 'c0ffee', '--arm', 'sparse-gold-sft-r0',
                    '--training-items', str(items), '--gold-order', str(order), '--out', str(out)]) == 0
    record = sp.load_launch_record(sealed, code_commit='c0ffee')
    plan = sp.load_plan(out, record, 'sparse-gold-sft-r0')
    assert set(plan['inputs']) >= {'training_items_sha256', 'gold_order_sha256'}


def test_gold_order_must_be_a_permutation_and_reconstruction_is_reported():
    ids, gold = _gold()
    with pytest.raises(ValueError, match='permutation'):
        sp.full_order(ids, gold[:-1] + [ids[0]], steps=128, examples_per_step=8, base_seed=42)
    swapped = list(gold)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert sp.gold_reconstruction(ids, swapped, 42) == 'differs'
    order = sp.full_order(ids, swapped, steps=64, examples_per_step=8, base_seed=42)
    assert [i for s in order for i in s] == swapped     # the saved file governs, not the reconstruction


def test_lr_multiplier_bounds():
    assert sp.lr_multiplier(1, warmup=20, total=1024) == 0.05
    assert sp.lr_multiplier(522, warmup=20, total=1024) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        sp.lr_multiplier(0, warmup=20, total=1024)
    with pytest.raises(ValueError):
        sp.lr_multiplier(1025, warmup=20, total=1024)
