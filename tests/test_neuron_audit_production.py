"""Ticket 04 on the tiny CPU world: preconditions, per-arm verified packages, mirror, resume, overrun, stop gate."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from test_neuron_audit_qualify import FakeBackend, qualification, world  # noqa: E402,F401  (fixture reuse)

from pilot_eval import neuron_audit_collector as nc  # noqa: E402
from pilot_eval import neuron_audit_plan as plan_mod  # noqa: E402
from pilot_eval import neuron_audit_production as prod  # noqa: E402
from pilot_eval import neuron_audit_qualify as q  # noqa: E402
from pilot_eval.neuron_audit_blocks import BlockStore  # noqa: E402


FROZEN = lambda: 0.0  # noqa: E731  timing-independent clock for tests not about the overrun rule


class FakeHub:
    cache_root = None

    def __init__(self):
        self.store, self.publishes = {}, 0

    def info(self, repo):
        return SimpleNamespace(private=True)

    def publish(self, repo, files):
        self.publishes += 1
        revision = hashlib.sha1(str(self.publishes).encode()).hexdigest()
        self.store[revision] = {n: Path(p).read_bytes() for n, p in files.items()}
        return revision

    def fetch_to(self, repo, name, revision, directory):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.store[revision][name])
        return path


def setup(world, *, signoff='accept', cap=1.0, edit=None):
    seq = world['root'] / 'seq'
    (seq / 'manifest.json').write_text(json.dumps(dict(
        sequences_sha256=hashlib.sha256((seq / 'sequences.jsonl').read_bytes()).hexdigest())))
    qualification(world).run()
    qual_dir = Path(world['config']['output_root']) / q.RUN_SUBDIR / 'q1'
    if signoff:
        q.signoff(qual_dir, reviewer='t', decision=signoff, notes='')
    value = copy.deepcopy(world['plan_value'])
    value['operations']['production_wall_time_cap_hours'] = cap
    value['approval']['approved_on'] = '2026-11-12'  # re-approval after the cap: allowed
    if edit:
        edit(value)
    path = world['tmp'] / 'plan-04.yaml'
    path.write_text(yaml.safe_dump(value))
    plan = plan_mod.load(path, '04')
    config = dict(world['config'], run_id='p1', local_root=str(world['tmp'] / 'local'),
                  volume_root=str(world['tmp'] / 'volume'), qualification_root=world['config']['output_root'],
                  qualification_run_id='q1', label='prod', repo='owner/private-audit')
    return plan, config


def test_full_production_packages_each_arm_and_publishes_reference_manifest(world):
    plan, config = setup(world)
    hub = FakeHub()
    calls = []
    config['stop_command'] = ['runpodctl', 'stop', 'pod', 'X']
    result = prod.Production(plan, config, world['backend'], client=hub, now=FROZEN,
                             stop_runner=lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0)).run()
    assert result['state'] == 'complete' and result['safe_to_stop_pod'] is True
    assert result['final_manifest']['completeness'] == 'complete'
    assert sorted(result['final_manifest']['arms']) == ['gold-sft', 'offline-grpo', 'untuned']
    assert calls == [['runpodctl', 'stop', 'pod', 'X']]
    paths = [n for rev in hub.store.values() for n in rev]
    assert sum(n.endswith('package.tar') for n in paths) == 3
    assert all(n.startswith('neuron-audit/prod/') for n in paths)
    final_name = result['final_manifest']['path']
    final_bytes = next(rev[final_name] for rev in hub.store.values() if final_name in rev)
    assert len(final_bytes) < 20_000  # references, not data
    local_units = list((Path(config['local_root']) / prod.RUN_SUBDIR / 'p1/blocks/units').glob('*'))
    assert local_units == []  # pruned after verified upload
    volume = BlockStore(Path(config['volume_root']) / prod.RUN_SUBDIR / 'p1/blocks')
    sealed = [k for k, s in volume.status().items() if s == 'sealed']
    assert sealed and all(volume.verify_unit(k, force=True, use_listing=False) for k in sealed)
    ident = volume.index(sealed[0])['identity']
    assert ident['sequences_sha256'] and ident['rehearsal'] is False
    marker = json.loads((Path(config['local_root']) / prod.RUN_SUBDIR / 'p1/meta/writer.json').read_text())
    assert marker['state'] == 'released' and marker['final_state'] == 'complete'


def test_crash_then_resume_skips_verified_arms_and_mirrored_units(world, monkeypatch):
    plan, config = setup(world)
    hub = FakeHub()
    original = nc.NeuronCollector.collect
    calls = dict(n=0, fail_at=None)

    def flaky(self, rows, **kw):
        calls['n'] += 1
        if calls['fail_at'] and calls['n'] == calls['fail_at']:
            raise RuntimeError('worker killed (simulated)')
        return original(self, rows, **kw)

    monkeypatch.setattr(nc.NeuronCollector, 'collect', flaky)
    batches = len(json.loads((Path(world['config']['output_root']) / q.RUN_SUBDIR /
                              'q1/stages/batch_freeze/batch_manifest.json').read_text())['batches'])
    calls['fail_at'] = batches + 1  # first batch of the second arm
    with pytest.raises(RuntimeError, match='killed'):
        prod.Production(plan, config, world['backend'], client=hub, now=FROZEN).run()
    state = json.loads((Path(config['volume_root']) / prod.RUN_SUBDIR / 'p1/meta/progress.json').read_text())
    assert list(state['arm_packages']) == ['untuned'] and state['final_manifest']['completeness'] == 'partial'
    assert state['safe_to_stop_pod'] is False
    calls.update(n=0, fail_at=None)
    result = prod.Production(plan, config, world['backend'], client=hub, now=FROZEN).run()
    assert result['state'] == 'complete' and calls['n'] == 2 * batches  # untuned not recollected


def test_foreign_writer_marker_stops_for_investigation(world):
    plan, config = setup(world)
    marker = Path(config['volume_root']) / prod.RUN_SUBDIR / 'p1/meta/writer.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(host='other-pod', pid=1, state='running', heartbeat_unix=0)))
    with pytest.raises(prod.ProductionRefused, match='investigate'):
        prod.Production(plan, config, world['backend'], client=FakeHub()).run()


def test_overrun_rule_stops_with_partial_export_and_keeps_pod(world):
    plan, config = setup(world, cap=1e-9)
    clock = dict(t=0.0)

    def now():
        clock['t'] += 100.0
        return clock['t']

    result = prod.Production(plan, config, world['backend'], client=FakeHub(), now=now).run()
    assert result['state'] == 'stopped-overrun' and result['safe_to_stop_pod'] is False
    assert result['final_manifest']['completeness'] == 'partial'


@pytest.mark.parametrize('signoff,edit,match', [
    (None, None, 'sign-off is not recorded'),
    ('reject', None, 'not an accept'),
    ('accept', lambda v: v['statistics'].update(primary_list_budget=10), 'plan content changed'),
])
def test_preconditions_refuse(world, signoff, edit, match):
    plan, config = setup(world, signoff=signoff, edit=edit)
    with pytest.raises(prod.ProductionRefused, match=match):
        prod.Production(plan, config, world['backend'], client=FakeHub())


def test_rehearsal_label_and_limits_are_enforced(world):
    plan, config = setup(world)
    with pytest.raises(prod.ProductionRefused, match='rehearsal'):
        prod.Production(plan, dict(config, run_id='rehearsal-x'), world['backend'], client=FakeHub())
    with pytest.raises(prod.ProductionRefused, match='rehearsals only'):
        prod.Production(plan, dict(config, max_batches_per_arm=1), world['backend'], client=FakeHub()).run()


def test_overrun_check_rule():
    history = dict(seen=set(), consecutive=0, records=[])
    assert prod.overrun_check(0.05, 10, projected_seconds=100, cap_seconds=1000, history=history) is None
    assert prod.overrun_check(0.1, 10, projected_seconds=100, cap_seconds=1000, history=history)['bad'] is False
    assert prod.overrun_check(0.2, 50, projected_seconds=100, cap_seconds=1000, history=history)['bad'] is True
    with pytest.raises(prod.OverrunStop):
        prod.overrun_check(0.3, 90, projected_seconds=100, cap_seconds=1000, history=history)


def test_colab_rehearsal_flow_is_labelled_limited_and_refused_by_selection(world):
    from pilot_eval import neuron_audit_selection as sel
    seq = world['root'] / 'seq'
    (seq / 'manifest.json').write_text(json.dumps(dict(
        sequences_sha256=hashlib.sha256((seq / 'sequences.jsonl').read_bytes()).hexdigest())))
    approved = world['tmp'] / 'approved.yaml'
    approved.write_text(yaml.safe_dump(world['plan_value']))
    rehearsal_path = plan_mod.make_rehearsal_plan(approved, world['tmp'] / 'rehearsal.yaml')
    with pytest.raises(plan_mod.PlanError, match='not rehearsal'):
        plan_mod.load(approved, '03b', rehearsal=True)
    plan = plan_mod.load(rehearsal_path, '03b', rehearsal=True)
    config = dict(world['config'], run_id='rehearsal-q1')
    summary = q.Qualification(plan, config, world['backend'], preflight=world['preflight']).run()
    assert summary['rehearsal'] is True and summary['label'].startswith('REHEARSAL')
    qual_dir = Path(config['output_root']) / q.RUN_SUBDIR / 'rehearsal-q1'
    q.signoff(qual_dir, reviewer='t', decision='accept')
    hub = FakeHub()
    pconfig = dict(world['config'], run_id='rehearsal-p1', local_root=str(world['tmp'] / 'rl'),
                   volume_root=str(world['tmp'] / 'rv'), qualification_root=world['config']['output_root'],
                   qualification_run_id='rehearsal-q1', label='colab', repo='owner/private-audit',
                   max_batches_per_arm=1, rehearsal_cap_hours=1)
    result = prod.Production(plan, pconfig, world['backend'], client=hub, now=FROZEN).run()
    assert result['rehearsal'] is True and result['label'] == 'rehearsal-colab'
    assert all(n.startswith('neuron-audit/rehearsal-colab/') for rev in hub.store.values() for n in rev)
    volume = Path(pconfig['volume_root']) / prod.RUN_SUBDIR / 'rehearsal-p1/blocks'
    rows = [json.loads(l) for l in (seq / 'sequences.jsonl').read_text().splitlines()]
    with pytest.raises(sel.SelectionRefused, match='rehearsal'):
        sel.Discovery([volume], rows, sequences_sha256='x')
    # an approved-plan production run cannot consume the rehearsal qualification
    prod_plan, prod_config = setup(world)
    with pytest.raises(prod.ProductionRefused, match='rehearsal'):
        prod.Production(prod_plan, dict(prod_config, qualification_run_id='rehearsal-q1'), world['backend'],
                        client=FakeHub())
