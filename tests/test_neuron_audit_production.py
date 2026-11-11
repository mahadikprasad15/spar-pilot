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


def test_resume_on_a_new_pod_restores_mirrored_units_before_packaging(world, monkeypatch):
    plan, config = setup(world)
    hub = FakeHub()
    original = prod.Production._package_arm
    state = dict(stop=True)

    def crash_before_packaging(self, arm, st, keys):
        if state['stop']:
            raise RuntimeError('pod lost before packaging (simulated)')
        return original(self, arm, st, keys)

    monkeypatch.setattr(prod.Production, '_package_arm', crash_before_packaging)
    with pytest.raises(RuntimeError, match='pod lost'):
        prod.Production(plan, config, world['backend'], client=hub, now=FROZEN).run()
    import shutil
    shutil.rmtree(Path(config['local_root']))  # a new Pod: empty container disk, only the volume survives
    state['stop'] = False
    calls = dict(n=0)
    collect = nc.NeuronCollector.collect

    def counted(self, rows, **kw):
        calls['n'] += 1
        return collect(self, rows, **kw)

    monkeypatch.setattr(nc.NeuronCollector, 'collect', counted)
    result = prod.Production(plan, config, world['backend'], client=hub, now=FROZEN).run()
    first_arm = result['arms'][0]
    batches = len(json.loads((Path(world['config']['output_root']) / q.RUN_SUBDIR /
                              'q1/stages/batch_freeze/batch_manifest.json').read_text())['batches'])
    assert result['state'] == 'complete' and result['safe_to_stop_pod'] is True
    assert calls['n'] == 2 * batches  # the first arm came back from the volume, not recollected
    record = json.loads((Path(config['volume_root']) / prod.RUN_SUBDIR / 'p1/meta/progress.json').read_text())
    assert len(record['arm_packages'][first_arm]['units']) == batches  # exact, never a smaller package


def test_reused_run_id_with_a_different_context_is_refused(world):
    plan, config = setup(world)
    prod.Production(plan, config, world['backend'], client=FakeHub(), now=FROZEN).run()
    with pytest.raises(prod.ProductionRefused, match='different context'):
        prod.Production(plan, dict(config, label='other'), world['backend'], client=FakeHub(), now=FROZEN).run()


def test_changed_inputs_or_runtime_after_qualification_are_refused(world, monkeypatch):
    plan, config = setup(world)
    adapter = Path(config['adapter_dirs']['gold-sft']) / 'adapter_model.safetensors'
    original = adapter.read_bytes()
    adapter.write_bytes(original + b'\0')
    with pytest.raises(prod.ProductionRefused, match='adapter for gold-sft'):
        prod.Production(plan, config, world['backend'], client=FakeHub())
    adapter.write_bytes(original)
    monkeypatch.setattr(type(world['backend']), 'device_info', lambda self: dict(name='other-gpu', total_memory=1))
    with pytest.raises(prod.ProductionRefused, match='runtime differs'):
        prod.Production(plan, config, world['backend'], client=FakeHub())


def test_changed_sequences_bytes_are_refused(world):
    plan, config = setup(world)
    seq = world['root'] / 'seq' / 'sequences.jsonl'
    seq.write_text(seq.read_text() + '\n')
    with pytest.raises(prod.ProductionRefused, match='failed its seal|differs'):
        prod.Production(plan, config, world['backend'], client=FakeHub())


def test_wall_cap_is_cumulative_across_restarts_and_checked_per_unit(world, monkeypatch):
    plan, config = setup(world, cap=1.0)  # 3600 s
    clock = dict(t=0.0)
    hub = FakeHub()

    def now():
        clock['t'] += 400.0
        return clock['t']

    original = nc.NeuronCollector.collect
    calls = dict(n=0)

    def flaky(self, rows, **kw):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('worker killed (simulated)')
        return original(self, rows, **kw)

    monkeypatch.setattr(nc.NeuronCollector, 'collect', flaky)
    with pytest.raises(RuntimeError, match='killed'):
        prod.Production(plan, config, world['backend'], client=hub, now=now).run()
    saved = json.loads((Path(config['volume_root']) / prod.RUN_SUBDIR / 'p1/meta/progress.json').read_text())
    assert saved['wall_seconds_saved'] > 0
    result = prod.Production(plan, config, world['backend'], client=hub, now=now).run()
    assert result['state'] == 'stopped-overrun'  # the second session inherits the first session's time
    events = json.loads((Path(config['volume_root']) / prod.RUN_SUBDIR / 'p1/meta/progress.json').read_text())['events']
    assert any(e['kind'] == 'overrun-stop' and 'cumulative' in e['reason'] for e in events)


def test_optional_stop_after_verified_partial_backup(world):
    plan, config = setup(world, cap=1e-9)
    calls = []
    runner = lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0)  # noqa: E731
    config['stop_command'] = ['runpodctl', 'stop', 'pod', 'X']
    ticks = iter(range(10 ** 6))
    clock = lambda: float(next(ticks))  # noqa: E731
    result = prod.Production(plan, config, world['backend'], client=FakeHub(), now=clock, stop_runner=runner).run()
    assert result['state'] == 'stopped-overrun' and calls == []  # default: never stop without the final backup
    config2 = dict(config, run_id='p2', stop_after_verified_partial_backup=True)
    prod.Production(plan, config2, world['backend'], client=FakeHub(), now=clock, stop_runner=runner).run()
    assert calls == [['runpodctl', 'stop', 'pod', 'X']]


def test_worker_uses_an_online_child_process_for_hub_calls(world):
    plan, config = setup(world)
    production = prod.Production(plan, config, world['backend'])
    assert isinstance(production.client, prod.ex.OnlineHubProcess)
