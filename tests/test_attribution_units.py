"""Ticket 05: attribution units, sealed run directories, budget, export/restore and GPU monitor (CPU)."""
import json
import shutil

import numpy as np
import pytest

from pilot_eval import attribution_gpu as gpu
from pilot_eval import attribution_units as au
from pilot_eval import neuron_audit_export as ex
from tests.test_neuron_audit_export import REPO, FakeHub

IDS = dict(plan_sha256='p' * 64, code_sha256='c' * 64, runtime_sha256='r' * 64)


def blocks(rows, value=1.0):
    return {('examples', 0, 'attr_trace'): np.full((rows, 4), value, dtype=np.float32)}


def write_ab(units, arm, batch, ids, value=1.0, **extra):
    key = au.unit_key('ab', arm, batch=batch)
    identity = units.identity('ab', arm, ids, input_sha256='i' * 64, batch=batch)
    return key, units.write(key, identity=identity, example_ids=ids, blocks=blocks(len(ids), value),
                            counts={'examples': [1] * len(ids)}, results=[dict(id=i) for i in ids],
                            config=dict(batch=batch), log='ok\n', **extra)


def test_unit_keys_per_kind():
    assert au.unit_key('ab', 'gold-sft', batch=3) == 'ab-gold-sft-b00003'
    assert au.unit_key('c', 'gold-sft', condition='top5-trace-1') == 'c-gold-sft-top5-trace-1'
    assert au.unit_key('d', 'untuned', condition='reference', batch=0) == 'd-untuned-reference-b00000'
    for bad in [dict(kind='ab', arm='x'), dict(kind='c', arm='x', batch=1), dict(kind='zz', arm='x', batch=1),
                dict(kind='c', arm='x/y', condition='a')]:
        with pytest.raises(ValueError):
            au.unit_key(bad.pop('kind'), bad.pop('arm'), **bad)


def test_unit_sealed_with_identity_hashes_results_and_resume(tmp_path):
    units = au.Units(tmp_path, **IDS)
    key, state = write_ab(units, 'gold-sft', 0, ['e1', 'e2'])
    assert state == 'written' and units.state(key) == 'sealed'
    arrays, meta = units.read(key)
    assert arrays[('examples', 0, 'attr_trace')].shape == (2, 4)
    assert meta['results'] == [dict(id='e1'), dict(id='e2')] and meta['config'] == dict(batch=0)
    identity = json.loads((units.store.unit_dir(key) / 'index.json').read_text())['identity']
    assert {identity[k] for k in IDS} == set(IDS.values()) and identity['example_ids'] == ['e1', 'e2']
    assert write_ab(units, 'gold-sft', 0, ['e1', 'e2'])[1] == 'reused'  # hash-verified resume
    with pytest.raises(ValueError):  # same key, different identity (different code)
        au.Units(tmp_path, **dict(IDS, code_sha256='d' * 64)).write(
            key, identity=au.Units(tmp_path, **dict(IDS, code_sha256='d' * 64)).identity(
                'ab', 'gold-sft', ['e1', 'e2'], input_sha256='i' * 64, batch=0),
            example_ids=['e1', 'e2'], blocks=blocks(2), counts={'examples': [1, 1]})


def test_duplicate_examples_across_units_refused(tmp_path):
    units = au.Units(tmp_path, **IDS)
    write_ab(units, 'gold-sft', 0, ['e1', 'e2'])
    with pytest.raises(ValueError, match='duplicate'):
        write_ab(units, 'gold-sft', 1, ['e2', 'e3'])
    write_ab(units, 'online-grpo', 0, ['e1', 'e2'])  # other arm: fine


def test_interrupted_and_corrupted_units_detected(tmp_path):
    units = au.Units(tmp_path, **IDS)
    key, _ = write_ab(units, 'gold-sft', 0, ['e1'])
    (units.store.unit_dir('ab-gold-sft-b00001')).mkdir(parents=True)  # interrupted: no seal
    assert units.state('ab-gold-sft-b00001') == 'interrupted'
    raw = units.store.unit_dir(key) / 'blocks.f32'
    raw.write_bytes(b'\x00' * raw.stat().st_size)
    assert au.Units(tmp_path, **IDS).state(key) == 'corrupt'
    manifest = au.Units(tmp_path, **IDS).manifest([key, 'ab-gold-sft-b00001', 'ab-gold-sft-b00002'])
    assert manifest['sealed'] == [] and manifest['corrupt'] == [key]
    assert manifest['interrupted'] == ['ab-gold-sft-b00001'] and manifest['missing'] == ['ab-gold-sft-b00002']
    assert manifest['completeness'] == 'partial'


def test_sealed_run_directory_and_refusal_to_rewrite(tmp_path):
    run = tmp_path / 'attribution/freeze'
    assert au.seal_run(run, {'freeze.json': dict(top=[1, 2]), 'notes.md': 'x\n'}) == 'written'
    assert au.verify_run(run) == {'freeze.json', 'notes.md'}
    assert au.seal_run(run, {'freeze.json': dict(top=[1, 2]), 'notes.md': 'x\n'}) == 'reused'
    with pytest.raises(ValueError):
        au.seal_run(run, {'freeze.json': dict(top=[2, 1]), 'notes.md': 'x\n'})
    (run / 'notes.md').write_text('tampered\n')
    with pytest.raises(ValueError):
        au.verify_run(run)
    with pytest.raises(ValueError):
        au.seal_run(tmp_path / 'other', {'../escape.json': {}})


def test_budget_hard_caps_at_boundaries_persisted_across_restart(tmp_path):
    clock = [0.0]
    path = tmp_path / 'budget.json'
    budget = au.Budget(path, stage_caps=dict(attribution=100, stage_c=50), total_cap=120, now=lambda: clock[0])
    budget.start('attribution')
    clock[0] = 60
    assert budget.boundary('attribution')['remaining_total'] == pytest.approx(60)
    # restart: a fresh object resumes the persisted elapsed budget, not a fresh one
    clock[0] = 1000
    budget = au.Budget(path, stage_caps=dict(attribution=100, stage_c=50), total_cap=120, now=lambda: clock[0])
    assert budget.elapsed('attribution') == pytest.approx(60)
    budget.start('attribution')
    clock[0] = 1045
    with pytest.raises(au.CapReached, match='attribution'):
        budget.boundary('attribution')
    budget.start('stage_c')
    budget.charge('export', 20)  # export time counts toward the total
    with pytest.raises(au.CapReached, match='total'):
        budget.boundary('stage_c')
    with pytest.raises(ValueError):  # caps must be fixed numbers
        au.Budget(tmp_path / 'b2.json', stage_caps=dict(attribution=None), total_cap=1)
    with pytest.raises(ValueError):  # a different cap set cannot reuse a persisted budget
        au.Budget(path, stage_caps=dict(attribution=999), total_cap=120)


def test_longest_first_projection_is_advisory_and_does_not_stop_a_healthy_run(tmp_path):
    clock = [0.0]
    budget = au.Budget(tmp_path / 'b.json', stage_caps=dict(attribution=100), total_cap=100, now=lambda: clock[0])
    budget.start('attribution')
    durations = [30, 25, 10, 5, 5, 5, 4, 3, 2, 1]  # longest batches first: 90 s total
    projections = []
    for done, seconds in enumerate(durations, 1):
        clock[0] += seconds
        budget.boundary('attribution')
        projections.append(budget.projection('attribution', done=done, total=len(durations)))
    assert projections[0]['projected_seconds'] > 100 and projections[0]['advisory'] is True
    assert budget.elapsed('attribution') == pytest.approx(90)


def test_summary_only_fixture_is_exported_and_restored_with_missing_work(tmp_path):
    root, clean = tmp_path / 'root', tmp_path / 'clean'
    au.write_summary(root, summary=dict(stages=dict(attribution='cap-reached')), markdown='# Morning summary\n',
                     missing_work=['ab-gold-sft-b00004'])
    hub = FakeHub()
    receipt = au.export(root, label='rehearsal-x', include=['attribution'], expected_units=[],
                        expected_runs=[au.SUMMARY_DIR], client=hub, temp_dir=tmp_path / 'tmp', on_event=lambda e: None)
    assert receipt['completeness'] == 'complete'
    restored = au.restore_summary(receipt, clean, client=hub, on_event=lambda e: None)
    assert restored['summary']['stages']['attribution'] == 'cap-reached'
    assert restored['markdown'].startswith('# Morning summary') and restored['missing_work'] == ['ab-gold-sft-b00004']


def test_failed_partial_run_exports_as_partial_and_restores_into_empty_root(tmp_path):
    root, clean = tmp_path / 'root', tmp_path / 'clean'
    units = au.Units(root, **IDS)
    key, _ = write_ab(units, 'gold-sft', 0, ['e1'])
    expected = [key, 'ab-gold-sft-b00001']
    au.seal_run(root / 'attribution/launch', {'launch_record.json': dict(run_id='r1')})
    au.write_summary(root, summary=dict(state='failed'), markdown='failed\n', missing_work=['ab-gold-sft-b00001'])
    hub = FakeHub()
    receipt = au.export(root, label='r1', include=['attribution'], expected_units=expected,
                        expected_runs=[au.SUMMARY_DIR, 'attribution/launch', 'attribution/freeze'],
                        client=hub, temp_dir=tmp_path / 'tmp', on_event=lambda e: None)
    assert receipt['completeness'] == 'partial'
    snap = json.loads((root / receipt['snapshot_path']).read_text())
    assert snap['missing_expected_units'] == ['ab-gold-sft-b00001']
    assert snap['missing_expected_runs'] == ['attribution/freeze']
    restored = au.restore_summary(receipt, clean, client=hub, on_event=lambda e: None)
    assert restored['proof']['completeness'] == 'partial' and restored['missing_work'] == ['ab-gold-sft-b00001']
    assert au.Units(clean, **IDS).state(key) == 'sealed'
    with pytest.raises(ValueError):  # restore target must be empty
        au.restore_summary(receipt, clean, client=hub, on_event=lambda e: None)


def test_snapshot_expected_runs_is_backward_compatible(tmp_path):
    root = tmp_path / 'root'
    au.seal_run(root / 'x/run', {'a.json': {}})
    snap = ex.snapshot(root, ['x'], 'lbl')
    assert snap['completeness'] == 'complete' and snap['missing_expected_runs'] == []


# --- GPU monitor -------------------------------------------------------------------------------

class FakeNvml:
    def __init__(self):
        self.calls = 0

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUtilizationRates(self, handle):
        self.calls += 1
        return type('U', (), dict(gpu=80 + self.calls % 3))()

    def nvmlDeviceGetMemoryInfo(self, handle):
        return type('M', (), dict(used=40, total=80))()

    def nvmlDeviceGetPowerUsage(self, handle):
        return 300_000


def test_nvml_summary_needs_thirty_seconds_of_samples():
    sampler = gpu.NvmlSampler(nvml=FakeNvml())
    for t in range(10):
        sampler.sample(now=float(t))
    short = sampler.summary()
    assert short['available'] is False and 'fewer than 30 s' in short['reason']
    for t in range(10, 40):
        sampler.sample(now=float(t))
    full = sampler.summary()
    assert full['available'] is True and full['samples'] == 40
    assert 80 <= full['median_busy_percent'] <= 82 and full['peak_memory_fraction'] == pytest.approx(0.5)
    assert full['median_power_watts'] == pytest.approx(300.0) and full['targets_are_gates'] is False


def test_nvml_unavailable_on_cpu_is_recorded_not_fatal():
    sampler = gpu.NvmlSampler(nvml=None)
    sampler.start()
    sampler.stop()
    assert sampler.summary()['available'] is False


def test_throughput_padding_and_mfu_flop_model():
    assert gpu.padding_fraction([3, 5], padded_width=5) == pytest.approx(0.2)
    timing = gpu.throughput(tokens=1000, seconds=2.0)
    assert timing['tokens_per_second'] == 500
    est = gpu.mfu(parameters=10**9, tokens=1000, seconds=1.0, peak_flops=1e14, backward_passes=2)
    assert est['flops'] == pytest.approx(2e9 * 1000 * 3)  # forward 2N + two activation backwards 2N each
    assert est['mfu'] == pytest.approx(6e12 / 1e14) and 'attention_scores' in est['excludes']
    with pytest.raises(ValueError):
        gpu.mfu(parameters=1, tokens=1, seconds=0, peak_flops=1)


def test_synchronized_timer_calls_sync_on_both_sides():
    calls = []
    clock = iter([1.0, 4.0])
    with gpu.synchronized(lambda: calls.append('sync'), now=lambda: next(clock)) as timer:
        calls.append('work')
    assert calls == ['sync', 'work', 'sync'] and timer['seconds'] == pytest.approx(3.0)


def test_profiler_trace_hook_writes_a_trace(tmp_path):
    torch = pytest.importorskip('torch', exc_type=ImportError)
    path = tmp_path / 'trace.json'
    with gpu.profiler_trace(path):
        torch.ones(4) @ torch.ones(4)
    assert path.is_file() and path.stat().st_size > 0
