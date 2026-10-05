"""Production workflow at the agreed public interface, using real saved evidence."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.activation_prepare import PROJECTIONS, VIEWS, load_prepared
from test_activation_validation import prepared
from test_activation_profile import ProfileDependencies, ProfileEngine


@pytest.fixture(scope='module')
def frozen_source(tmp_path_factory):
    from pilot_eval.activation_profile import profile_activation, freeze_execution
    root = tmp_path_factory.mktemp('frozen-measurement')
    path = prepared(root)
    profile = profile_activation(path, root, dependencies=ProfileDependencies())
    execution = freeze_execution(path, root, profile=profile['profile_path'], batch_size=16,
                                  name='production', review_notes='Reviewed fixture evidence.')
    return root, execution.relative_to(root)


@pytest.fixture
def execution(tmp_path, frozen_source):
    root, relative = frozen_source
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    return tmp_path / relative


class MeasurementEngine(ProfileEngine):
    def capture_reference(self, rows):
        result = super().capture_reference(rows)
        self.deps.live_references += 1
        self.deps.max_live_references = max(self.deps.max_live_references, self.deps.live_references)
        return result

    def release_reference(self, reference):
        self.deps.live_references -= 1

    def summarize_reference(self, rows):
        counts = np.array([[r['counts'][v] for v in VIEWS] for r in rows], dtype=float)
        counts = np.repeat(counts[..., None], 28, axis=2)
        return {'block_count': counts,
                'block_base_norm_sum': counts * 2 + self.deps.reference_shift * (counts > 0),
                'block_base_sum': np.repeat(counts[..., None], 4, axis=3)}

    def measure(self, checkpoint, rows, *, step):
        self.deps.measure_calls.append((tuple(r['id'] for r in rows), step))
        if len(self.deps.measure_calls) == self.deps.fail_at:
            raise self.deps.exception('controlled interruption')
        result = super().measure(checkpoint, rows, step=step)
        result['validation'].update(
            sample_limit=16, positions=[],
            block_position='decoder-block-output-before-final-model-norm',
            block_hooks=[f'model.layers.{i}' for i in range(28)],
            module_hooks=[f"model.layers.{i}.{'self_attn' if p in PROJECTIONS[:4] else 'mlp'}.{p}"
                          for i in range(28) for p in PROJECTIONS],
            thresholds={'atol': 1e-6, 'rtol': 1e-5, 'rounding_factor': 4})
        for view in VIEWS:
            selected = [[i for i, flag in enumerate(row['masks'][view]) if flag] for row in rows]
            positions = [[rows[b]['id'], indices[offset]]
                         for offset in range(max(map(len, selected), default=0))
                         for b, indices in enumerate(selected) if offset < len(indices)][:16]
            if positions:
                result['validation']['positions'].extend([
                    {'layer': layer, 'projection': projection, 'view': view, 'positions': positions,
                     'branch_max_error': 0., 'subtraction_max_error': 0.,
                     'branch_max_fraction': 0., 'subtraction_max_fraction': 0.,
                     'coordinate_count': len(positions) * 4,
                     'below_resolution_coordinates': len(positions) * 4 if step == 0 else 0}
                    for layer in range(28) for projection in PROJECTIONS])
        if self.deps.fault == 'nonzero-init' and step == 0:
            result['arrays']['block_delta_sum'][0, 0, 0, 0] = 1e-20
        if step == 8:
            if self.deps.fault == 'nonfinite':
                result['arrays']['block_delta_sum'][0, 0, 0, 0] = np.nan
            if self.deps.fault == 'wrong-hooks':
                result['validation']['module_hooks'].pop()
            if self.deps.fault == 'rank-evidence':
                result['validation']['positions'][0]['branch_max_fraction'] = 2.
        return result


class MeasurementDependencies(ProfileDependencies):
    def __init__(self, *, fail_at=None, exception=RuntimeError, reference_shift=0, fault=None):
        super().__init__()
        self.fail_at, self.exception, self.reference_shift = fail_at, exception, reference_shift
        self.fault = fault
        self.measure_calls = []
        self.engine_loads = 0
        self.live_references = self.max_live_references = 0

    def activation_engine(self, config, root):
        self.engine_loads += 1
        return MeasurementEngine(self)


def run_dir(config, root):
    return root / json.loads(config.read_text())['run_path']


def test_full_run_saves_one_baseline_per_batch_all_checkpoints_and_verified_totals(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation, verify_measurement
    deps = MeasurementDependencies()
    result = measure_activation(execution, tmp_path, dependencies=deps)
    directory = run_dir(execution, tmp_path)
    assert result['measurement_complete'] is True
    assert result['reported'] is False
    assert result['example_count'] == 300
    assert result['corpus_counts'] == {'gsm8k': 150, 'fineweb': 150}
    assert result['completed_combinations'] == 95
    assert len(deps.batch_calls) == 19
    assert len(deps.measure_calls) == 95
    assert deps.engine_loads == deps.closed == 1
    assert deps.max_live_references == 1
    assert deps.live_references == 0
    assert len(list((directory / 'batches').glob('*/baseline/complete.json'))) == 19
    assert len(list((directory / 'batches').glob('*/checkpoints/step-*/complete.json'))) == 95
    with np.load(directory / 'results/aggregate-sums.npz', allow_pickle=False) as totals:
        assert totals['baseline_block_count'][2, 0] == 19200
        assert totals['step64_module_delta_norm_sum'][2, 0, 0] == 1228800
    assert verify_measurement(execution, tmp_path) == result
    calls = len(deps.measure_calls)
    assert measure_activation(execution, tmp_path, dependencies=deps) == result
    assert len(deps.measure_calls) == calls
    assert deps.engine_loads == 1


def test_interruption_resumes_only_missing_units_and_matches_uninterrupted(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation
    failing = MeasurementDependencies(fail_at=3, exception=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt, match='controlled'):
        measure_activation(execution, tmp_path, dependencies=failing)
    directory = run_dir(execution, tmp_path)
    assert failing.closed == 1
    assert failing.live_references == 0
    assert not (directory / 'complete.json').exists()
    progress = json.loads((directory / 'checkpoints/progress.json').read_text())
    assert progress['completed_units'] == [[0, 0], [0, 8]]
    baseline_marker = directory / 'batches/batch-000000/baseline/complete.json'
    baseline_bytes = baseline_marker.read_bytes()
    # Unmarked leftovers are not trusted. They can be replaced, marked siblings cannot.
    incomplete = directory / 'batches/batch-000000/checkpoints/step-16'
    incomplete.mkdir(parents=True, exist_ok=True)
    (incomplete / 'summaries.npz').write_bytes(b'half-written')
    resumed = MeasurementDependencies()
    result = measure_activation(execution, tmp_path, dependencies=resumed)
    assert result['completed_combinations'] == 95
    assert len(resumed.measure_calls) == 93
    assert resumed.measure_calls[0][1] == 16
    assert baseline_marker.read_bytes() == baseline_bytes
    source = tmp_path / json.loads(execution.read_text())['prepared_path']
    from pilot_eval.activation_profile import freeze_execution
    manifest = freeze_execution(source, tmp_path,
        profile=json.loads(execution.read_text())['profile_path'], batch_size=16,
        name='uninterrupted', review_notes='Matched fixture run.')
    measure_activation(manifest, tmp_path, dependencies=MeasurementDependencies())
    other = run_dir(manifest, tmp_path)
    with np.load(directory / 'results/aggregate-sums.npz', allow_pickle=False) as a, np.load(
            other / 'results/aggregate-sums.npz', allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    for payload in (directory / 'batches').rglob('summaries.npz'):
        with np.load(payload, allow_pickle=False) as a, np.load(other / payload.relative_to(directory), allow_pickle=False) as b:
            assert all(np.array_equal(a[k], b[k]) for k in a.files)


def test_public_commands_measure_and_verify_without_loading_model_for_cpu_check(execution, tmp_path, capsys):
    from pilot_eval.cli import main
    args = ['--config', str(execution), '--output-root', str(tmp_path)]
    deps = MeasurementDependencies()
    assert main(['activation-measure', *args], dependencies=deps) == 0
    assert deps.engine_loads == 1
    assert main(['activation-verify', *args]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['measurement_complete'] is True


@pytest.mark.parametrize('fault,message,kept', [
    ('nonfinite', 'nonfinite', 1), ('nonzero-init', 'nonzero', 0),
    ('wrong-hooks', 'hook identity', 1), ('rank-evidence', 'rank-1', 1)])
def test_production_rejects_bad_measurement_evidence_without_fallback(execution, tmp_path, fault, message, kept):
    from pilot_eval.activation_measurement import measure_activation
    deps = MeasurementDependencies(fault=fault)
    with pytest.raises(ValueError, match=message):
        measure_activation(execution, tmp_path, dependencies=deps)
    directory = run_dir(execution, tmp_path)
    assert not (directory / 'complete.json').exists()
    assert deps.closed == 1
    progress = json.loads((directory / 'checkpoints/progress.json').read_text())
    assert progress['completed'] == kept
    assert len(list((directory / 'batches').glob('*/checkpoints/*/complete.json'))) == kept
    assert all(len(ids) == 16 for ids in deps.batch_calls)
    errors = [json.loads(line) for line in (directory / 'logs/errors.jsonl').read_text().splitlines()]
    assert errors[-1]['unit'] == {'batch_index': 0, 'step': 0 if fault == 'nonzero-init' else 8}


class ObservedProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
    def poll(self):
        return self.returncode


def test_monitor_uses_process_liveness_and_tolerates_transient_status(tmp_path):
    from pilot_eval.activation_measurement import monitor_activation_process
    process = ObservedProcess()
    snapshot = monitor_activation_process(process, tmp_path)
    assert snapshot['process_state'] == 'running'
    assert snapshot['snapshot'] is None
    status = tmp_path / 'meta/status.json'
    status.parent.mkdir()
    status.write_text('')
    snapshot = monitor_activation_process(process, tmp_path)
    assert snapshot['process_state'] == 'running'
    assert snapshot['status_error']
    status.write_text(json.dumps({'state': 'running', 'completed': 2, 'total': 95}))
    good = monitor_activation_process(process, tmp_path)
    assert good['snapshot']['completed'] == 2
    status.write_text('{"state":')
    recovered = monitor_activation_process(process, tmp_path, last_snapshot=good['snapshot'])
    assert recovered['snapshot']['completed'] == 2
    assert recovered['snapshot_is_current'] is False
    # A stale status file cannot override a living child, or prove its success.
    status.write_text(json.dumps({'state': 'completed', 'completed': 95, 'total': 95}))
    assert monitor_activation_process(process, tmp_path)['process_state'] == 'running'
    process.returncode = 1
    assert monitor_activation_process(process, tmp_path)['process_state'] == 'exited-with-error'
    process.returncode = 0
    finished = monitor_activation_process(process, tmp_path)
    assert finished['process_state'] == 'exited-successfully'
    assert finished['completion_verified'] is False


def test_production_oom_keeps_verified_units_and_same_batch_can_resume(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation
    from test_activation_profile import FixtureOOM
    deps = MeasurementDependencies(fail_at=3, exception=FixtureOOM)
    with pytest.raises(FixtureOOM):
        measure_activation(execution, tmp_path, dependencies=deps)
    directory = run_dir(execution, tmp_path)
    assert deps.live_references == 0
    assert json.loads((directory / 'meta/status.json').read_text())['completed'] == 2
    assert not (directory / 'complete.json').exists()
    saved_config = execution.read_bytes()
    recovered = MeasurementDependencies()
    assert measure_activation(execution, tmp_path, dependencies=recovered)['measurement_complete']
    assert recovered.measure_calls[0][1] == 16
    assert len(recovered.measure_calls) == 93
    assert execution.read_bytes() == saved_config


def test_reference_mismatch_or_corrupt_marked_shard_stops_before_new_unit(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation
    with pytest.raises(RuntimeError):
        measure_activation(execution, tmp_path, dependencies=MeasurementDependencies(fail_at=3))
    directory = run_dir(execution, tmp_path)
    first = directory / 'batches/batch-000000/checkpoints/step-0/complete.json'
    marker = first.read_bytes()
    shifted = MeasurementDependencies(reference_shift=1)
    with pytest.raises(ValueError, match='recreated reference'):
        measure_activation(execution, tmp_path, dependencies=shifted)
    assert shifted.measure_calls == []
    assert shifted.live_references == 0
    assert first.read_bytes() == marker
    payload = first.parent / 'summaries.npz'
    payload.write_bytes(b'corrupt marked evidence')
    clean = MeasurementDependencies()
    with pytest.raises(ValueError, match='marked shard'):
        measure_activation(execution, tmp_path, dependencies=clean)
    assert clean.engine_loads == 0
    assert payload.read_bytes() == b'corrupt marked evidence'
    assert first.read_bytes() == marker


def test_runtime_mismatch_and_competing_writer_do_not_run_inference(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation
    from pilot_eval.training import run_lock
    deps = MeasurementDependencies()
    deps.runtime = lambda: {'device': 'different-numerical-runtime'}
    with pytest.raises(ValueError, match='runtime mismatch'):
        measure_activation(execution, tmp_path, dependencies=deps)
    assert deps.engine_loads == 0
    directory = run_dir(execution, tmp_path)
    status = directory / 'meta/status.json'
    before = status.read_bytes()
    clean = MeasurementDependencies()
    with run_lock(directory), pytest.raises(RuntimeError, match='another process'):
        measure_activation(execution, tmp_path, dependencies=clean)
    assert clean.engine_loads == 0
    assert status.read_bytes() == before
    config = json.loads(execution.read_text())
    config['batches'][0]['example_ids'].reverse()
    execution.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='membership'):
        measure_activation(execution, tmp_path, dependencies=clean)
    assert clean.engine_loads == 0


def test_final_check_rejects_missing_completion_marker(execution, tmp_path):
    from pilot_eval.activation_measurement import measure_activation, verify_measurement
    measure_activation(execution, tmp_path, dependencies=MeasurementDependencies())
    directory = run_dir(execution, tmp_path)
    (directory / 'batches/batch-000018/checkpoints/step-64/complete.json').unlink()
    with pytest.raises(ValueError, match='missing a required'):
        verify_measurement(execution, tmp_path)
