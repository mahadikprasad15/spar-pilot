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
        return result


class MeasurementDependencies(ProfileDependencies):
    def __init__(self, *, fail_at=None, exception=RuntimeError, reference_shift=0):
        super().__init__()
        self.fail_at, self.exception, self.reference_shift = fail_at, exception, reference_shift
        self.measure_calls = []
        self.engine_loads = 0

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
