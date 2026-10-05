"""Independent numerical examples and verified CPU reporting boundary."""
import numpy as np
import pytest


def test_bootstrap_recomputes_ratio_and_cancellation():
    from pilot_eval.activation_report import bootstrap_measurement
    # Two examples: lengths 1 and 3, vector sums +2 and -2; norms 2 and 6.
    summary = dict(count=np.array([1., 3.]), base_norm_sum=np.array([2., 6.]),
                   delta_norm_sum=np.array([2., 2.]), base_sum=np.array([[2.], [6.]]),
                   delta_sum=np.array([[2.], [-2.]]))
    draws = np.array([[0, 1], [0, 0], [1, 1]])
    token = bootstrap_measurement(summary, 'token', draws)
    equal = bootstrap_measurement(summary, 'example', draws)
    np.testing.assert_allclose(token['replicates'], [0., 1., 1/3])
    np.testing.assert_allclose(equal['replicates'], [1/3, 1., 1/3])
    assert token['defined_replicates'] == 3


def test_undefined_bootstrap_is_not_zero():
    from pilot_eval.activation_report import bootstrap_measurement
    summary = dict(count=np.array([1., 1.]), base_norm_sum=np.array([0., 2.]),
                   delta_norm_sum=np.array([0., 1.]), ratio_sum=np.array([0., .5]),
                   defined_count=np.array([0., 1.]))
    result = bootstrap_measurement(summary, 'token', np.array([[0, 0], [0, 1], [1, 1]]))
    assert np.isnan(result['replicates'][0])
    assert result['defined_replicates'] == 2
    assert result['interval_95'] == [.5, .5]
    assert result['undefined_replicates'] == 1

from test_activation_measurement import frozen_source, MeasurementDependencies, MeasurementEngine


class UndefinedEngine(MeasurementEngine):
    def measure(self, checkpoint, rows, *, step):
        result = super().measure(checkpoint, rows, step=step)
        for key in ['module_base_norm_sum', 'module_defined_count', 'module_ratio_sum']:
            result['arrays'][key] = result['arrays'][key].copy()
            result['arrays'][key][:, 2, 0, 0] = 0
        return result


class ReportDependencies(MeasurementDependencies):
    def activation_engine(self, config, root):
        self.engine_loads += 1
        return UndefinedEngine(self)


@pytest.fixture(scope='module')
def completed_report_source(frozen_source):
    from pilot_eval.activation_measurement import measure_activation
    root, relative = frozen_source
    path = root / relative
    measure_activation(path, root, dependencies=ReportDependencies())
    return root, path


def test_report_reconstructs_and_reuses_verified_cpu_evidence(completed_report_source):
    import json
    from pilot_eval.activation_report import report_activation
    root, path = completed_report_source
    result = report_activation(path, root, name='scientific-report')
    assert len(result['measurements']) == 5 * 3 * 28 * 8 * 2
    block = next(r for r in result['measurements'] if r['step'] == 8 and r['view'] == 'question'
                 and r['layer'] == 0 and r['kind'] == 'block' and r['weighting'] == 'token')
    # Fixture base vectors [1,1,1,1] per token, delta [8,8,8,8], base norm 2.
    assert block['relative_write'] == 8
    assert block['interval_95'] == [8., 8.]
    module = next(r for r in result['measurements'] if r['step'] == 8 and r['view'] == 'user'
                  and r['layer'] == 1 and r['projection'] == 'q_proj' and r['weighting'] == 'example')
    assert module['relative_write'] == 4
    assert module['mean_token_ratio'] == 4
    undefined = next(r for r in result['measurements'] if r['step'] == 8 and r['view'] == 'user'
                     and r['layer'] == 0 and r['projection'] == 'q_proj' and r['weighting'] == 'token')
    assert undefined['relative_write'] is None
    assert undefined['interval_95'] is None
    assert undefined['undefined_replicates'] == 2000
    assert undefined['ratio_example_count'] == 0
    assert result['provenance']['prepared']['source_evidence']['base_sha256']
    directory = root / 'reports/scientific-report'
    assert (directory / 'plots/block-depth.png').exists()
    with np.load(directory / 'results/bootstrap-draws.npz', allow_pickle=False) as saved:
        rng = np.random.default_rng(42)
        np.testing.assert_array_equal(saved['gsm8k'], rng.integers(0, 150, (2000, 150)))
        np.testing.assert_array_equal(saved['fineweb'], rng.integers(0, 150, (2000, 150)))
    assert len(result['bootstrap_cohort_ids']['gsm8k']) == 150
    assert all(r['relative_write'] in [None, 0.] for r in result['measurements'] if r['step'] == 0)
    assert json.loads((directory / 'plots/manifest.json').read_text())['heatmap_scale']['min'] == 0
    assert report_activation(path, root, name='scientific-report') == result
    import subprocess, sys, os
    guard = """
import sys
class NoModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'datasets', 'trl'}:
            raise AssertionError('CPU reporting imported model dependencies: ' + fullname)
sys.meta_path.insert(0, NoModels())
from pilot_eval.cli import main
assert main(['activation-report', '--config', sys.argv[1], '--output-root', sys.argv[2], '--name', 'scientific-report']) == 0
"""
    subprocess.run([sys.executable, '-c', guard, str(path), str(root)], check=True, env=os.environ.copy())
    # Corrupt a marked report: do not silently regenerate scientific evidence.
    (directory / 'results/measurements.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='payload hash'):
        report_activation(path, root, name='scientific-report')


def test_partial_source_cannot_produce_report(tmp_path, frozen_source):
    import shutil
    from pilot_eval.activation_report import report_activation
    root, relative = frozen_source
    shutil.copytree(root, tmp_path, dirs_exist_ok=True)
    execution = __import__('json').loads((tmp_path / relative).read_text())
    (tmp_path / execution['run_path'] / 'complete.json').unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        report_activation(tmp_path / relative, tmp_path, name='partial')
    assert not (tmp_path / 'reports/partial/complete.json').exists()
