"""Public profiling/freezing workflow with controlled hardware boundaries."""
import json
import numpy as np
import pytest
from pilot_eval.activation_prepare import load_prepared
from test_activation_validation import prepared


class FixtureOOM(RuntimeError):
    pass


class ProfileEngine:
    def __init__(self, deps):
        self.deps = deps
    def base_hash(self):
        return 'b' * 64
    def capture_reference(self, rows):
        self.deps.batch_calls.append([r['id'] for r in rows])
        if len(rows) == self.deps.oom:
            raise FixtureOOM('controlled memory limit')
        if len(rows) == self.deps.failure:
            raise ValueError('planted missing hook')
        return rows
    def measure(self, checkpoint, rows, *, step):
        counts = np.array([[r['counts'][v] for v in ['question', 'solution', 'user']]
                           for r in rows], dtype=float)[:, :, None]
        counts = np.repeat(counts, 28, axis=2)
        modules = np.repeat(counts[..., None], 7, axis=3)
        arrays = {'block_count': counts, 'block_base_norm_sum': counts * 2,
                  'block_delta_norm_sum': counts * step,
                  'block_base_sum': np.repeat(counts[..., None], 4, axis=3),
                  'block_delta_sum': np.repeat((counts * step)[..., None], 4, axis=3),
                  'module_count': modules, 'module_defined_count': modules,
                  'module_base_norm_sum': modules * 2, 'module_delta_norm_sum': modules * step,
                  'module_ratio_sum': modules * step / 2}
        if len(rows) == self.deps.drift:
            arrays['block_delta_sum'] += .1
        self.deps.time += 1
        return {'arrays': arrays, 'validation': {'exact_zero': step == 0, 'rank1_passed': True,
                'reference_invariant': True, 'module_count': 196, 'base_sha256': 'b' * 64}}
    def close(self):
        self.deps.closed += 1


class ProfileDependencies:
    def __init__(self, *, oom=None, drift=None, failure=None):
        self.oom, self.drift, self.failure = oom, drift, failure
        self.time = 0
        self.closed = 0
        self.batch_calls = []
    def runtime(self):
        return {'device': 'controlled-CPU-fixture', 'precision': 'float32', 'tf32': False}
    def activation_engine(self, config, root):
        return ProfileEngine(self)
    def synchronize(self):
        pass
    def clock(self):
        return self.time
    def reset_peak_memory(self):
        pass
    def peak_memory(self):
        return {'allocated_bytes': 1000, 'reserved_bytes': 2000}
    def cleanup(self):
        pass
    def is_oom(self, exc):
        return isinstance(exc, FixtureOOM)


def test_profile_uses_same_longest_workload_and_freezes_reviewed_passing_candidate(tmp_path):
    from pilot_eval.activation_profile import profile_activation, freeze_execution
    config = prepared(tmp_path)
    before = config.read_bytes()
    deps = ProfileDependencies()
    result = profile_activation(config, tmp_path, dependencies=deps)
    assert [r['batch_size'] for r in result['measurements']] == [1, 2, 4, 8, 16]
    assert all(r['status'] == 'passed' for r in result['measurements'])
    _, rows = load_prepared(config, tmp_path)
    longest = max((r for r in rows if r['corpus'] == 'gsm8k'), key=lambda r: len(r['input_ids']))
    assert longest['id'] in result['example_ids']
    assert len(result['example_ids']) == 16
    assert result['scientific_run_complete'] is False
    manifest = freeze_execution(config, tmp_path, profile=result['profile_path'], batch_size=16,
                                name='production', review_notes='Reviewed speed, memory and agreement.')
    saved = json.loads(manifest.read_text())
    assert [i for b in saved['batches'] for i in b['example_ids']] == [r['id'] for r in rows]
    assert max(len(b['example_ids']) for b in saved['batches']) == 16
    assert config.read_bytes() == before
    calls = len(deps.batch_calls)
    assert profile_activation(config, tmp_path, dependencies=deps) == result
    assert len(deps.batch_calls) == calls


def test_cli_profiles_then_freezes_and_rejects_unreviewed_or_changed_evidence(tmp_path):
    from pilot_eval.cli import main
    config = prepared(tmp_path)
    args = ['activation-profile', '--config', str(config), '--output-root', str(tmp_path)]
    assert main(args, dependencies=ProfileDependencies()) == 0
    _, rows = load_prepared(config, tmp_path)
    profile = tmp_path / json.loads(config.read_text())['run_path'] / 'profile'
    freeze = ['activation-freeze', '--config', str(config), '--output-root', str(tmp_path),
              '--profile', str(profile), '--batch-size', '8', '--name', 'production',
              '--review-notes', 'Reviewed memory and numerical evidence.']
    assert main(freeze) == 0
    assert main(freeze) == 0
    assert main(freeze[:-1] + ['']) == 1
    changed = list(freeze)
    changed[changed.index('--batch-size') + 1] = '4'
    assert main(changed) == 1
    (profile / 'batch-8/step-8.npz').write_bytes(b'broken')
    assert main(freeze) == 1


def test_profile_preserves_oom_evidence_continues_and_refuses_to_freeze_it(tmp_path):
    from pilot_eval.activation_profile import profile_activation, freeze_execution
    config = prepared(tmp_path)
    deps = ProfileDependencies(oom=8)
    result = profile_activation(config, tmp_path, dependencies=deps)
    assert [r['status'] for r in result['measurements']] == ['passed', 'passed', 'passed', 'oom', 'passed']
    assert deps.closed == 6  # diagnostic plus each candidate, including failed one
    with pytest.raises(ValueError, match='passing'):
        freeze_execution(config, tmp_path, profile=result['profile_path'], batch_size=8,
                         name='bad', review_notes='Reviewed failed candidate.')
    assert not (tmp_path / 'plans/bad/activation.execution.json').exists()


@pytest.mark.parametrize('setting,message', [('drift', 'summary agreement'), ('failure', 'missing hook')])
def test_non_oom_validation_errors_stop_with_evidence_and_no_success_marker(tmp_path, setting, message):
    from pilot_eval.activation_profile import profile_activation
    config = prepared(tmp_path)
    deps = ProfileDependencies(**{setting: 4})
    with pytest.raises(ValueError, match=message):
        profile_activation(config, tmp_path, dependencies=deps)
    profile = tmp_path / json.loads(config.read_text())['run_path'] / 'profile'
    assert not (profile / 'complete.json').exists()
    assert json.loads((profile / 'meta/status.json').read_text())['state'] == 'failed'
    assert (profile / 'logs/errors.jsonl').exists()
    assert not (profile / 'batch-8/complete.json').exists()


class SmallDenominatorEngine(ProfileEngine):
    def measure(self, checkpoint, rows, *, step):
        result = super().measure(checkpoint, rows, step=step)
        for key in ['block_base_norm_sum', 'module_base_norm_sum']:
            result['arrays'][key] *= 1e-8 * (2 if len(rows) == 4 else 1)
        return result


class SmallDenominatorDependencies(ProfileDependencies):
    def activation_engine(self, config, root):
        return SmallDenominatorEngine(self)


def test_profile_compares_derived_ratios_not_just_small_sufficient_sums(tmp_path):
    from pilot_eval.activation_profile import profile_activation
    config = prepared(tmp_path)
    # Small norm sums differ by less than atol; their ratios differ by a factor of two.
    with pytest.raises(ValueError, match='summary agreement'):
        profile_activation(config, tmp_path, dependencies=SmallDenominatorDependencies())
