"""Calibration is separate from validation and cannot authorize corrupt measurements."""
import json
import numpy as np
import pytest
from test_activation_validation import prepared
from test_activation_profile import ProfileDependencies, ProfileEngine


def test_calibration_metric_uses_vector_norm_and_measured_resolution():
    from pilot_eval.activation_calibration import comparison_metrics, fit_rule, check_agreement
    from pilot_eval.activation_profile import ARRAYS
    counts = np.ones((2, 3, 28))
    modules = np.ones((2, 3, 28, 7))
    a = {key: (modules.copy() if key.startswith('module_') else counts.copy()) for key in ARRAYS}
    a['block_base_sum'] = np.zeros((2, 3, 28, 2)); a['block_base_sum'][..., 0] = 1
    a['block_delta_sum'] = a['block_base_sum'].copy()
    b = {key: value.copy() for key, value in a.items()}
    b['block_delta_sum'][..., 1] = 1e-6
    metrics = comparison_metrics({8:a}, {8:b})
    assert metrics['errors']['block_delta_vector'] == pytest.approx(1e-6)
    rule = fit_rule([metrics], margin=3)
    assert check_agreement({8:a}, {8:b}, rule)['passed']
    wrong = {key: value.copy() for key, value in a.items()}
    wrong['block_delta_sum'] *= -1
    assert not check_agreement({8:a}, {8:wrong}, rule)['passed']
    wrong['block_delta_sum'] *= -2
    wrong['block_delta_norm_sum'] *= 2
    assert not check_agreement({8:a}, {8:wrong}, rule)['passed']
    wrong['block_count'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='count'):
        comparison_metrics({8:a}, {8:wrong})


class CalibrationEngine(ProfileEngine):
    def capture_reference(self, rows, *, padded_width=None):
        if padded_width:
            self.deps.padded_calls.append(padded_width)
        return super().capture_reference(rows)

class CalibrationDeps(ProfileDependencies):
    def __init__(self):
        super().__init__()
        self.padded_calls = []
    def activation_engine(self, config, root):
        return CalibrationEngine(self)


def test_calibration_flow_freezes_disjoint_cohorts_and_requires_independent_validation(tmp_path):
    from pilot_eval.activation_calibration import prepare_calibration, collect_calibration, freeze_rule, validate_rule, load_validated_rule
    config = prepared(tmp_path)
    before = config.read_bytes()
    deps = CalibrationDeps()
    plan = prepare_calibration(config, tmp_path, name='noise-v2', dependencies=deps)
    frozen = json.loads(plan.read_text())
    assert len(frozen['cohorts']['calibration']) == len(frozen['cohorts']['validation']) == 16
    assert not set(frozen['cohorts']['calibration']) & set(frozen['cohorts']['validation'])
    assert not set(frozen['excluded_profile_ids']) & set(sum(frozen['cohorts'].values(), []))
    with pytest.raises(ValueError, match='freeze'):
        validate_rule(plan, tmp_path, dependencies=deps)
    collect_calibration(plan, tmp_path, dependencies=deps)
    calls = len(deps.batch_calls)
    collect_calibration(plan, tmp_path, dependencies=deps)
    assert len(deps.batch_calls) == calls
    assert deps.padded_calls
    with pytest.raises(ValueError, match='review'):
        freeze_rule(plan, tmp_path, review_notes='')
    freeze_rule(plan, tmp_path, review_notes='Reviewed calibration evidence and the 3x engineering margin.')
    validated = validate_rule(plan, tmp_path, dependencies=deps)
    assert validated['validated_batches'] == [1, 2, 4, 8, 16]
    rule = load_validated_rule(plan, tmp_path, prepared=json.loads(config.read_text()), runtime=deps.runtime())
    assert rule['protocol'] == 'pilot3-agreement-v2'
    assert config.read_bytes() == before
    target = tmp_path / frozen['run_path'] / 'calibration/batch-2/step-8.npz'
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        load_validated_rule(plan, tmp_path, prepared=json.loads(config.read_text()), runtime=deps.runtime())
