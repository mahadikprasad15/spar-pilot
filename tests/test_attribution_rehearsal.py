"""Ticket 06a driver on CPU: the Colab covering set with a tiny backend and a mocked Hub (nothing run on a GPU)."""
import json

import pytest

pytest.importorskip('torch', exc_type=ImportError)
pytest.importorskip('peft', exc_type=ImportError)

from pilot_eval import attribution_rehearsal as rh  # noqa: E402
from pilot_eval import attribution_units as au  # noqa: E402
from tests.test_attribution_stages import TinyBackend, launch  # noqa: E402
from tests.test_neuron_audit_export import FakeHub  # noqa: E402


def record(tmp_path):
    rec = dict(launch(tmp_path), rehearsal_label='rehearsal-colab-cpu')
    return rec


def test_covering_set_runs_and_exports_a_restorable_partial_output(tmp_path):
    hub = FakeHub()
    summary = rh.Rehearsal(record(tmp_path), tmp_path / 'local', TinyBackend(tmp_path), client=hub,
                           clean_root=tmp_path / 'clean').run()
    assert summary['failed'] == [], json.dumps(summary, default=str)[:4000]
    steps = summary['steps']
    for name in ['calibration_traces', 'attribution:gold-sft', 'finite_differences:gold-sft', 'conditions:gold-sft',
                 'generation:gold-sft', 'corrupt_and_interrupted_units', 'token_budget_probe']:
        assert steps[name]['status'] == 'ok'
    conditions = steps['conditions:online-grpo']['result']['conditions']
    assert set(conditions) == {'noop', 'single', 'group', 'down_full', 'read64', 'constant'}
    assert conditions['noop']['mean_kl'] == 0.0 and conditions['group']['mean_kl'] > 0
    assert steps['corrupt_and_interrupted_units']['result'] == dict(corrupted='corrupt', interrupted='interrupted')
    assert {p['probe'] for p in steps['token_budget_probe']['result']} == {'singleton-longest', 'packed'}
    assert summary['export']['status'] == 'ok' and summary['export']['restored_summary_label'] == 'rehearsal-colab-cpu'
    restored = au.latest_summary(tmp_path / 'clean')
    assert restored['summary']['label'] == 'rehearsal-colab-cpu' and summary['export']['completeness'] == 'complete'


def test_a_failing_step_is_recorded_and_later_steps_still_run(tmp_path):
    backend = TinyBackend(tmp_path)
    backend.rehearsal_items = lambda: (_ for _ in ()).throw(RuntimeError('tokenizer exploded'))
    summary = rh.Rehearsal(record(tmp_path), tmp_path / 'local', backend).run()
    assert 'generation:gold-sft' in summary['failed'] and 'tokenizer exploded' in \
        summary['steps']['generation:gold-sft']['error']
    assert summary['steps']['token_budget_probe']['status'] == 'ok'


def test_only_labelled_colab_rehearsal_records_are_accepted(tmp_path):
    with pytest.raises(ValueError, match='rehearsal-colab'):
        rh.Rehearsal(launch(tmp_path), tmp_path / 'local', TinyBackend(tmp_path))
