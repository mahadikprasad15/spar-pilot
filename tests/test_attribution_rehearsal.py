"""Ticket 06a driver on CPU: the Colab covering set with a tiny backend and a mocked Hub (nothing run on a GPU)."""
import json

import numpy as np

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
    backend = TinyBackend(tmp_path, strict_rehearsal=True)  # any discovery/held-out/cohort access raises
    summary = rh.Rehearsal(record(tmp_path), tmp_path / 'local', backend, client=hub,
                           clean_root=tmp_path / 'clean').run()
    rehearsal_prompts = [r['prompt_ids'] for r in backend.rehearsal_items()]
    assert backend.probes and all(p in rehearsal_prompts for p in backend.probes)  # review r4 (11)
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


def test_06a_steps_that_need_calibration_rows_are_skipped_with_a_reason_when_traces_fail(tmp_path):
    backend = TinyBackend(tmp_path, strict_rehearsal=True)
    backend.sources = lambda: (_ for _ in ()).throw(ValueError('inventory-v2 failed verification'))
    summary = rh.Rehearsal(record(tmp_path), tmp_path / 'local', backend).run()
    steps = summary['steps']
    assert steps['calibration_traces']['status'] == 'failed'
    for name in ['attribution:gold-sft', 'finite_differences:gold-sft', 'conditions:gold-sft', 'generation:gold-sft',
                 'token_budget_probe']:
        assert steps[name]['status'] == 'skipped' and steps[name]['reason'] == 'needs calibration_traces'
    assert steps['corrupt_and_interrupted_units']['status'] == 'ok'  # independent steps still run
    assert summary['failed'] == ['calibration_traces'] and 'attribution:gold-sft' in summary['skipped']


def test_06a_attribution_with_no_rows_fails_instead_of_passing(tmp_path):
    reh = rh.Rehearsal(record(tmp_path), tmp_path / 'local', TinyBackend(tmp_path, strict_rehearsal=True))
    with pytest.raises(ValueError, match='no calibration rows'):
        reh._attribution('gold-sft', [])


def test_06a_r2_selection_reads_the_sealed_attribution_units_and_never_recomputes_on_the_gpu(tmp_path, monkeypatch):
    """06a r2: recomputing attribution over all calibration rows in one batch ran out of GPU memory; the
    selection must come from the units the attribution step already sealed."""
    from pilot_eval import attribution_instrument as ai
    reh = rh.Rehearsal(record(tmp_path), tmp_path / 'local', TinyBackend(tmp_path, strict_rehearsal=True))
    reh.step('calibration_traces', lambda: reh.stages._traces('calibration', 1, minimum=1)['counts'])
    rows = rh.at.load_rows(reh.root, 'calibration')
    reh._attribution('gold-sft', rows)
    expected = None
    original = ai.Instrument.attribute
    expected = original(reh.stages._instrument(reh.backend.load('gold-sft')), rows)  # one batch, for reference

    def forbidden(self, rows_, **kw):
        raise AssertionError('selection must not run attribution on the GPU')

    monkeypatch.setattr(ai.Instrument, 'attribute', forbidden)
    sel = reh._calibration_selection('gold-sft', rows)
    mean = np.nanmean(expected['attr']['trace'], 0)
    assert sel['top'] == rh.ab.top_k(mean, reh.record['selection']['top_group'])
    position = sorted(reh.stages._instrument(reh.backend.load('gold-sft')).down_index).index(sel['layer'])
    assert sel['cbar'] == pytest.approx(expected['signed_sum_c'][:, position].sum() / expected['token_count'].sum(),
                                        rel=1e-5)


def test_06a_r2_interventions_are_evaluated_in_bounded_batches(tmp_path, monkeypatch):
    reh = rh.Rehearsal(record(tmp_path), tmp_path / 'local', TinyBackend(tmp_path, strict_rehearsal=True))
    reh.step('calibration_traces', lambda: reh.stages._traces('calibration', 1, minimum=1)['counts'])
    rows = rh.at.load_rows(reh.root, 'calibration')
    reh._attribution('gold-sft', rows)
    budget = min(reh.record['token_budget_sweep'])
    seen, original = [], rh.ab.StageC.evaluate

    def evaluate(self, batch, condition):
        seen.append(len(batch) * max(len(r['input_ids']) for r in batch))
        return original(self, batch, condition)

    monkeypatch.setattr(rh.ab.StageC, 'evaluate', evaluate)
    out = reh._conditions('gold-sft', rows)
    assert len(seen) > 6 and max(seen) <= budget and set(out['conditions']) == {
        'noop', 'single', 'group', 'down_full', 'read64', 'constant'}


def test_fd_step_uses_the_fp64_reference_and_keeps_every_comparison(tmp_path):
    reh = rh.Rehearsal(record(tmp_path), tmp_path / 'local', TinyBackend(tmp_path, strict_rehearsal=True))
    reh.step('calibration_traces', lambda: reh.stages._traces('calibration', 1, minimum=1)['counts'])
    fd = reh._fd('gold-sft', rh.at.load_rows(reh.root, 'calibration'), reh.record['tolerances'])
    assert fd['precision'] == 'fp64' and fd['items'] and len(fd['worst']) <= 10 and 'peak_bytes' in fd
