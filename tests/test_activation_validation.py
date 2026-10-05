import json
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from pilot_eval.activation_prepare import prepare_activation
from test_activation_prepare import ActivationData, completed_source


class DiagnosticEngine:
    closed = False
    calls = []
    failure = None

    def capture_reference(self, rows):
        return rows

    def measure(self, checkpoint, reference, *, step):
        from pilot_eval.activation_engine import InstrumentFailure
        self.calls.append(step)
        if self.failure == step:
            raise InstrumentFailure('planted rank-1 failure', {'layer': 5, 'projection': 'v_proj'})
        arrays = {'block_delta_sum': np.full((len(reference), 3, 28, 4), step, dtype=np.float64),
                  'module_count': np.ones((len(reference), 3, 28, 7), dtype=np.float64)}
        return {'arrays': arrays, 'validation': {'exact_zero': step == 0, 'rank1_passed': True,
                'reference_invariant': True, 'module_count': 196, 'base_sha256': 'b' * 64}}

    def base_hash(self):
        return 'b' * 64

    def close(self):
        self.closed = True


class DiagnosticDependencies:
    def __init__(self, failure=None):
        self.engine = DiagnosticEngine()
        self.engine.calls = []
        self.engine.failure = failure

    def runtime(self):
        return {'device': 'controlled-CPU-fixture', 'precision': 'float32', 'tf32': False}

    def activation_engine(self, config, root):
        return self.engine


def prepared(root):
    return prepare_activation(completed_source(root), root, 'instrument', dependencies=ActivationData())


def test_public_diagnostic_saves_five_verified_summaries_and_reuses_completion(tmp_path):
    config = prepared(tmp_path)
    args = ['activation-validate', '--config', str(config), '--output-root', str(tmp_path), '--batch-size', '2']
    dependencies = DiagnosticDependencies()
    assert main(args, dependencies=dependencies) == 0
    assert dependencies.engine.calls == [0, 8, 16, 32, 64]
    assert dependencies.engine.closed
    run = tmp_path / json.loads(config.read_text())['run_path']
    diagnostic = next((run / 'validation').glob('diagnostic-*'))
    result = json.loads((diagnostic / 'results/results.json').read_text())
    assert result['mode'] == 'diagnostic-only'
    assert result['scientific_run_complete'] is False
    assert result['checkpoint_steps'] == [0, 8, 16, 32, 64]
    assert len(result['example_ids']) == 2
    for step in result['checkpoint_steps']:
        with np.load(diagnostic / f'checkpoints/step-{step}/summaries.npz', allow_pickle=False) as saved:
            assert saved['block_delta_sum'][0, 0, 0, 0] == step
    assert main(args, dependencies=dependencies) == 0
    assert dependencies.engine.calls == [0, 8, 16, 32, 64]
    payload = diagnostic / 'checkpoints/step-8/summaries.npz'
    payload.write_bytes(b'corrupt')
    assert main(args, dependencies=dependencies) == 1


def test_failed_instrument_persists_diagnostics_and_does_not_publish_success(tmp_path):
    config = prepared(tmp_path)
    dependencies = DiagnosticDependencies(failure=16)
    assert main(['activation-validate', '--config', str(config), '--output-root', str(tmp_path)],
                dependencies=dependencies) == 1
    run = tmp_path / json.loads(config.read_text())['run_path']
    diagnostic = next((run / 'validation').glob('diagnostic-*'))
    assert not (diagnostic / 'complete.json').exists()
    assert json.loads((diagnostic / 'meta/status.json').read_text())['state'] == 'failed'
    errors = [json.loads(line) for line in (diagnostic / 'logs/errors.jsonl').read_text().splitlines()]
    assert errors[-1]['diagnostics'] == {'layer': 5, 'projection': 'v_proj'}
    assert dependencies.engine.closed
