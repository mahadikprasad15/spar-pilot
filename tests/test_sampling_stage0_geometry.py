import json

import numpy as np
import pytest

from pilot_eval import sampling_stage0_evaluation as evaluation
from pilot_eval import sampling_stage0_geometry as geometry
from pilot_eval.training import file_hash

ARMS = ('k0', 'k2', 'k5')
MODULES = ['base_model.model.model.layers.0.mlp.down_proj', 'base_model.model.model.layers.3.self_attn.q_proj']


class Weights:
    def __init__(self, root): self.root = root; self.reads = 0
    def inventory(self, directory, audit):
        assert audit['identity']['base_sha256'] == 'base'
        return {'execution_kind': 'cpu-local-pinned-weights', 'base_sha256': 'base',
                'modules': {m: {'path': str(self.root/'w/model.safetensors'), 'key': m} for m in MODULES}}
    def matrix(self, entry):
        self.reads += 1
        return np.random.default_rng(MODULES.index(entry['key'])).normal(size=(48, 44))


def factors(directory):
    step = int(directory.name.split('-')[1]); arm = directory.parent.parent.name
    scale = 0.0 if step == 0 else step/64*(1+ARMS.index(arm))
    rng = np.random.default_rng(7)
    return {m: (rng.normal(size=(1, 44)).astype(np.float32), (scale*np.ones((48, 1))).astype(np.float32)) for m in MODULES}


def setup(tmp_path, monkeypatch):
    (tmp_path/'w').mkdir(); (tmp_path/'w/model.safetensors').write_bytes(b'x')
    zero = None
    for arm in ARMS:
        for step in geometry.STEPS:
            cp = tmp_path/f'runs/{arm}/checkpoints/checkpoint-{step}'; cp.mkdir(parents=True)
            (cp/'adapter_config.json').write_text('{}')
            (cp/'adapter_model.safetensors').write_bytes(b'zero' if step == 0 else f'{arm}{step}'.encode())
            (cp/'complete.json').write_text(json.dumps({'step': step, 'files': {n: file_hash(cp/n) for n in ['adapter_config.json', 'adapter_model.safetensors']}}))
            if step == 0: zero = file_hash(cp/'adapter_model.safetensors')
    behaviour = {'protocol': 'sampling-stage0-evaluation-v2', 'parents': {}, 'training_plans': {}, 'cohort_path': 'c', 'cohort_sha256': 'h',
                 'weights_dir': 'w', 'model_snapshot': {}, 'adapter_files': {}, 'model_identity': {'base_sha256': 'base'},
                 'variants': {a: {'adapter_path': f'runs/{a}/checkpoints/checkpoint-64'} for a in ARMS},
                 'zero_adapter': {'path': 'runs/k0/checkpoints/checkpoint-0', 'adapter_sha256': zero}}
    (tmp_path/'plans').mkdir(); (tmp_path/'plans/behaviour.json').write_text(json.dumps(behaviour))
    monkeypatch.setattr(evaluation, 'check_sources', lambda config, root: [])
    return geometry.prepare('plans/behaviour.json', tmp_path, 'night', weights=Weights(tmp_path))


def test_projections_trajectory_and_cross_arm_for_all_steps(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    config = json.loads(path.read_text())
    assert len(config['checkpoints']) == 15 and config['same_base_as_stage0'] is False
    summary = geometry.run(path, tmp_path, weights=Weights(tmp_path), load_factors=factors)
    assert summary['rows'] == len(MODULES)*3*15 # modules x energy levels x (arm, step)
    rows = [json.loads(l) for l in (tmp_path/'runs/stage0/night/geometry/results/projections.jsonl').read_text().splitlines()]
    zero = [r for r in rows if r['step'] == 0]
    assert zero and all(r['pi_out'] is None and r['undefined_out_reason'] == 'zero factor' for r in zero)
    assert all(0 <= r['pi_in'] <= 1 for r in rows) and all(r['chance_out'] == r['k']/48 for r in rows)
    trajectory = {(t['arm'], t['step']): t['total_delta_frobenius'] for t in summary['trajectory']}
    assert trajectory[('k0', 0)] == 0 and trajectory[('k5', 64)] > trajectory[('k0', 64)] > trajectory[('k0', 8)]
    assert all(abs(p['mean_delta_cosine']-1) < 1e-9 for p in summary['cross_arm']) # same directions, different sizes
    assert geometry.verify(path, tmp_path) == summary


def test_resume_skips_sealed_modules_and_tampering_is_detected(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    first = Weights(tmp_path)
    assert geometry.run(path, tmp_path, weights=first, load_factors=factors, max_new_units=1)['state'] == 'incomplete'
    second = Weights(tmp_path)
    geometry.run(path, tmp_path, weights=second, load_factors=factors)
    assert first.reads == 1 and second.reads == 1
    unit = tmp_path/'runs/stage0/night/geometry/units/L00.down_proj/result.json'
    unit.write_text(unit.read_text().replace('"k0"', '"k9"', 1))
    with pytest.raises(ValueError, match='changed'):
        geometry.verify(path, tmp_path)


def test_changed_checkpoint_payload_blocks_the_run(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    (tmp_path/'runs/k2/checkpoints/checkpoint-32/adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        geometry.run(path, tmp_path, weights=Weights(tmp_path), load_factors=factors)
