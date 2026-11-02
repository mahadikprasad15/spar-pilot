import json

import pytest

from pilot_eval import sampling_stage0_agreement as agreement
from pilot_eval import sampling_stage0_evaluation as evaluation

ARMS = ('k0', 'k2', 'k5')


def cohort():
    return [{'id': f'q{i}', 'prompt': f'p{i}', 'gold': '#### 1'} for i in range(150)]


class Tokenizer:
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False): return [5, 6]


class Engine:
    loaded_adapter_checks = [{'variant': a, 'tensor_count': 392, 'exact_payload_match': True} for a in ARMS]
    tokenizer = Tokenizer()
    def __init__(self, diverge=(), error=None, crash_after=None):
        self.inputs = {}; self.calls = []; self.diverge = set(diverge); self.error = error or {}
        self.crash_after = crash_after
    def prepare_inputs(self, rows):
        for r in rows: self.inputs[r['id']] = [7]
        return [{'id': r['id'], 'input_ids': [7], 'attention_mask': [1]} for r in rows]
    def generate(self, rows, variant, seed):
        self.calls.append(('generate', variant, len(rows)))
        if self.crash_after is not None and len(self.calls) > self.crash_after: raise RuntimeError('pod stopped')
        batched = len(rows) > 1
        return [{'text': '#### 1', 'token_ids': [9 if batched and (variant, r['id']) in self.diverge else 1],
                 'generated_token_ids': [9 if batched and (variant, r['id']) in self.diverge else 1, 2],
                 'stop_reason': 'eos'} for r in rows]
    def compare(self, rows, variant, token_rows):
        self.calls.append(('compare', variant, len(rows)))
        error = self.error.get(variant, 1e-5)
        return {'max_absolute_error': error, 'examples': [{'id': r['id'], 'errors': [error], 'positions': [0]} for r in rows]}
    def divergence(self, rows, variant, index, reference, responses):
        return {'near_tie': False, 'first_divergence': 0}
    def check_integrity(self, *, full=False): return {'base_unchanged': True, 'adapters_unchanged': True}
    def close(self): pass


class Dependencies:
    def __init__(self, engine, session='boot-1', driver='580'):
        self.model = engine; self.session = session; self.driver = driver
    def runtime(self):
        return {'execution_kind': 'real-cuda', 'device': 'NVIDIA L4', 'driver': self.driver,
                'session_id': self.session, 'git_commit': 'c'}
    def engine(self, config, root): return self.model


def plan(tmp_path, monkeypatch):
    rows = cohort()
    source = {'parents': {}, 'cohort_path': 'items.json', 'cohort_sha256': 'h', 'weights_dir': 'w', 'model_snapshot': {},
              'model_identity': {'base_sha256': 'b'}, 'prompt_contract': {}, 'decoding': {'eos_token_id': 2, 'pad_token_id': 0, 'max_new_tokens': 10},
              'variants': {a: {'adapter_path': a} for a in ARMS}, 'variant_order': list(ARMS),
              'adapter_files': {a: {} for a in ARMS}, 'training_plans': {}, 'zero_adapter': {}}
    monkeypatch.setattr(evaluation, 'sources', lambda manifest, root: (source, rows))
    monkeypatch.setattr(evaluation, 'check_sources', lambda config, root: rows)
    return agreement.prepare('manifest.json', tmp_path, 'night')


def test_all_arms_pass_and_qualify_batching(tmp_path, monkeypatch):
    path = plan(tmp_path, monkeypatch)
    engine = Engine()
    result = agreement.run(path, tmp_path, dependencies=Dependencies(engine))
    assert result['batch_sizes'] == {'k0': 16, 'k2': 16, 'k5': 16}
    assert sum(c == ('generate', a, 1) for c in engine.calls for a in ARMS) == 48 # fresh singleton references
    assert {c for c in engine.calls if c[0] == 'compare'} == {('compare', a, s) for a in ARMS for s in (16, 6)}
    assert agreement.verify(path, tmp_path)['batch_sizes'] == result['batch_sizes']
    saved = tmp_path/'runs/stage0/night/batch-agreement/results/agreement.json'
    saved.write_text(saved.read_text().replace('"k2": 16', '"k2": 1'))
    with pytest.raises(ValueError, match='changed'):
        agreement.verify(path, tmp_path)


def test_greedy_divergence_without_near_tie_makes_only_that_arm_singleton(tmp_path, monkeypatch):
    path = plan(tmp_path, monkeypatch)
    result = agreement.run(path, tmp_path, dependencies=Dependencies(Engine(diverge={('k2', 'q3')})))
    assert result['batch_sizes'] == {'k0': 16, 'k2': 1, 'k5': 16}
    assert result['arms']['k2']['greedy_passed'] is False and result['arms']['k2']['exact_answers'] == 20


def test_logit_error_above_unchanged_gate_makes_arm_singleton(tmp_path, monkeypatch):
    path = plan(tmp_path, monkeypatch)
    result = agreement.run(path, tmp_path, dependencies=Dependencies(Engine(error={'k5': 2e-3})))
    assert result['batch_sizes'] == {'k0': 16, 'k2': 16, 'k5': 1} and result['batch_gate'] == 1e-3


def test_resume_after_pod_restart_reuses_sealed_units_but_not_other_hardware(tmp_path, monkeypatch):
    path = plan(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match='pod stopped'):
        agreement.run(path, tmp_path, dependencies=Dependencies(Engine(crash_after=10)))
    with pytest.raises(ValueError, match='runtime'):
        agreement.run(path, tmp_path, dependencies=Dependencies(Engine(), driver='570'))
    resumed = Engine()
    agreement.run(path, tmp_path, dependencies=Dependencies(resumed, session='boot-2'))
    assert sum(c[0] == 'generate' and c[2] == 1 for c in resumed.calls) == 38 # ten references were sealed


def test_behaviour_plan_takes_batch_sizes_from_agreement(tmp_path, monkeypatch):
    path = plan(tmp_path, monkeypatch)
    agreement.run(path, tmp_path, dependencies=Dependencies(Engine(diverge={('k0', 'q0')})))
    behaviour = evaluation.prepare('manifest.json', tmp_path, 'night', agreement_plan=path)
    config = json.loads(behaviour.read_text())
    assert config['batch_sizes'] == {'k0': 1, 'k2': 16, 'k5': 16} and config['gpu'] == 'L4'
    assert config['batch_agreement']['environment']['driver'] == '580'
    assert 'session_id' not in config['batch_agreement']['environment']
