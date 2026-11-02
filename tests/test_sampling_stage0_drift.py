import json

import numpy as np
import pytest

from pilot_eval import sampling_stage0_drift as drift
from pilot_eval import sampling_stage0_evaluation as evaluation
from pilot_eval.activation_prepare import VIEWS
from pilot_eval.stage0_drift_math import SUM_FIELDS, positions
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

ARMS = ('k0', 'k2', 'k5')
MODULES = ['base_model.model.model.layers.0.mlp.down_proj', 'base_model.model.model.layers.0.self_attn.q_proj']


def adapter(folder, scale):
    from safetensors.numpy import save_file
    folder.mkdir(parents=True)
    tensors = {}
    for m in MODULES:
        tensors[m+'.lora_A.weight'] = np.full((1, 4), 1.0, dtype=np.float32)
        tensors[m+'.lora_B.weight'] = np.full((4, 1), scale, dtype=np.float32)
    save_file(tensors, str(folder/'adapter_model.safetensors'))
    (folder/'adapter_config.json').write_text('{}'); (folder/'complete.json').write_text('{}')


def fixed_rows():
    rows = []
    for corpus, view, n in [('gsm8k', 'question', 150), ('fineweb', 'user', 150)]:
        for i in range(n):
            masks = {v: [False, v == view, v == view] for v in VIEWS}
            rows.append({'id': f'{corpus}:{i}', 'corpus': corpus, 'input_ids': [3, 4, 5], 'attention_mask': [1, 1, 1], 'masks': masks})
    return rows


def setup(tmp_path, monkeypatch, items_hash=None):
    for arm in ARMS: adapter(tmp_path/f'runs/{arm}/checkpoint-64', 0.1)
    adapter(tmp_path/'runs/k0/checkpoint-0', 0.0)
    behaviour = {'protocol': 'sampling-stage0-evaluation-v2', 'parents': {}, 'training_plans': {}, 'cohort_path': 'c', 'cohort_sha256': 'h',
                 'model_snapshot': {}, 'gpu': 'L4', 'weights_dir': 'w', 'model_identity': {'model': 'qwen', 'model_revision': 'r', 'tokenizer_revision': 'r'},
                 'decoding': {}, 'prompt_contract': {}, 'variants': {a: {'adapter_path': f'runs/{a}/checkpoint-64'} for a in ARMS},
                 'adapter_files': {a: {} for a in ARMS},
                 'zero_adapter': {'path': 'runs/k0/checkpoint-0', 'adapter_sha256': file_hash(tmp_path/'runs/k0/checkpoint-0/adapter_model.safetensors')}}
    (tmp_path/'plans').mkdir(); (tmp_path/'plans/behaviour.json').write_text(json.dumps(behaviour))
    rows = fixed_rows(); (tmp_path/'items.json').write_text(json.dumps(rows))
    prepared = {'model': 'qwen', 'model_revision': 'r', 'tokenizer_revision': 'r', 'dtype': 'float32', 'items_path': 'items.json', 'items_sha256': _hash(rows)}
    (tmp_path/'inputs.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(evaluation, 'check_sources', lambda config, root: [])
    return drift.prepare('plans/behaviour.json', 'inputs.json', tmp_path, 'night',
                         items_file_sha256=items_hash or file_hash(tmp_path/'items.json'))


class Engine:
    loaded_adapter_checks = [{'variant': v, 'tensor_count': 392, 'exact_payload_match': True} for v in (*ARMS, 'k-zero')]
    def __init__(self, broken_control=False): self.calls = []; self.broken = broken_control
    def measure(self, rows, variant, *, diagnostic):
        self.calls.append(variant)
        mapping = positions(rows); control = variant in ('untuned', 'k-zero')
        kl = np.full(len(mapping['predictions']), 0.0 if control and not self.broken else 0.01)
        sums = np.zeros((len(rows), len(VIEWS), len(MODULES), 3))
        for e, row in enumerate(rows):
            for v, view in enumerate(VIEWS):
                if any(row['masks'][view]): sums[e, v] = [0.0 if control else 0.5, 2.0, 0.0 if control else 0.3]
        return {'arrays': {'kl': kl, 'write_sums': sums},
                'metadata': dict(example_ids=[r['id'] for r in rows], **mapping, modules=MODULES, variant=variant,
                                 kl_direction='variant||untuned', write_inputs='adapted-module-input', write_fields=SUM_FIELDS),
                'validation': dict(reference_invariant=True, rank1_passed=True, finite=True, mapping_exact=True,
                                   reconstruction_max_fraction=0.1, diagnostic=diagnostic,
                                   repeat_passed=True if diagnostic else None, chunk_passed=True if diagnostic else None,
                                   context_chunk=32, max_logit_contexts=2, workspace_bytes=536870912, max_abs_log_sum=1.0,
                                   chunk_comparison=dict(max_absolute_error=0.0) if diagnostic else None,
                                   kl_roundoff_bound=64*np.finfo(float).eps*2)}
    def check_integrity(self, *, full=False): return {'base_unchanged': True, 'adapters_unchanged': True}
    def close(self): pass


class Dependencies:
    def __init__(self, engine, session='boot-1', driver='580'): self.model = engine; self.session = session; self.driver = driver
    def runtime(self): return {'execution_kind': 'real-cuda', 'device': 'NVIDIA L4', 'driver': self.driver, 'session_id': self.session, 'git_commit': 'c'}
    def engine(self, config, root): return self.model


def test_full_collection_matches_stage0_layout_and_verifies(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    config = json.loads(path.read_text())
    assert config['controls'] == ['untuned', 'k-zero'] and config['variant_order'] == [*ARMS, 'k-zero']
    assert len(config['units']) == 2 + 38*3 and config['batch_size'] == 1 and config['context_chunk'] == 32
    result = drift.run(path, tmp_path, dependencies=Dependencies(Engine()))
    assert result['controls'] == {'untuned': {'documents': 4, 'exact_zero': True}, 'k-zero': {'documents': 4, 'exact_zero': True}}
    assert {c['variant'] for c in result['cells']} == set(ARMS) and result['comparison_provenance']['runtime']['driver'] == '580'
    assert drift.verify(path, tmp_path)['cells'] == result['cells']
    docs = tmp_path/'runs/stage0/night/fixed-context/results/per-document.jsonl'
    docs.write_text(docs.read_text().replace('0.01', '0.02', 1))
    with pytest.raises(ValueError, match='changed|disagree'):
        drift.verify(path, tmp_path)


def test_only_the_original_stage0_payload_is_accepted(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='original Stage 0'):
        setup(tmp_path, monkeypatch, items_hash='0'*64)


def test_zero_control_with_nonzero_kl_stops_the_run(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='control'):
        drift.run(path, tmp_path, dependencies=Dependencies(Engine(broken_control=True)))


def test_resume_after_restart_measures_only_unsealed_units(tmp_path, monkeypatch):
    path = setup(tmp_path, monkeypatch)
    assert drift.run(path, tmp_path, dependencies=Dependencies(Engine()), max_new_units=10)['state'] == 'incomplete'
    with pytest.raises(ValueError, match='runtime'):
        drift.run(path, tmp_path, dependencies=Dependencies(Engine(), driver='570'))
    resumed = Engine()
    drift.run(path, tmp_path, dependencies=Dependencies(resumed, session='boot-2'))
    assert len(resumed.calls) == 2 + 38*3 - 10
