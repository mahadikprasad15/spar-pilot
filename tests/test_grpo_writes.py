"""GRPO fixed-input preparation through the approved public workflow seam."""
import json
import pytest
from pilot_eval.cli import main
from pilot_eval.activation_prepare import load_prepared
from test_grpo_training import frozen, train, Dependencies, Engine
from pilot_eval.activation_prepare import PROJECTIONS


class MatchedEngine(Engine):
    def check_integrity(self):
        return dict(base_before="b" * 64, base_after="b" * 64, base_unchanged=True)

    def save_checkpoint(self, directory, step):
        super().save_checkpoint(directory, step)
        (directory / "adapter_config.json").write_text(json.dumps(dict(
            r=1, lora_alpha=1, lora_dropout=0, target_modules=PROJECTIONS)))


class MatchedDependencies(Dependencies):
    def load_training(self, *args):
        return MatchedEngine(self)


def source(root):
    assert train(root, frozen(root), MatchedDependencies()) == 0
    return root / "plans/arm/grpo.training.json"


def test_grpo_preparation_reuses_frozen_sequences_without_tokenizing(tmp_path):
    training = source(tmp_path)
    train = json.loads(training.read_text())
    frozen = json.loads((tmp_path / train['frozen_path']).read_text())
    plan = json.loads((tmp_path / frozen['source_path']).read_text())
    inputs = tmp_path / plan['sources']['measurement']['path']
    # This source is an existing Pilot 3 execution, not an SFT training config.
    args = ['grpo-writes-prepare', '--config', str(training), '--measurement-config', str(inputs),
            '--name', 'grpo-writes', '--output-root', str(tmp_path)]
    assert main(args) == 0
    path = tmp_path / 'plans/grpo-writes/activation.prepared.json'
    config, rows = load_prepared(path, tmp_path)
    old, original = load_prepared(inputs, tmp_path)
    assert rows == original
    assert config['items_sha256'] == old['items_sha256']
    assert config['source_contract']['kind'] == 'grpo'
    assert config['source_contract']['training_path'] == str(training.relative_to(tmp_path))
    assert config['source_evidence']['checkpoints']['64']['path'].startswith(train['run_path'])
    before = path.read_bytes()
    assert main(args) == 0
    assert path.read_bytes() == before


def test_random_control_matches_every_module_norm_without_changing_source_or_rng(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file, load_file
    from pilot_eval.grpo_random_control import construct_control
    source_dir = tmp_path / 'trained'
    source_dir.mkdir()
    factors = {}
    for layer in range(28):
        for projection in PROJECTIONS:
            stem = f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            factors[stem+'.lora_A.weight'] = np.array([[3.,4.]],dtype=np.float32)
            factors[stem+'.lora_B.weight'] = np.array([[0.],[2.],[-1.]],dtype=np.float32)
    factors['base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight'] *= 0
    save_file(factors,str(source_dir/'adapter_model.safetensors'))
    (source_dir/'adapter_config.json').write_text(json.dumps(dict(r=1,lora_alpha=1,lora_dropout=0,
                                                              target_modules=PROJECTIONS)))
    before={p:p.read_bytes() for p in source_dir.iterdir()}
    np.random.seed(19)
    expected=np.random.random(3)
    np.random.seed(19)
    result=construct_control(source_dir,tmp_path/'control')
    assert np.array_equal(np.random.random(3),expected)
    saved=load_file(str(tmp_path/'control/adapter_model.safetensors'))
    assert len(result['modules'])==196
    for key,A in factors.items():
        if '.lora_A.' not in key:continue
        B=factors[key.replace('.lora_A.','.lora_B.')]
        a,b=saved[key],saved[key.replace('.lora_A.','.lora_B.')]
        assert np.linalg.norm(a)==pytest.approx(1.,rel=1e-6)
        # Independent literal: ||[3,4]|| * ||[0,2,-1]|| = sqrt(125).
        target=0. if not np.any(B) else np.sqrt(125.)
        assert np.linalg.norm(b@a)==pytest.approx(target,rel=1e-6)
    construct_control(source_dir,tmp_path/'other')
    other=load_file(str(tmp_path/'other/adapter_model.safetensors'))
    assert all(np.array_equal(saved[k],other[k]) for k in saved)
    assert all(p.read_bytes()==value for p,value in before.items())
