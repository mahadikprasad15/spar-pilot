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
