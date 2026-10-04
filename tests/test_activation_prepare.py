"""Public Pilot 3 preparation tests: real files, controlled offline source access."""

import json
import pytest

from pilot_eval.cli import main
from pilot_eval.sft import prepare_sft
from pilot_eval.training import run_sft, training_directory, seal_checkpoint
from test_sft_prepare import TrainingData, ChatTokenizer, source_plan
from test_sft_training import TrainingBoundary, Engine


class OffsetTokenizer(ChatTokenizer):
    all_special_ids = [0, 3]
    def __call__(self, text, **kwargs):
        # Independent fixture: one character per token except the end-turn marker.
        ids, offsets = [], []
        position = 0
        while position < len(text):
            width = 5 if text.startswith('<end>', position) else 1
            ids.append(3 if width == 5 else ord(text[position]) + 10)
            offsets.append((position, position + width))
            position += width
        return {'input_ids': ids, 'offset_mapping': offsets}


class ActivationData(TrainingData):
    tokenizer = OffsetTokenizer()

    def stream_fineweb(self, config, revision):
        assert config == 'sample-10BT'
        assert revision == 'fineweb-revision'
        for index in range(2100):
            yield {'id': f'doc-{index}', 'text': f'{index:04d} ' + 'x' * 160}

    def resolve(self, repo, kind):
        return 'fineweb-revision' if repo == 'HuggingFaceFW/fineweb' else super().resolve(repo, kind)


def completed_source(root):
    deps = ActivationData()
    source = source_plan(root, deps)
    path = prepare_sft(source, root, 'sft', dependencies=deps)
    training = TrainingBoundary()
    training.engine = Engine()
    run_sft(path, root, dependencies=training)
    directory = training_directory(root, json.loads(path.read_text()))
    for step in [0, 8, 16, 32, 64]:
        cp = directory / 'checkpoints' / f'checkpoint-{step}'
        (cp / 'adapter_config.json').write_text(json.dumps({
            'r': 1, 'lora_alpha': 1, 'lora_dropout': 0,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        }))
        seal_checkpoint(cp, step)
    return path


def test_prepare_command_freezes_auditable_complete_inputs_and_reuses_them(tmp_path, capsys):
    source = completed_source(tmp_path)
    args = ['activation-prepare', '--source-config', str(source), '--name', 'write-v1',
            '--output-root', str(tmp_path)]
    assert main(args, dependencies=ActivationData()) == 0
    manifest_path = tmp_path / 'plans/write-v1/activation.prepared.json'
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads((tmp_path / manifest['items_path']).read_text())
    gsm = [row for row in rows if row['corpus'] == 'gsm8k']
    web = [row for row in rows if row['corpus'] == 'fineweb']
    assert len(gsm) == len(web) == 150
    assert [row['id'] for row in gsm] == [item['id'] for item in json.loads(
        (tmp_path / json.loads(source.read_text())['evaluation_items_path']).read_text())]
    tokenizer = OffsetTokenizer()
    for row in gsm:
        counted = ''.join(tokenizer.decode([token]) if hasattr(tokenizer, 'decode') else
                          chr(token - 10) for token, chosen in zip(row['input_ids'], row['masks']['question']) if chosen)
        assert counted == row['question']
        solution = ''.join(chr(token - 10) for token, chosen in
                           zip(row['input_ids'], row['masks']['solution']) if chosen)
        assert solution == row['gold']
        assert not any(a and b for a, b in zip(row['masks']['question'], row['masks']['solution']))
    assert all(sum(row['masks']['user']) == 128 for row in web)
    assert all(row['source_index'] < 2000 for row in web)
    assert manifest['checkpoint_steps'] == [0, 8, 16, 32, 64]
    assert manifest['source_evidence']['base_sha256'] == 'b' * 64
    assert manifest['fineweb']['config'] == 'sample-10BT'
    assert manifest['fineweb']['revision'] == 'fineweb-revision'
    audit = tmp_path / manifest['audit_path']
    assert 'Counted question' in audit.read_text()
    before = {p: p.read_bytes() for p in manifest_path.parent.rglob('*') if p.is_file()}

    class NoNetwork(ActivationData):
        def load_tokenizer(self, *args):
            raise AssertionError('completed preparation must not tokenize again')
        def stream_fineweb(self, *args):
            raise AssertionError('completed preparation must not stream again')
        def load_dataset(self, *args):
            raise AssertionError('completed preparation must not reload the dataset')

    assert main(args, dependencies=NoNetwork()) == 0
    assert all(p.read_bytes() == content for p, content in before.items())
    assert main(['activation-audit', '--config', str(manifest_path),
                 '--output-root', str(tmp_path)]) == 0
    assert 'prepared' in capsys.readouterr().out


def test_preparation_rejects_completion_marker_that_omits_required_audit(tmp_path):
    source = completed_source(tmp_path)
    args = ['activation-prepare', '--source-config', str(source), '--name', 'write-v1',
            '--output-root', str(tmp_path)]
    assert main(args, dependencies=ActivationData()) == 0
    path = tmp_path / 'plans/write-v1/activation.prepared.json'
    config = json.loads(path.read_text())
    marker_path = path.parent / 'prepare-complete.json'
    marker = json.loads(marker_path.read_text())
    del marker['files'][config['audit_path']]
    marker_path.write_text(json.dumps(marker))
    (tmp_path / config['audit_path']).unlink()
    assert main(args, dependencies=ActivationData()) == 1


def test_insufficient_control_pool_records_failure_without_a_completion_marker(tmp_path):
    source = completed_source(tmp_path)

    class ShortPool(ActivationData):
        def stream_fineweb(self, config, revision):
            yield {'id': 'only-one', 'text': 'x' * 160}

    assert main(['activation-prepare', '--source-config', str(source), '--name', 'short-pool',
                 '--output-root', str(tmp_path)], dependencies=ShortPool()) == 1
    plan = tmp_path / 'plans/short-pool'
    assert not (plan / 'prepare-complete.json').exists()
    status = json.loads((plan / 'meta/status.json').read_text())
    assert status['state'] == 'failed'
    assert 'eligible' in status['error']


def test_control_tokens_in_web_text_are_not_counted_as_content(tmp_path):
    source = completed_source(tmp_path)

    class ControlText(ActivationData):
        def stream_fineweb(self, config, revision):
            for index in range(2000):
                yield {'id': str(index), 'text': 'x' * 10 + '<end>' + 'y' * 150}

    assert main(['activation-prepare', '--source-config', str(source), '--name', 'controls',
                 '--output-root', str(tmp_path)], dependencies=ControlText()) == 0
    path = tmp_path / 'plans/controls/activation.prepared.json'
    config = json.loads(path.read_text())
    rows = json.loads((tmp_path / config['items_path']).read_text())
    for row in rows:
        if row['corpus'] != 'fineweb':
            continue
        assert sum(row['masks']['user']) == 128
        assert all(not counted for token, counted in zip(row['input_ids'], row['masks']['user']) if token == 3)
        assert row['content'] == 'x' * 10 + '<end>' + 'y' * 118
