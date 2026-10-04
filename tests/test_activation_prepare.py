"""Public Pilot 3 preparation tests: real files, controlled offline source access."""

import json

from pilot_eval.cli import main
from pilot_eval.sft import prepare_sft
from pilot_eval.training import run_sft, training_directory, seal_checkpoint
from test_sft_prepare import TrainingData, ChatTokenizer, source_plan
from test_sft_training import TrainingBoundary, Engine


class OffsetTokenizer(ChatTokenizer):
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
