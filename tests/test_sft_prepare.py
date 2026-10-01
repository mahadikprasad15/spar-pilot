import json

import pytest

from pilot_eval.workflow import prepare_plan
from test_workflow import Dependencies


class ChatTokenizer:
    eos_token_id = 3
    pad_token_id = 0
    chat_template = "tiny Qwen-style chat"

    def apply_chat_template(self, messages, **kwargs):
        text = "<system>Qwen</system>" + "".join(
            f"<{m['role']}>" + m['content'] + "<end>" for m in messages)
        return text + ("<assistant>" if kwargs.get('add_generation_prompt') else "")

    def encode(self, text, **kwargs):
        if text.endswith(tuple(' ' + letter for letter in 'ABCD')):
            return self.encode(text[:-2]) + [1000 + 'ABCD'.index(text[-1])]
        parts = text.split("<end>")
        ids = []
        for i, part in enumerate(parts):
            ids.extend(ord(c) + 10 for c in part)
            if i < len(parts) - 1:
                ids.append(3)
        return ids


class TrainingData(Dependencies):
    tokenizer = ChatTokenizer()

    def load_dataset(self, path, name, revision):
        data = super().load_dataset(path, name, revision)
        if name == 'main':
            data['train'] = [dict(question=f'Train {i}', answer='2+1=<<2+1=3>>3\n#### 3')
                             for i in range(600)]
        return data


def source_plan(root, deps=None):
    # Only GSM8K source is needed; the existing preparation also freezes MMLU.
    return prepare_plan(root, 'source', dependencies=deps or TrainingData())[0]


def test_prepare_sft_freezes_complete_masked_targets_and_reuses_artifacts(tmp_path):
    from pilot_eval.sft import prepare_sft
    source = source_plan(tmp_path)
    config_path = prepare_sft(source, tmp_path, 'sft-v1', dependencies=TrainingData())
    config = json.loads(config_path.read_text())
    rows = json.loads((tmp_path / config['training_items_path']).read_text())
    assert len(rows) == 512
    assert len({r['id'] for r in rows}) == 512
    assert rows[0]['id'].startswith('gsm8k:train:')
    assert '<<2+1=3>>' in rows[0]['gold']
    row = rows[0]
    boundary = row['prompt_tokens']
    assert row['labels'][:boundary] == [-100] * boundary
    assert row['labels'][boundary:] == row['input_ids'][boundary:]
    assert row['labels'][-1] == 3
    assert config['max_length'] == max(len(r['input_ids']) for r in rows)
    assert config['dtype'] == 'float32'
    assert config['optimizer']['gradient_accumulation_steps'] == 8
    plan = json.loads((config_path.parent / 'analysis-plan.json').read_text())
    assert plan['mode'] == 'exploratory'
    assert plan['binary_collapse_threshold'] is None
    before = {p: p.read_bytes() for p in config_path.parent.rglob('*') if p.is_file()}
    assert prepare_sft(source, tmp_path, 'sft-v1', dependencies=TrainingData()) == config_path
    assert all(p.read_bytes() == data for p, data in before.items())


def test_sft_prepare_cli_and_corruption_checks(tmp_path):
    from pilot_eval.cli import main
    source = source_plan(tmp_path)
    args = ['sft-prepare', '--source-config', str(source), '--name', 'sft',
            '--output-root', str(tmp_path)]
    assert main(args, dependencies=TrainingData()) == 0
    path = tmp_path / 'plans/sft/sft.config.json'
    config = json.loads(path.read_text())
    items = tmp_path / config['training_items_path']
    rows = json.loads(items.read_text())
    rows[0]['labels'][0] = 1
    items.write_text(json.dumps(rows))
    assert main(args, dependencies=TrainingData()) == 1


@pytest.mark.parametrize('failure', ['overlap', 'context', 'template'])
def test_sft_prepare_rejects_unmatched_or_truncated_data(tmp_path, failure):
    from pilot_eval.sft import prepare_sft
    source = source_plan(tmp_path)

    class BrokenData(TrainingData):
        def load_tokenizer(self, model, revision):
            if failure == 'template':
                tokenizer = ChatTokenizer()
                tokenizer.chat_template = 'changed'
                return tokenizer, 4096
            return self.tokenizer, 10 if failure == 'context' else 4096

        def load_dataset(self, *args):
            data = super().load_dataset(*args)
            if failure == 'overlap':
                for row in data['train']:
                    row['question'] = 'Compute 1'
            return data

    with pytest.raises(ValueError, match={'overlap': 'overlap', 'context': 'overflow',
                                         'template': 'template'}[failure]):
        prepare_sft(source, tmp_path, 'broken', dependencies=BrokenData())


def test_sft_rejects_configuration_that_cannot_match_execution(tmp_path):
    from pilot_eval.sft import prepare_sft, load_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'sft', dependencies=TrainingData())
    config = json.loads(path.read_text())
    config['adapter']['rank'] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='configuration|config'):
        load_sft(path, tmp_path)
