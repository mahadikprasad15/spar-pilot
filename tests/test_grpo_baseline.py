"""Public sampling workflow with controlled external generation; no downloads."""
import json
import pytest
from pilot_eval.cli import main
from test_grpo_prepare import sources


def prepared(root):
    sft, measurement = sources(root)
    assert main(['grpo-prepare', '--sft-config', str(sft), '--measurement-config', str(measurement), '--name', 'grpo', '--output-root', str(root)]) == 0
    path = root / 'plans/grpo/grpo.prepared.json'
    plan = json.loads(path.read_text())
    settings = dict(subset_ids=plan['optimizer_prompt_order'][:128], scorer='gsm8k-flexible-v3', seed=42,
                    groups_per_batch=2, max_new_tokens=4, temperature=1.0, top_p=1.0, top_k=0)
    settings_path = root / 'settings.json'
    settings_path.write_text(json.dumps(settings))
    return path, settings_path


class Sampler:
    def __init__(self):
        self.calls = 0
        self.fail_at = None
    def sample(self, rows, settings, seed):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('interrupted')
        return [dict(text=row['gold'], token_ids=[7], stop_reason='eos') for row in rows for _ in range(8)]
    def runtime(self):
        return {'controlled': True, 'dtype': 'float32'}
    def close(self):
        pass


class Boundary:
    def __init__(self):
        self.sampler = Sampler()
        self.loads = 0
    def load_sampler(self, plan):
        self.loads += 1
        return self.sampler


def command(root, path, settings, deps, name='sampling'):
    return main(['grpo-baseline', '--config', str(path), '--settings', str(settings), '--name', name, '--output-root', str(root)], dependencies=deps)


def test_sampling_saves_1024_draws_and_reuses_completed_run(tmp_path):
    path, settings = prepared(tmp_path)
    deps = Boundary()
    assert command(tmp_path, path, settings, deps) == 0
    run = tmp_path / 'plans/sampling/baseline.config.json'
    config = json.loads(run.read_text())
    directory = tmp_path / config['run_path']
    records = [json.loads(line) for line in (directory / 'results/responses.jsonl').read_text().splitlines()]
    assert len(records) == len({r['draw_id'] for r in records}) == 1024
    assert all(r['reward'] == 1 and r['token_ids'] == [7] and r['strict']['correct'] for r in records)
    summary = json.loads((directory / 'results/results.json').read_text())
    assert summary['dead_group_fraction'] == 1
    assert summary['flexible_accuracy'] == 1 and summary['cap_count'] == 0
    assert summary['length_percentiles']['99'] == 1
    saved = (directory / 'results/responses.jsonl').read_bytes()
    assert command(tmp_path, path, settings, deps) == 0
    assert deps.loads == 1 and deps.sampler.calls == 64
    assert (directory / 'results/responses.jsonl').read_bytes() == saved
