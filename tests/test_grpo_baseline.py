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


def test_interrupted_sampling_resumes_groups_and_preserves_settings(tmp_path, capsys):
    path, settings = prepared(tmp_path)
    deps = Boundary()
    deps.sampler.fail_at = 3
    assert command(tmp_path, path, settings, deps) == 1
    config = json.loads((tmp_path / 'plans/sampling/baseline.config.json').read_text())
    directory = tmp_path / config['run_path']
    protected = {p: p.read_bytes() for p in (directory / 'batches').glob('*')}
    assert json.loads((directory / 'meta/status.json').read_text())['state'] == 'failed'
    deps.sampler.fail_at = None
    assert command(tmp_path, path, settings, deps) == 0
    assert deps.sampler.calls == 65  # 64 successful batches plus the interrupted call.
    assert all(p.read_bytes() == b for p, b in protected.items())
    changed = json.loads(settings.read_text()); changed['top_p'] = .9
    settings.write_text(json.dumps(changed))
    assert command(tmp_path, path, settings, deps) == 1
    assert deps.sampler.calls == 65
    assert 'mismatch' in capsys.readouterr().err


def test_capped_and_ambiguous_answers_have_zero_reward_and_censored_p99(tmp_path):
    path, settings = prepared(tmp_path)
    deps = Boundary()
    def sample(rows, settings, seed):
        return [dict(text=row['gold'] if d < 4 else 'Final answer: 72 apples and 5 pears',
                     token_ids=[7] * (4 if d < 4 else 2), stop_reason='cap' if d < 4 else 'eos')
                for row in rows for d in range(8)]
    deps.sampler.sample = sample
    assert command(tmp_path, path, settings, deps) == 0
    config = json.loads((tmp_path / 'plans/sampling/baseline.config.json').read_text())
    directory = tmp_path / config['run_path']
    summary = json.loads((directory / 'results/results.json').read_text())
    assert summary['cap_count'] == 512 and summary['flexible_accuracy'] == 0
    assert summary['p99_censored'] is True and summary['final_cap_evidence_eligible'] is False
    assert summary['final_cap'] is None
    records = [json.loads(line) for line in (directory / 'results/responses.jsonl').read_text().splitlines()]
    assert all(r['reward'] == 0 and not r['flexible']['correct'] for r in records)


def test_settings_and_saved_shards_reject_mismatch_before_model_loading(tmp_path):
    path, settings = prepared(tmp_path)
    value = json.loads(settings.read_text()); value['scorer'] = 'gsm8k-flexible-v2'
    settings.write_text(json.dumps(value))
    deps = Boundary()
    assert command(tmp_path, path, settings, deps) == 1
    assert deps.loads == 0
    value['scorer'] = 'gsm8k-flexible-v3'; settings.write_text(json.dumps(value))
    assert command(tmp_path, path, settings, deps) == 0
    config = json.loads((tmp_path / 'plans/sampling/baseline.config.json').read_text())
    shard = tmp_path / config['run_path'] / 'batches/0000.json'
    shard.write_text('[]')
    assert command(tmp_path, path, settings, deps) == 1
    assert deps.loads == 1


def test_saved_rewards_are_bound_to_scorer_implementation(tmp_path):
    path, settings = prepared(tmp_path)
    deps = Boundary()
    assert command(tmp_path, path, settings, deps) == 0
    config = json.loads((tmp_path / 'plans/sampling/baseline.config.json').read_text())
    assert len(config['implementation_sha256']['scorer']) == 64
    assert len(config['implementation_sha256']['sampling']) == 64
    assert config['batches'][0]['seed'] == 42
    assert config['batches'][1]['seed'] == 43


def test_inconsistent_cap_metadata_cannot_seal_a_batch(tmp_path):
    path, settings = prepared(tmp_path)
    deps = Boundary()
    deps.sampler.sample = lambda rows, settings, seed: [dict(text=r['gold'], token_ids=[7], stop_reason='cap') for r in rows for _ in range(8)]
    assert command(tmp_path, path, settings, deps) == 1
    config = json.loads((tmp_path / 'plans/sampling/baseline.config.json').read_text())
    assert not list((tmp_path / config['run_path']).glob('batches/*.complete.json'))


def test_cached_runtime_evidence_is_verified_before_reuse(tmp_path):
    path, settings = prepared(tmp_path)
    deps = Boundary()
    assert command(tmp_path, path, settings, deps) == 0
    config = json.loads((tmp_path / 'plans/sampling/baseline.config.json').read_text())
    runtime = tmp_path / config['run_path'] / 'meta/runtime.json'
    runtime.write_text('{}')
    assert command(tmp_path, path, settings, deps) == 1
    assert deps.loads == 1
