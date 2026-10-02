import json
import pytest
from test_sft_prepare import TrainingData, source_plan
from test_sft_training import TrainingBoundary, Engine

class BenchmarkBoundary(TrainingBoundary):
    def __init__(self):
        self.engine = Engine()
        self.calls = []
        self.ticks = 0
    def synchronize(self):
        pass
    def reset_peak_memory(self):
        pass
    def peak_memory(self):
        return 1234
    def clock(self):
        self.ticks += 1
        return self.ticks * 10
    def load_backend(self, config):
        owner = self
        class Backend:
            def generate_batch(self, prompts, decoding):
                owner.calls.append(len(prompts))
                return [dict(text='#### 2', token_count=4, stop_reason='eos') for _ in prompts]
        return Backend()


def test_benchmark_times_both_batches_saves_outputs_and_reuses_verified_result(tmp_path):
    from pilot_eval.sft import prepare_sft
    from pilot_eval.training import run_sft, training_directory
    from pilot_eval.sft_benchmark import benchmark_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'l4', hardware='L4',
                       evaluation_batch_size=2, dependencies=TrainingData())
    deps = BenchmarkBoundary()
    run_sft(path, tmp_path, preflight_only=True, dependencies=deps)
    result = benchmark_sft(path, tmp_path, dependencies=deps)
    assert result['sample_size'] == 8
    assert [r['batch_size'] for r in result['measurements']] == [1, 2]
    assert result['measurements'][0]['generated_tokens'] == 32
    assert result['measurements'][0]['estimated_150_minutes'] == 3.125
    assert result['measurements'][0]['estimated_six_evaluations_minutes'] == 18.75
    assert result['output_differences'] == []
    calls = list(deps.calls)
    assert benchmark_sft(path, tmp_path, dependencies=deps) == result
    assert deps.calls == calls
    directory = training_directory(tmp_path, json.loads(path.read_text()))
    assert not list((tmp_path / 'runs/pilot-2-eval').rglob('responses.jsonl'))
    response = directory / 'benchmark/results/responses.jsonl'
    assert len(response.read_text().splitlines()) == 16
    response.write_text('corrupted')
    with pytest.raises(ValueError, match='hash'):
        benchmark_sft(path, tmp_path, dependencies=deps)


def test_benchmark_records_oom_without_starting_scientific_evaluation(tmp_path):
    from pilot_eval.sft import prepare_sft
    from pilot_eval.training import run_sft, training_directory
    from pilot_eval.sft_benchmark import benchmark_sft
    path = prepare_sft(source_plan(tmp_path), tmp_path, 'l4', hardware='L4',
                       evaluation_batch_size=2, dependencies=TrainingData())
    deps = BenchmarkBoundary()
    run_sft(path, tmp_path, preflight_only=True, dependencies=deps)
    class OOM:
        def generate_batch(self, prompts, decoding):
            raise RuntimeError('CUDA out of memory')
    deps.load_backend = lambda config: OOM()
    with pytest.raises(RuntimeError, match='out of memory'):
        benchmark_sft(path, tmp_path, dependencies=deps)
    training = training_directory(tmp_path, json.loads(path.read_text()))
    status = json.loads((training / 'benchmark/meta/status.json').read_text())
    assert status['state'] == 'failed'
    assert not (training / 'benchmark/results/results.json').exists()
    assert not (tmp_path / 'runs/pilot-2-eval').exists()
