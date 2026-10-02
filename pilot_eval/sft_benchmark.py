"""Disposable inference throughput measurements; never scientific run responses."""
import json
import time
from pathlib import Path

from pilot_eval.sft import load_sft
from pilot_eval.training import TrainingDependencies, training_directory, run_lock, file_hash
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval.run import _write_json


class BenchmarkDependencies(TrainingDependencies):
    @staticmethod
    def synchronize():
        import torch
        torch.cuda.synchronize()

    @staticmethod
    def reset_peak_memory():
        import torch
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def peak_memory():
        import torch
        return torch.cuda.max_memory_allocated()

    clock = staticmethod(time.perf_counter)


def benchmark_sft(config_path, output_root, *, dependencies=None):
    config, _, items = load_sft(config_path, output_root)
    root = Path(output_root).resolve()
    deps = dependencies or BenchmarkDependencies()
    training = training_directory(root, config)
    with run_lock(training):
        if not (training / 'results/preflight.json').exists():
            raise ValueError('passing GPU preflight is required before benchmark')
        runtime = deps.runtime(config)
        if runtime != json.loads((training / 'meta/runtime.json').read_text()):
            raise ValueError('benchmark/training runtime mismatch')
        # Include prompt-length endpoints and evenly spaced ranks, with no gold-based selection.
        ordered = sorted(items, key=lambda item: (item['prompt_tokens'], item['id']))
        sample = [ordered[round(i * (len(ordered) - 1) / 7)] for i in range(8)]
        directory = training / 'benchmark'
        benchmark_config = dict(training_config_sha256=_hash(config), runtime=runtime,
            sample_ids=[item['id'] for item in sample], items_sha256=_hash(sample),
            batch_sizes=[1, 2], decoding=config['source_config']['decoding'],
            warmup_items=2, selection='eight evenly spaced prompt-length ranks')
        _save_frozen(directory / 'config.json', benchmark_config)
        destination = directory / 'results/results.json'
        responses = directory / 'results/responses.jsonl'
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved['responses_sha256'] != file_hash(responses):
                raise ValueError('benchmark response hash mismatch')
            if saved['config_sha256'] != _hash(benchmark_config):
                raise ValueError('benchmark config hash mismatch')
            return saved
        _write_json(directory / 'meta/status.json', dict(state='running', completed=0, total=2))
        try:
            backend = deps.load_backend({**config['source_config'], 'dtype': 'float32',
                                         'adapter': None, 'adapter_revision': None})
            # Warm up both batch shapes; excluded from timings and saved scientific evidence.
            for size in (1, 2):
                backend.generate_batch([item['prompt'] for item in sample[:size]], benchmark_config['decoding'])
            records, measurements = [], []
            for size in (1, 2):
                deps.synchronize()
                deps.reset_peak_memory()
                start = deps.clock()
                outputs = []
                for offset in range(0, len(sample), size):
                    group = sample[offset:offset + size]
                    produced = backend.generate_batch([item['prompt'] for item in group], benchmark_config['decoding'])
                    if len(produced) != len(group):
                        raise ValueError('benchmark backend output count mismatch')
                    outputs.extend(produced)
                deps.synchronize()
                seconds = deps.clock() - start
                if seconds <= 0:
                    raise ValueError('benchmark timer must advance')
                tokens = sum(output['token_count'] for output in outputs)
                estimate = seconds * 150 / len(sample) / 60
                measurements.append(dict(batch_size=size, seconds=seconds, generated_tokens=tokens,
                    items_per_second=len(sample) / seconds, tokens_per_second=tokens / seconds,
                    peak_allocated_bytes=deps.peak_memory(), estimated_150_minutes=estimate,
                    estimated_six_evaluations_minutes=6 * estimate))
                records.extend(dict(id=item['id'], batch_size=size, **output)
                               for item, output in zip(sample, outputs))
                responses.parent.mkdir(parents=True, exist_ok=True)
                responses.write_text(''.join(json.dumps(record) + '\n' for record in records))
                _write_json(directory / 'meta/status.json', dict(state='running', completed=len(measurements), total=2))
            singles = {r['id']: r for r in records if r['batch_size'] == 1}
            differences = [r['id'] for r in records if r['batch_size'] == 2 and any(
                r[field] != singles[r['id']][field] for field in ['text', 'token_count', 'stop_reason'])]
            result = dict(sample_size=len(sample), sample_ids=benchmark_config['sample_ids'],
                measurements=measurements, output_differences=differences,
                selected_evaluation_batch_size=config.get('evaluation_batch_size', 1),
                config_sha256=_hash(benchmark_config), responses_sha256=file_hash(responses),
                limitations='Eight untuned prompts; estimates exclude loading/training and checkpoint lengths may differ.')
            _write_json(destination, result)
            _write_json(directory / 'meta/status.json', dict(state='completed', completed=2, total=2))
            return result
        except (RuntimeError, ValueError, OSError) as exc:
            _write_json(directory / 'meta/status.json', dict(state='failed', error=str(exc), total=2))
            raise
