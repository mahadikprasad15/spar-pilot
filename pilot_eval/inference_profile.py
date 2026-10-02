"""GPU-independent batch profiling, isolated from scientific evaluation runs."""
import gc
import json
import time
from pathlib import Path

from pilot_eval.config import validate_config
from pilot_eval.sft import read_artifact, safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import HFDependencies, _hash, _save_frozen
from pilot_eval.run import _write_json


class ProfileDependencies(HFDependencies):
    clock = staticmethod(time.perf_counter)

    @staticmethod
    def synchronize():
        import torch
        torch.cuda.synchronize()

    @staticmethod
    def reset_memory():
        import torch
        torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def memory():
        import torch
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    total_bytes=torch.cuda.get_device_properties(0).total_memory)

    @staticmethod
    def is_oom(error):
        import torch
        return isinstance(error, torch.cuda.OutOfMemoryError)

    @staticmethod
    def cleanup():
        import torch
        gc.collect()
        torch.cuda.empty_cache()


def profile_inference(config_path, output_root, name, *, dtype='float32',
                      batch_sizes=(1, 2, 4, 8), sample_size=8, repeats=1, dependencies=None):
    if (dtype not in ('float32', 'float16', 'bfloat16') or sample_size < 8 or repeats < 1
            or not batch_sizes or any(type(b) is not int or b < 1 or b > sample_size for b in batch_sizes)
            or len(set(batch_sizes)) != len(batch_sizes)):
        raise ValueError('invalid profiler precision/sample/repeats/batch sizes')
    root = Path(output_root).resolve()
    source = validate_config(json.loads(Path(config_path).read_text()))
    if source['scorer'] != 'gsm8k' or source['adapter'] is not None:
        raise ValueError('profiler requires an untuned GSM8K source config')
    items = read_artifact(root, source['items_path'])
    if _hash(items) != source['items_sha256'] or len(items) < sample_size:
        raise ValueError('source item hash/count mismatch')
    ordered = sorted(items, key=lambda item: (item['prompt_tokens'], item['id']))
    sample = [ordered[round(i * (len(ordered) - 1) / (sample_size - 1))] for i in range(sample_size)]
    inference = {**source, 'dtype': dtype, 'protocol_version': 'pilot2-eval-v1'}
    validate_config(inference)
    deps = dependencies or ProfileDependencies()
    directory = root / 'runs/diagnostics/inference-profile' / safe_name(name)
    with run_lock(directory):
        config = dict(source_sha256=_hash(source), dtype=dtype, batch_sizes=list(batch_sizes),
            sample_ids=[item['id'] for item in sample], items_sha256=_hash(sample),
            repeats=repeats, decoding=source['decoding'], warmup_new_tokens=8,
            runtime=deps.runtime(inference))
        _save_frozen(directory / 'config.json', config)
        _save_frozen(directory / 'inputs/items.json', sample)
        backend = None
        measurements, output_sets = [], {}
        total = len(batch_sizes)
        _write_json(directory / 'meta/status.json', dict(state='running', completed=0, total=total))
        try:
            for size in batch_sizes:
                candidate = directory / 'candidates' / f'batch-{size}'
                saved_path = candidate / 'results.json'
                responses = candidate / 'responses.jsonl'
                if saved_path.exists():
                    row = json.loads(saved_path.read_text())
                    if row['config_sha256'] != _hash(config):
                        raise ValueError('candidate config hash mismatch')
                    if row['status'] == 'completed':
                        if file_hash(responses) != row['responses_sha256']:
                            raise ValueError('candidate response hash mismatch')
                        output_sets[size] = [json.loads(line) for line in responses.read_text().splitlines()]
                else:
                    if backend is None:
                        backend = deps.load_backend(inference)
                    outputs = []
                    timings = []
                    try:
                        # Short warmup initializes kernels for this shape, outside measurement.
                        backend.generate_batch([item['prompt'] for item in sample[:size]],
                            {**source['decoding'], 'max_new_tokens': 8})
                        deps.synchronize()
                        deps.reset_memory()
                        for repeat in range(repeats):
                            deps.synchronize()
                            start = deps.clock()
                            for offset in range(0, sample_size, size):
                                group = sample[offset:offset + size]
                                produced = backend.generate_batch([item['prompt'] for item in group], source['decoding'])
                                if len(produced) != len(group):
                                    raise ValueError('profiler output count mismatch')
                                outputs.extend(dict(id=item['id'], repeat=repeat, **output)
                                               for item, output in zip(group, produced))
                            deps.synchronize()
                            timings.append(deps.clock() - start)
                        seconds = sum(timings)
                        if seconds <= 0:
                            raise ValueError('profiler timer must advance')
                        tokens = sum(output['token_count'] for output in outputs)
                        count = sample_size * repeats
                        estimate = seconds / count * 150 / 60
                        responses.parent.mkdir(parents=True, exist_ok=True)
                        responses.write_text(''.join(json.dumps(output) + '\n' for output in outputs))
                        row = dict(batch_size=size, status='completed', seconds=seconds,
                            repeat_seconds=timings, generated_tokens=tokens, items_per_second=count / seconds,
                            tokens_per_second=tokens / seconds, estimated_150_minutes=estimate,
                            estimated_six_evaluations_minutes=6 * estimate, **deps.memory(),
                            responses_sha256=file_hash(responses), config_sha256=_hash(config))
                        output_sets[size] = outputs
                    except RuntimeError as error:
                        if not deps.is_oom(error):
                            raise
                        row = dict(batch_size=size, status='oom', error=str(error), config_sha256=_hash(config))
                    # Release temporaries/cache after the OOM exception scope has exited.
                    deps.cleanup()
                    _write_json(saved_path, row)
                measurements.append(row)
                _write_json(directory / 'meta/status.json', dict(state='running', completed=len(measurements), total=total))
                print(f"Profile batch {size}: {row['status']}" +
                      (f"; {row['tokens_per_second']:.2f} tokens/sec; ~{row['estimated_150_minutes']:.1f} min/150" if row['status']=='completed' else ''), flush=True)
            successful = [row for row in measurements if row['status'] == 'completed']
            best = max(successful, key=lambda row: row['items_per_second']) if successful else None
            reference = min(output_sets) if output_sets else None
            differences = {}
            if reference is not None:
                first = {r['id']: r for r in output_sets[reference] if r['repeat'] == 0}
                for size, outputs in output_sets.items():
                    differences[str(size)] = [r['id'] for r in outputs if r['repeat'] == 0 and any(
                        r[field] != first[r['id']][field] for field in ('text', 'token_count', 'stop_reason'))]
            result = dict(measurements=measurements, recommended_batch_size=best['batch_size'] if best else None,
                reference_batch_size=reference, output_differences=differences, sample_size=sample_size,
                repeats=repeats, scientific_settings_changed=False,
                limitations='Small untuned sample, not a full-cohort memory guarantee. Setup/training excluded; checkpoint lengths may differ. Recommendation is provisional; inspect memory headroom and output differences.')
            _write_json(directory / 'results/results.json', result)
            _write_json(directory / 'meta/status.json', dict(state='completed', completed=total, total=total))
            return result
        except (RuntimeError, ValueError, OSError) as error:
            _write_json(directory / 'meta/status.json', dict(state='failed', error=str(error), completed=len(measurements), total=total))
            raise
