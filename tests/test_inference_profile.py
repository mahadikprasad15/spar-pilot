import json
import pytest
from test_sft_prepare import TrainingData, source_plan

class OOM(RuntimeError):
    pass

class ProfileDependencies(TrainingData):
    def __init__(self):
        self.calls = []
        self.tick = 0
        self.cleaned = 0
    def synchronize(self): pass
    def reset_memory(self): pass
    def memory(self): return {'peak_allocated_bytes': 100, 'peak_reserved_bytes': 150, 'total_bytes': 1000}
    def clock(self):
        self.tick += 1
        return self.tick * 10
    def is_oom(self, error): return isinstance(error, OOM)
    def cleanup(self): self.cleaned += 1
    def load_backend(self, config):
        owner = self
        class Backend:
            def generate_batch(self, prompts, decoding):
                owner.calls.append((len(prompts), decoding['max_new_tokens']))
                if len(prompts) == 4: raise OOM('test GPU full')
                return [dict(text='#### 2', token_count=4, stop_reason='eos') for _ in prompts]
        return Backend()


def test_profiler_preserves_source_continues_after_oom_and_reuses_candidates(tmp_path):
    from pilot_eval.inference_profile import profile_inference
    source = source_plan(tmp_path)
    old = source.read_bytes()
    deps = ProfileDependencies()
    result = profile_inference(source, tmp_path, 'profile', dependencies=deps)
    assert [r['batch_size'] for r in result['measurements']] == [1, 2, 4, 8]
    assert [r['status'] for r in result['measurements']] == ['completed', 'completed', 'oom', 'completed']
    assert result['measurements'][0]['estimated_150_minutes'] == 3.125
    assert result['measurements'][0]['generated_tokens'] == 32
    assert deps.cleaned >= 1
    assert source.read_bytes() == old
    assert any(batch == 8 and cap == 1024 for batch, cap in deps.calls)
    assert result['output_differences']['8'] == []
    before = list(deps.calls)
    assert profile_inference(source, tmp_path, 'profile', dependencies=deps) == result
    assert deps.calls == before
    with pytest.raises(ValueError, match='different|mismatch'):
        profile_inference(source, tmp_path, 'profile', dtype='float16', dependencies=deps)


def test_non_memory_errors_stop_and_completed_candidates_resume(tmp_path):
    from pilot_eval.inference_profile import profile_inference
    source = source_plan(tmp_path)
    deps = ProfileDependencies()
    original_loader = deps.load_backend
    def load(config):
        backend = original_loader(config)
        generate = backend.generate_batch
        def interrupted(prompts, decoding):
            if len(prompts) == 2:
                raise RuntimeError('unrelated backend failure')
            return generate(prompts, decoding)
        backend.generate_batch = interrupted
        return backend
    deps.load_backend = load
    with pytest.raises(RuntimeError, match='unrelated'):
        profile_inference(source, tmp_path, 'resume', dependencies=deps)
    deps.load_backend = original_loader
    deps.calls.clear()
    result = profile_inference(source, tmp_path, 'resume', dependencies=deps)
    assert result['measurements'][0]['status'] == 'completed'
    assert not any(batch == 1 for batch, cap in deps.calls)
    assert result['measurements'][-1]['status'] == 'completed'


def test_standalone_script_help_works_without_editable_package_install():
    import os
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).parents[1] / 'scripts/profile_inference.py'
    env = dict(os.environ, PYTHONPATH='')
    result = subprocess.run([sys.executable, str(script), '--help'], cwd='/tmp',
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--batch-sizes' in result.stdout
