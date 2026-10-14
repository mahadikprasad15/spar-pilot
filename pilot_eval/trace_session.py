"""Pilot 5 bootstrap: read-only restoration and explicit, bounded readiness checks.

No model libraries are imported in the notebook process. This module does not
qualify capture settings or choose scientific analysis rules.
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


@contextmanager
def _writer(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Pilot 5 bootstrap has an active writer; wait for it to finish.') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class TraceSession:
    """Saved-workflow interface shared by the notebook and offline tests."""

    def __init__(self, root, repo, name, *, environment_probe=None, process_probe=None,
                 source_loader=None):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
            raise ValueError('Use a simple, separately named Pilot 5 experiment.')
        self.root, self.repo, self.name = Path(root).resolve(), Path(repo).resolve(), name
        self.directory = self.root / 'runs' / 'pilot-5' / name / 'bootstrap'
        self.environment_probe = environment_probe or self.inspect_environment
        self.process_probe = process_probe or self.active_children
        self.source_loader = source_loader or self.load_source

    def artifact(self, path):
        path = Path(path)
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError('Source must be inside the selected artifact root.')
        return resolved

    def restore(self):
        selection = self.directory / 'selection.json'
        selected = json.loads(selection.read_text()) if selection.is_file() else {}
        audit_path = self.directory / 'source-audit.json'
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
        sources = {}
        for label, relative in selected.items():
            path = self.artifact(relative)
            previous = audit.get(label, {})
            state = 'missing' if not path.is_file() else 'pending-audit'
            if state != 'missing' and previous.get('config_sha256') == _digest(path):
                state = 'audited-config; revalidate before capture'
            sources[label] = dict(path=str(path), state=state)
        smoke = self.directory / 'smoke.json'
        receipt = self.directory / 'reconnect.json'
        smoke_state = 'missing'
        if smoke.exists():
            seal = json.loads((self.directory / 'smoke-seal.json').read_text())
            if seal['sha256'] != _digest(smoke):
                raise ValueError('Durable smoke payload hash mismatch.')
            smoke_state = 'pending-reconnect'
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record['smoke_sha256'] != _digest(smoke):
                    raise ValueError('Smoke artifact changed after reconnect verification.')
                smoke_state = 'completed'
        sites = [dict(name='embedding-output', normalization='before-blocks')]
        sites.extend(dict(name=f'block-{index}-output', block=index,
                          normalization='before-final-norm') for index in range(28))
        return dict(name=self.name, sources=sources, smoke=dict(state=smoke_state),
                    sites=sites, positions={'content': 'relative to counted content tokens',
                                           'absolute': 'index in exact model input'},
                    grading={'schema_version': 1, 'status': 'not-requested'},
                    capabilities={'cohort_position_means': 'all 29 sites',
                                  'example_primary_window': 'all 29 sites, positions 0–4 pooled',
                                  'example_positions': [14, 17, 18, 19, 20, 21]},
                    target_validation=('completed' if smoke_state == 'completed'
                        and json.loads(smoke.read_text())['runtime'].get('host_kind') == 'colab'
                        else 'pending'))

    def select_sources(self, sources):
        if not sources or set(sources) - {'sft', 'grpo', 'random'}:
            raise ValueError('Select explicit sft/grpo/random prepared configurations.')
        selected = {}
        for label, path in sources.items():
            if not isinstance(path, (str, Path)) or not str(path).strip():
                raise ValueError(f'Ambiguous {label} source: supply exactly one path.')
            selected[label] = str(self.artifact(path).relative_to(self.root))
        with _writer(self.directory):
            _save(self.directory / 'selection.json', selected)
        return self.restore()

    @staticmethod
    def load_source(path, root):
        from pilot_eval.activation_prepare import load_prepared
        config, _ = load_prepared(path, root)
        return config

    def audit_sources(self):
        selected = self.restore()['sources']
        if set(selected) != {'sft', 'grpo', 'random'}:
            raise ValueError('Select all three existing prepared sources: sft, grpo, random.')
        with _writer(self.directory):
            audit = {}
            identities = []
            for label, row in selected.items():
                path = Path(row['path'])
                if not path.is_file():
                    raise ValueError(f'Missing {label} source: {path}')
                config = self.source_loader(path, self.root)
                expected = 'pilot3-write-v1' if label == 'sft' else 'pilot4-writes-v1'
                if config.get('protocol_version') != expected:
                    raise ValueError(f'{label} source has the wrong protocol.')
                kind = {'grpo': 'grpo', 'random': 'random-rank1-control'}.get(label)
                if kind and config.get('source_contract', {}).get('kind') != kind:
                    raise ValueError(f'{label} source has the wrong adapter kind.')
                identities.append(tuple(config.get(key) for key in
                    ('model', 'model_revision', 'tokenizer_revision', 'dtype'))
                    + (config['source_evidence']['base_sha256'],))
                audit[label] = dict(config_sha256=_digest(path), model=config['model'],
                                    checkpoint_steps=config['checkpoint_steps'],
                                    source_evidence=config['source_evidence'])
            if len(set(identities)) != 1:
                raise ValueError('Selected model/tokenizer/precision/base source identities differ.')
            _save(self.directory / 'source-audit.json', audit)
        return audit

    @staticmethod
    def active_children():
        from pilot_eval.notebook_session import NotebookSession
        return NotebookSession.active_children()

    def inspect_environment(self):
        # A timeout bounds import problems. No weights or tokenizer are downloaded.
        code = '''
import importlib.metadata, importlib.util, json, platform
from pilot_eval.sft_backend import PINS
versions = {key: importlib.metadata.version(key) for key in PINS}
wrong = [key for key in PINS if versions[key].split('+')[0] != PINS[key]]
if wrong: raise RuntimeError('Install pinned dependencies first: ' + ', '.join(wrong))
import torch
from transformers import PreTrainedModel, Qwen2Config, Qwen2ForCausalLM
import peft
from trl import GRPOTrainer
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Select exactly one CUDA GPU, then restore setup.')
print('TRACE_RUNTIME=' + json.dumps(dict(python=platform.python_version(), versions=versions,
    host_kind='colab' if importlib.util.find_spec('google.colab') else 'other',
    cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
    memory_bytes=torch.cuda.get_device_properties(0).total_memory)))
'''
        env = dict(os.environ, PYTHONPATH=str(self.repo) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        try:
            result = subprocess.run([sys.executable, '-u', '-c', code], cwd=self.repo,
                                    env=env, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Model-library imports timed out before inference; inspect dependencies.') from exc
        if result.returncode:
            raise RuntimeError('GPU import smoke failed before loading weights.\n' + result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith('TRACE_RUNTIME='))
        return json.loads(line.split('=', 1)[1])

    def readiness(self, *, expected_runtime=None):
        active = self.process_probe()
        if active:
            raise RuntimeError('Active experiment child: finish it before checking a new GPU stage. ' + json.dumps(active))
        runtime = self.environment_probe()
        if expected_runtime is not None and runtime != expected_runtime:
            raise ValueError('Runtime changed; select a new qualified execution before inference.')
        return runtime

    def write_smoke(self):
        started = time.monotonic()
        runtime = self.readiness()
        with _writer(self.directory):
            path = self.directory / 'smoke.json'
            if path.exists():
                previous = json.loads(path.read_text())
                if previous['runtime'] != runtime:
                    raise ValueError('Smoke runtime changed; use a new smoke experiment name.')
                return previous
            record = dict(schema_version=1, runtime=runtime, origin_pid=os.getpid(),
                          origin_session=os.environ.get('COLAB_RELEASE_TAG', ''),
                          nonce=uuid.uuid4().hex, elapsed_seconds=time.monotonic() - started,
                          scope='actual imports and durable small-file round trip; no model inference')
            _save(path, record)
            if json.loads(path.read_text()) != record:
                raise ValueError('Durable smoke readback failed.')
            _save(self.directory / 'smoke-seal.json', {'sha256': _digest(path)})
        return record

    def verify_reconnect(self):
        started = time.monotonic()
        with _writer(self.directory):
            path = self.directory / 'smoke.json'
            seal = json.loads((self.directory / 'smoke-seal.json').read_text())
            if _digest(path) != seal['sha256']:
                raise ValueError('Durable smoke payload hash mismatch.')
            record = json.loads(path.read_text())
            if record['origin_pid'] == os.getpid():
                raise ValueError('Reconnect in a fresh notebook kernel before verifying storage.')
            receipt = dict(schema_version=1, smoke_sha256=seal['sha256'],
                           restored_pid=os.getpid(), elapsed_seconds=time.monotonic() - started)
            _save(self.directory / 'reconnect.json', receipt)
        return self.restore()
