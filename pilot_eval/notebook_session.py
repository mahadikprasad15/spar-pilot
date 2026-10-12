"""Notebook orchestration only: restore saved paths, guard runtime, stream CLI output.

Scientific commands remain the authority for hashes, gates and completion. This
module never infers verified completion from a status counter or changes a seal.
"""
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

GPU_COMMANDS = {
    'grpo-baseline', 'grpo-preflight', 'grpo-train', 'grpo-evaluate',
    'activation-calibration-prepare', 'activation-calibrate',
    'activation-calibration-validate', 'activation-profile', 'activation-measure',
    'activation-validate', 'activation-recover-measurement',
    'tokens-profile', 'tokens-measure',
}


def _read(path):
    return json.loads(Path(path).read_text())


def _argument(args, key):
    return args[args.index(key) + 1] if key in args else None


class NotebookSession:
    def __init__(self, root, repo, *, plan_name='pilot4-grpo-fp32-v1',
                 write_variant='', extra_write_products=None, environment_probe=None,
                 process_probe=None, heartbeat_seconds=15):
        self.root = Path(root).resolve()
        self.repo = Path(repo).resolve()
        self.plan_name = plan_name
        self.write_variant = write_variant
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', plan_name) or (
                write_variant and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', write_variant)):
            raise ValueError('Plan and write variant must be safe artifact names')
        self.extra_write_products = extra_write_products or {}
        self.environment_probe = environment_probe or self.inspect_environment
        self.process_probe = process_probe or self.active_children
        self.heartbeat_seconds = heartbeat_seconds

    @staticmethod
    def saved_selection(root, plan_name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', plan_name):
            raise ValueError('Invalid operational selection name')
        path = Path(root) / 'notebook' / 'operations' / plan_name / 'selection.json'
        return _read(path) if path.is_file() else {}

    def save_selection(self, selection):
        """Mutable notebook preferences only; scientific manifests remain immutable."""
        folder = self.root / 'notebook' / 'operations' / self.plan_name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'selection.json'
        temporary = folder / 'selection.tmp'
        previous = _read(path) if path.is_file() else {}
        temporary.write_text(json.dumps({**previous, **selection}, indent=2) + '\n')
        temporary.replace(path)

    def artifact(self, path):
        path = Path(path)
        path = (path if path.is_absolute() else self.root / path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Artifact path outside selected root')
        return path

    def plan(self, name, filename):
        return self.artifact(Path('plans') / name / filename)

    def write_name(self, label, kind):
        ending = self.write_variant or 'v1'
        return f'{self.plan_name}-{label}-write-{kind}-{ending}'

    def restore(self):
        """Read-only; safe in a fresh CPU kernel. Missing stages stay pending."""
        p = self.plan_name
        selected = self.saved_selection(self.root, p)
        state = dict(
            PREPARED_PATH=self.plan(p, 'grpo.prepared.json'),
            BASELINE_NAME='pilot4-untuned-sampling-v1',
            CONTROL_NAME='tiny-grpo-controls-v3-preflight-v1',
            TRAIN_RUN_NAME=p + '-arm-v1', BEHAVIOUR_NAME=p + '-behaviour-v1',
            TOKEN_NAME=p + '-supplemental-tokens-' + (self.write_variant or 'v1'),
            WRITE_PRODUCTS={'grpo': self.plan(p + '-writes-v1', 'activation.prepared.json'),
                            'random': self.plan(p + '-random-writes-v1', 'activation.prepared.json')},
            WRITE_CALIBRATIONS={}, WRITE_PROFILES={}, WRITE_EXECUTIONS={}, WRITE_REPORTS={},
        )
        for key in ['BASELINE_NAME', 'CONTROL_NAME', 'TRAIN_RUN_NAME', 'BEHAVIOUR_NAME', 'TOKEN_NAME',
                    'PREFLIGHT_NAME', 'FROZEN_NAME']:
            if key in selected:
                state[key] = selected[key]
        state['SAMPLING_SETTINGS_PATH'] = self.root / 'notebook' / state['BASELINE_NAME'] / 'settings.json'
        state['BASELINE_CONFIG_PATH'] = self.plan(state['BASELINE_NAME'], 'baseline.config.json')
        state['TRAINING_CONFIG_PATH'] = self.plan(state['TRAIN_RUN_NAME'], 'grpo.training.json')
        state['BEHAVIOUR_CONFIG_PATH'] = self.plan(state['BEHAVIOUR_NAME'], 'grpo.evaluation.json')
        if 'PREFLIGHT_NAME' in state:
            state['PREFLIGHT_PATH'] = self.plan(state['PREFLIGHT_NAME'], 'grpo.preflight.json')
            state['PREFLIGHT_SETTINGS_PATH'] = self.root / 'notebook' / state['PREFLIGHT_NAME'] / 'settings.json'
            if state['PREFLIGHT_SETTINGS_PATH'].is_file():
                state['PREFLIGHT_SETTINGS'] = _read(state['PREFLIGHT_SETTINGS_PATH'])
        if 'FROZEN_NAME' in state:
            state['FROZEN_PATH'] = self.plan(state['FROZEN_NAME'], 'grpo.frozen.json')
            state['REVIEW_PATH'] = self.root / 'notebook' / state['FROZEN_NAME'] / 'review.json'
        training = state['TRAINING_CONFIG_PATH']
        if training.is_file():
            config = _read(training)
            state['TRAINING_CONFIG'] = config
            state['TRAINING_DIR'] = self.artifact(config['run_path'])
            state['FROZEN_PATH'] = self.artifact(config['frozen_path'])
            state['FROZEN_NAME'] = state['FROZEN_PATH'].parent.name
            if state['FROZEN_PATH'].is_file():
                frozen = _read(state['FROZEN_PATH'])
                # Frozen plans bind the selected preflight; never pick the newest by time.
                preflight = frozen.get('preflight_path')
                if preflight:
                    state['PREFLIGHT_PATH'] = self.artifact(preflight)
                    state['PREFLIGHT_NAME'] = state['PREFLIGHT_PATH'].parent.name
                    state['PREFLIGHT_SETTINGS_PATH'] = self.root / 'notebook' / state['PREFLIGHT_NAME'] / 'settings.json'
                    if state['PREFLIGHT_SETTINGS_PATH'].is_file():
                        state['PREFLIGHT_SETTINGS'] = _read(state['PREFLIGHT_SETTINGS_PATH'])
            state['REVIEW_PATH'] = self.root / 'notebook' / state['FROZEN_NAME'] / 'review.json'
        for key in ['BASELINE', 'BEHAVIOUR']:
            path = state[key + '_CONFIG_PATH']
            if path.is_file():
                state[key + '_CONFIG'] = _read(path)
                state[key + '_DIR'] = self.artifact(state[key + '_CONFIG']['run_path'])
        state['WRITE_PRODUCTS'].update({k: self.artifact(v) for k,v in self.extra_write_products.items()})
        for label, prepared in state['WRITE_PRODUCTS'].items():
            calibration = self.plan(self.write_name(label, 'calibration'), 'activation.calibration.json')
            state['WRITE_CALIBRATIONS'][label] = calibration
            execution = self.plan(self.write_name(label, 'execution'), 'activation.execution.json')
            state['WRITE_REPORTS'][label] = self.root / 'reports' / self.write_name(label, 'report')
            state['WRITE_EXECUTIONS'][label] = execution
            if execution.is_file():
                saved = _read(execution)
                if label == 'grpo':
                    state['WRITE_BATCH_SIZE'] = saved['batch_size']
                    state['WRITE_PROFILE_NOTES'] = saved['review']['notes']
            if calibration.is_file():
                rule = self.artifact(_read(calibration)['run_path']) / 'rule.json'
                if rule.is_file() and label == 'grpo':
                    state['WRITE_CALIBRATION_NOTES'] = _read(rule)['review_notes']
            if prepared.is_file():
                matches = []
                for result in self.artifact(_read(prepared)['run_path']).glob('profile-v2-*/results.json'):
                    data = _read(result)
                    if data['identity'].get('calibration_path') == str(calibration.relative_to(self.root)):
                        matches.append(self.artifact(data['profile_path']))
                if len(matches) > 1:
                    raise RuntimeError(f'Multiple profiles for {label}; select evidence explicitly: {matches}')
                if matches:
                    state['WRITE_PROFILES'][label] = matches[0]
        token = self.plan(state['TOKEN_NAME'], 'tokens.prepared.json')
        state['TOKEN_PREPARED'] = token
        state['TOKEN_EXECUTION'] = self.plan(state['TOKEN_NAME'], 'tokens.execution.json')
        if token.is_file():
            state['TOKEN_CONFIG'] = _read(token)
            state['TOKEN_RUN_DIR'] = self.artifact(state['TOKEN_CONFIG']['run_path'])
        if 'sft' in state['WRITE_PRODUCTS']:
            state['SFT_WRITE_REPORT'] = selected.get('SFT_WRITE_REPORT', str(state['WRITE_REPORTS']['sft']))
            state['SFT_TOKEN_EXECUTION'] = str(state['WRITE_EXECUTIONS']['sft'])
        final_name = selected.get('FINAL_REPORT_NAME', p + '-final-handoff-' + (self.write_variant or 'v1'))
        state.update(FINAL_REPORT_NAME=final_name,
                     FINAL_SELECTION_PATH=self.plan(final_name, 'selection.json'),
                     FINAL_REPORT_DIR=self.root / 'reports' / final_name)
        return state

    def inventory(self):
        """Display saved progress, explicitly distinct from scientific verification."""
        rows = []
        for folder in sorted((self.root / 'plans').glob(self.plan_name + '*')):
            for path in sorted(folder.glob('*.json')):
                config = _read(path)
                if not isinstance(config, dict) or not config.get('run_path'):
                    continue
                directory = self.artifact(config['run_path'])
                status = directory / 'meta/status.json'
                snapshot = _read(status) if status.is_file() else {}
                if path.name == 'activation.calibration.json':
                    phases = {}
                    for phase in ['calibration', 'validation', 'validated']:
                        phase_status = directory / phase / 'meta/status.json'
                        phases[phase] = _read(phase_status) if phase_status.is_file() else {'state': 'pending'}
                    print(folder.name, '| calibration phases:',
                          {k: v.get('state') for k,v in phases.items()})
                rows.append(dict(name=folder.name, config=path.name,
                    state=snapshot.get('state', 'prepared/no-status'),
                    completed=snapshot.get('completed'), total=snapshot.get('total'),
                    device=config.get('runtime', {}).get('device'), error=snapshot.get('error')))
        print('Saved progress (completion seals are verified by each scientific command):')
        for row in rows:
            print(f"{row['name']} / {row['config']}: {row['state']} "
                  f"{row['completed']}/{row['total']} | GPU: {row['device']}"
                  + (f" | error: {row['error']}" if row['error'] else ''))
        return rows

    def inspect_environment(self):
        """Fresh child imports the actual model classes without loading any weights."""
        code = '''
import importlib.metadata, json, sys
from pilot_eval.sft_backend import PINS
versions = {k: importlib.metadata.version(k) for k in PINS}
errors = [f"{k}: {versions[k]} (expected {v})" for k,v in PINS.items() if versions[k].split('+')[0] != v]
if errors: raise RuntimeError('Run section 4: ' + '; '.join(errors))
import torch
from transformers import PreTrainedModel, Qwen2Config, Qwen2ForCausalLM
from trl import GRPOTrainer
import peft
result = dict(versions=versions, cuda_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count())
if result['cuda_available'] and result['gpu_count'] == 1:
    from pilot_eval.activation_workflow import HFActivationDependencies
    result['activation_runtime'] = HFActivationDependencies().runtime()
print('NOTEBOOK_ENVIRONMENT=' + json.dumps(result))
'''
        env = dict(os.environ, PYTHONPATH=str(self.repo) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        result = subprocess.run([sys.executable, '-u', '-c', code], cwd=self.repo,
                                env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Environment check failed before model loading. Run section 4.\n' + result.stdout + result.stderr)
        line = next(v for v in result.stdout.splitlines() if v.startswith('NOTEBOOK_ENVIRONMENT='))
        return json.loads(line.split('=', 1)[1])

    def check_environment(self, *, require_gpu=False):
        value = self.environment_probe()
        if require_gpu and (not value['cuda_available'] or value['gpu_count'] != 1):
            raise RuntimeError('Select a GPU runtime with exactly one visible CUDA device; restore setup afterward.')
        print('Pinned model imports passed. GPU:', value.get('activation_runtime', {}).get('device', 'none'))
        return value

    def readiness(self, environment):
        """Explain selected saved-runtime compatibility before stage execution."""
        actual = environment.get('activation_runtime')
        state = self.restore()
        rows = []
        for kind, paths in [('calibration', state['WRITE_CALIBRATIONS']),
                            ('execution', state['WRITE_EXECUTIONS'])]:
            for product, path in paths.items():
                saved = _read(path).get('runtime', {}) if path.is_file() else None
                differences = {k: {'saved': v, 'current': actual.get(k)}
                    for k,v in saved.items() if k!='git_commit' and v!=actual.get(k)} if saved and actual else {}
                row = dict(product=product, kind=kind, path=str(path),
                           state='pending' if saved is None else ('mismatch' if differences else 'saved'),
                           differences=differences)
                rows.append(row)
                print(f'{product} {kind}: {row["state"]}'
                      + (' | ' + json.dumps(differences) if differences else ''))
        if any(r['differences'] for r in rows):
            print('No GPU work started. Restore matching hardware or explicitly select a new WRITE_VARIANT; existing evidence stays intact.')
        return rows

    def install_environment(self):
        from pilot_eval.sft_backend import PINS
        self.run([sys.executable, '-m', 'pip', 'install', *[f'{k}=={v}' for k,v in PINS.items()]], check=True)
        # Optional vision/audio wheels supplied by Colab can conflict with pinned Torch.
        self.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio'], check=True)
        return self.check_environment()

    def _guard(self, args, command):
        if command not in GPU_COMMANDS:
            return
        active = self.process_probe()
        if active:
            raise RuntimeError('Active experiment child; finish or stop it before another GPU stage:\n'
                               + json.dumps(active, indent=2))
        value = self.check_environment(require_gpu=True)
        config_arg = _argument(args, '--config')
        expected = None
        if config_arg and not Path(config_arg).is_file():
            raise RuntimeError(f'Missing saved config: {config_arg}. Run its preparation/freeze stage first.')
        if config_arg and Path(config_arg).is_file():
            config = _read(config_arg)
            expected = config.get('profile_runtime') or config.get('runtime')
            if command == 'activation-profile' and _argument(args, '--calibration'):
                expected = _read(_argument(args, '--calibration'))['runtime']
        actual = value.get('activation_runtime', {})
        if expected:
            # Activation settings are exact except commit, with an explicit code-only recovery.
            activation = command.startswith('activation-') or command.startswith('tokens-')
            keys = (set(expected) - {'git_commit'}) if activation else {'device', 'cuda', 'python', 'compute_capability', 'versions'}
            if command in {'activation-calibrate', 'activation-calibration-validate', 'activation-profile'}:
                keys = set(expected)  # Incomplete calibration/profile binds the original implementation.
            differences = {}
            for key in keys:
                if key == 'versions':
                    observed = {k: value['versions'].get(k, actual.get('versions', {}).get(k)) for k in expected[key]}
                else:
                    observed = actual.get(key)
                if expected.get(key) != observed:
                    differences[key] = {'saved': expected.get(key), 'current': observed}
            if differences:
                raise RuntimeError('Runtime mismatch BEFORE GPU work; saved evidence is preserved. '
                    'Select a matching runtime or explicitly named new calibration/execution.\n' + json.dumps(differences, indent=2))
        if command == 'activation-measure' and expected and expected.get('git_commit') != actual.get('git_commit'):
            config = _read(config_arg)
            record = self.artifact(config['run_path']) / 'meta/production-recovery.json'
            if not record.is_file() or _read(record).get('actual_runtime') != actual:
                raise RuntimeError('Code-only update detected BEFORE GPU work. Run session.recover_writes(WRITE_EXECUTIONS, notes=...) once; do not recreate calibration.')

    @staticmethod
    def active_children():
        """Colab/Linux process inventory; do not terminate user processes."""
        active = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = path.read_bytes().decode().strip('\0').split('\0')
                if '-m' in args and args[args.index('-m') + 1] == 'pilot_eval':
                    active.append(dict(pid=int(path.parent.name), command=' '.join(args)))
            except (OSError, UnicodeError, IndexError):
                continue
        return active

    def recover_writes(self, executions, *, notes):
        if not notes.strip():
            raise ValueError('Explicit code-only recovery notes required')
        for path in executions.values():
            self.run([sys.executable, '-u', '-m', 'pilot_eval', 'activation-recover-measurement',
                      '--config', str(path), '--review-notes', notes,
                      '--output-root', str(self.root)], check=True)

    def _completed_command(self, args, command):
        """Route candidates through real CPU verifiers; never trust status alone."""
        verifiers = {'activation-measure': 'activation-verify',
                     'activation-recover-measurement': 'activation-verify',
                     'tokens-measure': 'tokens-verify',
                     'grpo-train': 'grpo-verify-training'}
        path = _argument(args, '--config')
        if command == 'activation-calibration-prepare' and _argument(args, '--name'):
            existing = self.plan(_argument(args, '--name'), 'activation.calibration.json')
            if existing.is_file():
                if self.artifact(_read(existing)['prepared_path']) != self.artifact(path):
                    raise RuntimeError('Existing calibration name belongs to different prepared inputs.')
                return [sys.executable, '-u', '-c',
                        'from pilot_eval.notebook_session import NotebookSession; import sys; '
                        'NotebookSession.verify_saved_calibration(sys.argv[1], sys.argv[2], "plan")',
                        str(existing), str(self.root)]
        if command in {'activation-calibrate', 'activation-calibration-validate'} and path and Path(path).is_file():
            phase = 'calibration' if command == 'activation-calibrate' else 'validated'
            directory = self.artifact(_read(path)['run_path'])
            if (directory / phase / 'complete.json').is_file():
                return [sys.executable, '-u', '-c',
                        'from pilot_eval.notebook_session import NotebookSession; import sys; '
                        'NotebookSession.verify_saved_calibration(sys.argv[1], sys.argv[2], sys.argv[3])',
                        str(path), str(self.root), phase]
        if command == 'activation-profile' and path and Path(path).is_file() and _argument(args, '--calibration'):
            calibration = str(self.artifact(_argument(args, '--calibration')).relative_to(self.root))
            matches = [p.parent for p in self.artifact(_read(path)['run_path']).glob('profile-v2-*/results.json')
                       if _read(p)['identity'].get('calibration_path') == calibration]
            if len(matches) > 1:
                raise RuntimeError('Multiple profiles; choose evidence explicitly before running.')
            if matches and (matches[0] / 'complete.json').is_file():
                return [sys.executable, '-u', '-c',
                        'from pilot_eval.notebook_session import NotebookSession; import sys; '
                        'NotebookSession.verify_saved_profile(sys.argv[1], sys.argv[2], sys.argv[3])',
                        str(matches[0]), str(self.root), str(path)]
        if command == 'grpo-train':
            name = _argument(args, '--name')
            path = self.plan(name, 'grpo.training.json') if name else None
        if command in verifiers and path and Path(path).is_file():
            config = _read(path)
            if config.get('run_path') and (self.artifact(config['run_path']) / 'complete.json').is_file():
                print(f'{command}: saved completion candidate; verifying on CPU before reuse.', flush=True)
                return [sys.executable, '-u', '-m', 'pilot_eval', verifiers[command],
                        '--config', str(path), '--output-root', str(self.root)]
        if command == 'grpo-evaluate' and _argument(args, '--name'):
            path = self.plan(_argument(args, '--name'), 'grpo.evaluation.json')
            if path.is_file() and (self.artifact(_read(path)['run_path']) / 'complete.json').is_file():
                return [sys.executable, '-u', '-c',
                        'from pilot_eval.grpo_evaluation import verify_evaluation; '
                        'import sys; verify_evaluation(sys.argv[1], sys.argv[2]); '
                        'print("Saved evaluation verified; no inference repeated.")', str(path), str(self.root)]
        return args

    @staticmethod
    def verify_saved_calibration(path, root, phase):
        from pilot_eval.activation_calibration import _plan, _verify_phase, load_validated_rule
        root = Path(root).resolve()
        plan, prepared, _ = _plan(Path(path), root)
        if phase == 'validated':
            load_validated_rule(Path(path), root, prepared=prepared, runtime=plan['runtime'])
        elif phase == 'calibration':
            _verify_phase(plan, root, phase)
        elif phase != 'plan':
            raise ValueError('Unknown saved calibration phase')
        print(f'Saved {phase} verified on CPU; no model passes repeated.')

    @staticmethod
    def verify_saved_profile(directory, root, prepared_path):
        from pilot_eval.activation_prepare import load_prepared
        from pilot_eval.activation_profile import _verified, _profile_agreement, BATCHES
        from pilot_eval.activation_workflow import _verify_complete
        from pilot_eval.workflow import _hash
        root, directory = Path(root).resolve(), Path(directory)
        prepared, _ = load_prepared(Path(prepared_path), root)
        identity = _read(directory / 'config.json')
        if identity['prepared_sha256'] != _hash(prepared):
            raise ValueError('Saved profile belongs to different prepared inputs')
        _verified(directory, identity, ['config.json', 'results.json'] + [
            f'batch-{batch}/{name}' for batch in BATCHES for name in ['complete.json', 'results.json']])
        _profile_agreement(identity, root, prepared)
        diagnostic = (root / identity['validation_path'] if 'validation_path' in identity else
                      root / prepared['run_path'] / 'validation' /
                      ('diagnostic-' + _hash(identity['validation_identity']['example_ids'])[:12]))
        if not _verify_complete(diagnostic, identity['validation_identity'])['all_gates_passed']:
            raise ValueError('Saved profile diagnostic did not pass')
        print('Saved profile verified on CPU; no benchmark passes repeated.')

    def run(self, args, *, cwd=None, check=True, **kwargs):
        """Subprocess-compatible call; visible output, heartbeat, durable logs, actual exit."""
        args = list(map(str, args))
        command = args[args.index('pilot_eval') + 1] if 'pilot_eval' in args else 'environment'
        args = self._completed_command(args, command)
        command = args[args.index('pilot_eval') + 1] if 'pilot_eval' in args else 'environment'
        self._guard(args, command)
        folder = self.root / 'notebook' / 'operations' / self.plan_name
        folder.mkdir(parents=True, exist_ok=True)
        console = folder / (command + '.console.log')
        print(f'Starting {command} | console: {console}', flush=True)
        q = queue.Queue()
        started = time.monotonic()
        env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        env.update(kwargs.pop('env', {}))
        with console.open('a') as log:
            process = subprocess.Popen(args, cwd=cwd or self.repo, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            print('Child PID:', process.pid, flush=True)
            def read_output():
                for line in process.stdout:
                    q.put(line)
                q.put(None)
            threading.Thread(target=read_output, daemon=True).start()
            tail = []
            try:
                while True:
                    try:
                        line = q.get(timeout=self.heartbeat_seconds)
                    except queue.Empty:
                        message = f'{command}: child active; elapsed {(time.monotonic()-started)/60:.1f} min (not a completion counter)\n'
                        print(message, end='', flush=True); log.write(message); log.flush()
                        continue
                    if line is None:
                        break
                    print(line, end='', flush=True); log.write(line); log.flush()
                    tail = (tail + [line])[-20:]
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
        result = subprocess.CompletedProcess(args, returncode)
        if check and returncode:
            raise RuntimeError(f'{command} exited with code {returncode}. Console: {console}\n' + ''.join(tail))
        print(f'{command}: child exited with code {returncode}', flush=True)
        return result
