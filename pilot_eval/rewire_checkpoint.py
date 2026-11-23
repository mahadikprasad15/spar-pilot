"""Sparse rewiring ticket 08: checkpoints and restart (D5, D24).

What a checkpoint is (D24, decided 2026-11-23). ``checkpoints/checkpoint-<k>/`` holds exactly two files:

* ``rewire_state.safetensors``: every ``rewire_`` parameter of the model, unmerged, in FP32. That is the read
  factors (rank k >= 1: ``rewire_A_{gate,up}`` [k, d] and ``rewire_B_{gate,up}`` [n, k]; rank 0:
  ``rewire_W_{gate,up}`` [n, d]) and ``rewire_log_alpha`` [n], for every layer, under the names
  ``rewire_module.install`` gives them.
* ``complete.json``: the seal, written last. It binds the step, the plan hash, the runtime hash and the code
  commit, and lists the file's SHA-256, the tensor names and shapes, and the payload size.

No optimizer, scheduler or RNG state, at any step. A checkpoint folder with any other file is refused.

Restart, not resume (D24). An arm either finishes in one attempt or starts again from step 0 under the same
plan. Before an arm trains, ``prepare_arm`` looks at its run directory:

* a sealed, verified completed arm (``training-complete.json`` plus every planned checkpoint) is kept and
  never rerun; a completed arm restored from the volume is verified the same way;
* a sealed arm that fails verification is refused, never silently retrained;
* anything else is an interrupted attempt. Its directory is moved aside as evidence
  (``<run>.interrupted-<ns>``, the convention of ``attribution_units.seal_run``), its tensor payloads are
  deleted so the evidence does not eat the D24 disk budget (their hashes stay in the seals), and the restart is
  logged with ``sampling_sft_training.event``.

Interface with the training loop (ticket 07). This module never imports the loop. The loop receives a
``CheckpointSink`` and calls it as ``sink(step, model)``:

* once with ``step = 0`` before the first optimizer step (checkpoint 0 is the untuned model: every read
  coefficient is exactly zero, and ``verify_checkpoint`` checks that);
* after optimizer step ``s`` for ``s = 1..steps``, with the state after that step's update and before anything
  that belongs to step ``s + 1``. In particular checkpoint 960 holds the continuous ``logα`` of step 960, and
  the binary hardening (``freeze_after_step``) shows up from checkpoint 1024 on, so the two bracket it.

The sink writes a checkpoint only at the plan's steps and ignores the rest, so the loop may call it every step.
``run_arm`` is the whole arm-level contract in one call (prepare, fresh model, train, check, seal); ticket 14's
chain stage calls it with the real loop, and the fixture tests call it with a stand-in.

Reuse. The seal mirrors ``sampling_sft_training.seal`` / ``checkpoint`` field for field (``step``,
``recoverable``, ``config_sha256``, ``runtime_sha256``, ``files``) and is written with the same write-once
``sampling_sft.frozen``; hashes are ``training.file_hash`` (streamed); the plan hash is ``sparse_rewiring_plan.plan_sha256``. Those two functions
are not called directly: they hard-code the PEFT adapter file set, and adding a parameter to them would change
``sampling_sft_training.implementation()``, which hashes that file and is bound into every recorded Sampling SFT
plan, so every existing plan would stop loading. Their optimizer and RNG recovery paths are not used (D24).

``sft_evaluation.evaluate_sft`` only loads PEFT checkpoints 0 to 64; ``load_into`` is the small loader for rewire
checkpoints that evaluation (ticket 09) uses instead.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval.training import file_hash

CHECKPOINT_SCHEMA = 'sparse-rewiring/checkpoint-v1'
ARM_SCHEMA = 'sparse-rewiring/training-complete-v1'
STATE_FILE = 'rewire_state.safetensors'
SEAL_FILE = 'complete.json'
ARM_SEAL = 'training-complete.json'
CONTENTS = ['read_factors', 'log_alpha', 'seal']      # plan settings.storage.checkpoint_contents (D24)
PARTS = ('gate', 'up')
FP32_BYTES = 4


class CheckpointRefused(ValueError):
    """A checkpoint or completed arm does not match its plan, runtime, code commit or content."""


# --- what a checkpoint must contain ---------------------------------------------------------------

def expected_tensors(plan):
    """``{name: shape}`` of every tensor a checkpoint of this plan holds, from the plan's model shape and rank.

    Names are exactly the ``rewire_`` parameter names ``rewire_module.install`` gives a ``Qwen2ForCausalLM``."""
    model, rank = plan['settings']['model'], plan['rank']
    layers, d, n = model['layers'], model['hidden'], model['intermediate']
    out = {}
    for layer in range(layers):
        prefix = f'model.layers.{layer}.mlp.rewire_'
        for part in PARTS:
            if rank == 0:
                out[f'{prefix}W_{part}'] = [n, d]
            else:
                out[f'{prefix}A_{part}'] = [rank, d]
                out[f'{prefix}B_{part}'] = [n, rank]
        out[f'{prefix}log_alpha'] = [n]
    return dict(sorted(out.items()))


def payload_bytes(plan):
    """FP32 bytes of one checkpoint's tensors, recomputed from the shapes (independent of the plan's own
    ``checkpoint_bytes``, which ``storage`` compares it with)."""
    total = 0
    for shape in expected_tensors(plan).values():
        size = 1
        for dim in shape:
            size *= dim
        total += size * FP32_BYTES
    return total


def storage(plan):
    """D24 storage of one arm: per-checkpoint payload, the number of checkpoints and the total. Refuses a plan
    whose ``checkpoint_bytes`` disagrees with the shapes or whose storage settings are not D24's."""
    per = payload_bytes(plan)
    if plan['checkpoint_bytes'] != per:
        raise CheckpointRefused(f"plan checkpoint_bytes {plan['checkpoint_bytes']} != {per} from the shapes")
    settings = plan['settings']['storage']
    if settings['checkpoint_contents'] != CONTENTS or settings['optimizer_state'] != 'none' or \
            settings['interruption'] != 'restart-from-step-0':
        raise CheckpointRefused(f'plan storage settings are not D24: {settings}')
    return dict(checkpoint_bytes=per, checkpoints=len(plan['checkpoints']), total_bytes=per * len(plan['checkpoints']))


def _check_plan(plan):
    from pilot_eval.sparse_rewiring_plan import plan_sha256
    if plan.get('plan_sha256') != plan_sha256(plan):
        raise CheckpointRefused('plan does not match its own hash')
    points = plan['checkpoints']
    if points != sorted(set(points)) or points[0] != 0:
        raise CheckpointRefused('plan checkpoints must be increasing and start at 0')
    storage(plan)
    return plan['plan_sha256']


def checkpoint_dir(run_dir, step):
    return Path(run_dir) / 'checkpoints' / f'checkpoint-{step}'


# --- writing --------------------------------------------------------------------------------------

def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def state_of(model, plan):
    """The rewire parameters of ``model`` as CPU FP32 tensors, refused unless the names and shapes are exactly
    the plan's (a model installed at another rank or shape cannot be checkpointed under this plan)."""
    import torch
    from pilot_eval import rewire_module as rw
    state = {name: p.detach().to('cpu').contiguous() for name, p in rw.trainable_parameters(model)}
    expected = expected_tensors(plan)
    actual = {name: list(t.shape) for name, t in state.items()}
    if actual != expected:
        missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        raise CheckpointRefused(f'model rewire parameters differ from the plan: missing {missing[:4]}, '
                                f'extra {extra[:4]}, or shapes differ')
    if any(t.dtype != torch.float32 for t in state.values()):
        raise CheckpointRefused('rewire parameters must be FP32 (D8)')
    return state


def save_checkpoint(run_dir, step, model, plan, *, runtime_sha256, code_commit):
    """Write and seal checkpoint ``step``. The folder is built under a temporary name and renamed into place
    after the seal, so ``checkpoint-<k>`` exists only complete. An existing checkpoint is never overwritten."""
    from safetensors.torch import save_file
    plan_sha = _check_plan(plan)
    if step not in plan['checkpoints']:
        raise CheckpointRefused(f'step {step} is not a planned checkpoint')
    final = checkpoint_dir(run_dir, step)
    if final.exists():
        raise CheckpointRefused(f'{final} already exists; an attempt writes each checkpoint once')
    state = state_of(model, plan)
    if step == 0:
        _require_zero_change(state)
    tmp = final.with_name(final.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    # One metadata key: safetensors stores metadata in a hash map, so several keys are written in varying
    # order and identical tensors would hash differently between runs.
    save_file(state, str(tmp / STATE_FILE), metadata={'checkpoint': json.dumps(
        dict(schema=CHECKPOINT_SCHEMA, step=step, plan_sha256=plan_sha), sort_keys=True)})
    with (tmp / STATE_FILE).open('rb') as stream:
        os.fsync(stream.fileno())
    seal = dict(schema=CHECKPOINT_SCHEMA, step=step, recoverable=False, config_sha256=plan_sha,
                runtime_sha256=runtime_sha256, code_commit=code_commit, run_id=plan['run_id'], rank=plan['rank'],
                contents=list(CONTENTS), files={STATE_FILE: file_hash(tmp / STATE_FILE)},
                tensors=expected_tensors(plan), payload_bytes=payload_bytes(plan))
    source.frozen(tmp / SEAL_FILE, seal)
    with (tmp / SEAL_FILE).open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, final)
    _fsync_dir(final.parent)
    return seal


def _require_zero_change(state):
    import torch
    for name, tensor in state.items():
        if ('rewire_B_' in name or 'rewire_W_' in name) and torch.count_nonzero(tensor).item():
            raise CheckpointRefused(f'checkpoint 0 must be the untuned model, but {name} is not zero (D10)')


class CheckpointSink:
    """What the training loop calls: ``sink(step, model)`` at step 0 and after every optimizer step.

    Writes the planned checkpoints, ignores other steps, refuses a step out of order, and calls
    ``on_checkpoint(step, folder)`` after each one (ticket 14 mirrors the checkpoint to the volume there)."""

    def __init__(self, run_dir, plan, *, runtime_sha256, code_commit, on_checkpoint=None):
        self.run_dir, self.plan = Path(run_dir), plan
        self.runtime_sha256, self.code_commit = runtime_sha256, code_commit
        self.on_checkpoint = on_checkpoint
        self.planned = set(plan['checkpoints'])
        self.last = None
        self.written = []
        _check_plan(plan)

    def __call__(self, step, model):
        if not isinstance(step, int) or isinstance(step, bool):
            raise CheckpointRefused(f'step must be an int, got {step!r}')
        expected = 0 if self.last is None else self.last + 1
        if step != expected:
            raise CheckpointRefused(f'training loop reported step {step}, expected {expected}')
        self.last = step
        if step in self.planned:
            save_checkpoint(self.run_dir, step, model, self.plan, runtime_sha256=self.runtime_sha256,
                            code_commit=self.code_commit)
            self.written.append(step)
            if self.on_checkpoint is not None:
                self.on_checkpoint(step, checkpoint_dir(self.run_dir, step))

    def missing(self):
        return sorted(self.planned - set(self.written))


# --- verifying and loading ------------------------------------------------------------------------

def verify_checkpoint(folder, plan, *, runtime_sha256=None, code_commit=None):
    """The seal of one checkpoint, refused unless it matches the plan (and the runtime and code commit when
    given), holds exactly D24's contents, and its payload matches its hash. Mirrors
    ``sampling_sft_training.checkpoint``; reads only the safetensors header except at step 0, where it also
    checks that every read coefficient is zero."""
    from safetensors import safe_open
    folder = Path(folder)
    plan_sha = _check_plan(plan)
    seal_path = folder / SEAL_FILE
    if not seal_path.is_file():
        raise CheckpointRefused(f'{folder} is not sealed')
    seal = json.loads(seal_path.read_text())
    present = sorted(p.name for p in folder.iterdir())
    if present != sorted([STATE_FILE, SEAL_FILE]):
        raise CheckpointRefused(f'{folder} must hold only {STATE_FILE} and {SEAL_FILE} (D24), found {present}')
    step = int(folder.name.removeprefix('checkpoint-'))
    if (seal.get('schema') != CHECKPOINT_SCHEMA or seal.get('step') != step or seal.get('recoverable') is not False
            or seal.get('config_sha256') != plan_sha or seal.get('run_id') != plan['run_id']
            or seal.get('rank') != plan['rank'] or seal.get('contents') != CONTENTS
            or set(seal.get('files', {})) != {STATE_FILE} or step not in plan['checkpoints']):
        raise CheckpointRefused(f'{folder}: checkpoint plan/step/file identity changed')
    if runtime_sha256 is not None and seal['runtime_sha256'] != runtime_sha256:
        raise CheckpointRefused(f'{folder}: checkpoint runtime differs')
    if code_commit is not None and seal['code_commit'] != code_commit:
        raise CheckpointRefused(f'{folder}: checkpoint code commit differs')
    if seal['tensors'] != expected_tensors(plan) or seal['payload_bytes'] != payload_bytes(plan):
        raise CheckpointRefused(f'{folder}: checkpoint tensor list differs from the plan')
    if file_hash(folder / STATE_FILE) != seal['files'][STATE_FILE]:
        raise CheckpointRefused(f'{folder}: checkpoint content changed')
    with safe_open(str(folder / STATE_FILE), framework='pt') as handle:
        names = sorted(handle.keys())
        if names != list(seal['tensors']):
            raise CheckpointRefused(f'{folder}: payload tensors differ from the seal')
        for name in names:
            sliced = handle.get_slice(name)
            if list(sliced.get_shape()) != seal['tensors'][name] or sliced.get_dtype() != 'F32':
                raise CheckpointRefused(f'{folder}: {name} has the wrong shape or dtype')
        if step == 0:
            _require_zero_change({n: handle.get_tensor(n) for n in names
                                  if 'rewire_B_' in n or 'rewire_W_' in n})
    return seal


def load_tensors(folder, plan, **identity):
    """Verified checkpoint tensors as a CPU ``{name: tensor}`` dict (for CPU analyses)."""
    from safetensors.torch import load_file
    seal = verify_checkpoint(folder, plan, **identity)
    return seal, load_file(str(Path(folder) / STATE_FILE))


def load_into(model, folder, plan, **identity):
    """Load a verified rewire checkpoint into ``model`` (already rewired with ``rewire_module.install`` at the
    plan's rank). Every rewire parameter is overwritten in place on its own device; base weights are untouched.
    This is the loader evaluation uses: ``sft_evaluation.evaluate_sft`` only loads PEFT checkpoints."""
    import torch
    from pilot_eval import rewire_module as rw
    seal, tensors = load_tensors(folder, plan, **identity)
    params = dict(rw.trainable_parameters(model))
    if sorted(params) != sorted(tensors):
        raise CheckpointRefused('model rewire parameters do not match the checkpoint')
    with torch.no_grad():
        for name, param in params.items():
            if tuple(param.shape) != tuple(tensors[name].shape):
                raise CheckpointRefused(f'{name}: model shape {tuple(param.shape)} differs from the checkpoint')
            param.copy_(tensors[name].to(device=param.device, dtype=param.dtype))
    return seal


# --- arm level: seal, verify, keep or restart -----------------------------------------------------

def seal_arm(run_dir, plan, *, runtime_sha256, code_commit, files=()):
    """Seal a finished arm: every planned checkpoint verified under one runtime and commit, plus the hashes of
    the loop's own outputs named in ``files`` (relative to ``run_dir``, e.g. ``logs/steps.jsonl``)."""
    run_dir = Path(run_dir)
    plan_sha = _check_plan(plan)
    checkpoints = {}
    for step in plan['checkpoints']:
        folder = checkpoint_dir(run_dir, step)
        verify_checkpoint(folder, plan, runtime_sha256=runtime_sha256, code_commit=code_commit)
        checkpoints[str(step)] = file_hash(folder / SEAL_FILE)
    hashed = {}
    for name in sorted(files):
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
            raise CheckpointRefused(f'arm output {name} is missing or outside the run directory')
        hashed[name] = file_hash(path)
    value = dict(schema=ARM_SCHEMA, run_id=plan['run_id'], arm=plan['arm'], rank=plan['rank'],
                 config_sha256=plan_sha, runtime_sha256=runtime_sha256, code_commit=code_commit,
                 checkpoints=checkpoints, files=hashed)
    source.frozen(run_dir / ARM_SEAL, value)
    return value


def verify_arm(run_dir, plan, *, runtime_sha256=None, code_commit=None):
    """A completed arm, refused unless its seal matches the plan and every one of the plan's expected
    checkpoints, and every sealed output, is present and unchanged. Works the same on a directory restored
    from the volume."""
    run_dir = Path(run_dir)
    plan_sha = _check_plan(plan)
    path = run_dir / ARM_SEAL
    if not path.is_file():
        raise CheckpointRefused(f'{run_dir} has no completed-arm seal')
    value = json.loads(path.read_text())
    if (value.get('schema') != ARM_SCHEMA or value.get('config_sha256') != plan_sha
            or value.get('run_id') != plan['run_id'] or value.get('rank') != plan['rank']):
        raise CheckpointRefused(f'{run_dir}: completed arm was trained under another plan')
    if runtime_sha256 is not None and value['runtime_sha256'] != runtime_sha256:
        raise CheckpointRefused(f'{run_dir}: completed arm runtime differs')
    if code_commit is not None and value['code_commit'] != code_commit:
        raise CheckpointRefused(f'{run_dir}: completed arm code commit differs')
    if sorted(map(int, value['checkpoints'])) != list(plan['checkpoints']):
        raise CheckpointRefused(f"{run_dir}: sealed checkpoints differ from the plan's expected checkpoints")
    for step in plan['checkpoints']:
        folder = checkpoint_dir(run_dir, step)
        verify_checkpoint(folder, plan, runtime_sha256=value['runtime_sha256'], code_commit=value['code_commit'])
        if file_hash(folder / SEAL_FILE) != value['checkpoints'][str(step)]:
            raise CheckpointRefused(f'{folder}: seal differs from the completed arm')
    for name, digest in value['files'].items():
        file = (run_dir / name).resolve()
        if not file.is_relative_to(run_dir.resolve()) or not file.is_file() or file_hash(file) != digest:
            raise CheckpointRefused(f'{run_dir}: sealed output {name} changed')
    return value


def discard_attempt(run_dir, *, reason):
    """Move an interrupted attempt aside as evidence and drop its tensor payloads (D24 disk budget); the seals,
    logs and a list of what was removed stay. Returns the evidence directory."""
    run_dir = Path(run_dir)
    evidence = run_dir.with_name(f'{run_dir.name}.interrupted-{time.time_ns()}')
    os.replace(run_dir, evidence)
    removed = []
    for payload in sorted(evidence.rglob('*.safetensors')):
        removed.append(dict(path=str(payload.relative_to(evidence)), bytes=payload.stat().st_size))
        payload.unlink()
    (evidence / 'discarded.json').write_text(json.dumps(dict(reason=reason, removed_payloads=removed),
                                                        indent=1, sort_keys=True) + '\n')
    return evidence


def prepare_arm(run_dir, plan):
    """Decide what an arm's next attempt does. Returns ``('keep', seal)`` for a verified completed arm,
    ``('fresh', None)`` for an empty or absent directory, and ``('restart', evidence_dir)`` after moving an
    interrupted attempt aside. A completed arm that fails verification raises; it is never retrained silently."""
    from pilot_eval.sampling_sft_training import event
    run_dir = Path(run_dir)
    _check_plan(plan)
    if (run_dir / ARM_SEAL).exists():
        return 'keep', verify_arm(run_dir, plan)
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return 'fresh', None
    found = sorted(int(p.name.removeprefix('checkpoint-')) for p in (run_dir / 'checkpoints').glob('checkpoint-*')
                   if p.is_dir() and p.name.removeprefix('checkpoint-').isdigit()) \
        if (run_dir / 'checkpoints').is_dir() else []
    evidence = discard_attempt(run_dir, reason='interrupted before the arm was sealed (D24: restart from step 0)')
    event(run_dir, 'restart-from-step-0', plan_sha256=plan['plan_sha256'], discarded=evidence.name,
          discarded_checkpoints=found)
    return 'restart', evidence


def run_arm(run_dir, plan, *, model_factory, train, runtime_sha256, code_commit, files=(), on_checkpoint=None):
    """One arm, end to end: keep a verified completed arm; otherwise build a fresh step-0 model with
    ``model_factory()``, call ``train(model, plan, sink)`` (ticket 07's loop, which calls ``sink(step, model)``),
    refuse if any planned checkpoint was not written, and seal the arm with the loop's ``files``.

    An exception from ``train`` propagates and leaves the attempt unsealed, so the next ``run_arm`` restarts
    it from step 0."""
    action, detail = prepare_arm(run_dir, plan)
    if action == 'keep':
        return dict(action='kept', seal=detail)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    model = model_factory()
    sink = CheckpointSink(run_dir, plan, runtime_sha256=runtime_sha256, code_commit=code_commit,
                          on_checkpoint=on_checkpoint)
    train(model, plan, sink)
    if sink.missing():
        raise CheckpointRefused(f'training ended without planned checkpoints {sink.missing()}')
    seal = seal_arm(run_dir, plan, runtime_sha256=runtime_sha256, code_commit=code_commit, files=files)
    return dict(action='trained' if action == 'fresh' else 'restarted', seal=seal,
                evidence=None if detail is None else str(detail))

