"""Read-only audit of selected prepared inputs and rank-one adapter payloads.

Only used adapter/config seals are verified, not every historical optimizer file.
No model construction, CUDA, network access or optional tensor package is needed.
"""
import json
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import PROJECTIONS
from pilot_eval.trace_readout_assets import read_weight, weight_header
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Stage 0 input is outside the selected artifact root.')
    return path


def bind(root, path, parents):
    path = inside(root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = str(path.relative_to(root))
    digest = file_hash(path)
    if relative in parents and parents[relative] != digest:
        raise ValueError('Source changed during the audit.')
    parents[relative] = digest
    return json.loads(path.read_text())


def factors(directory):
    adapter = json.loads((directory / 'adapter_config.json').read_text())
    if (adapter.get('r') != 1 or adapter.get('lora_alpha') != 1
            or adapter.get('lora_dropout') != 0 or adapter.get('rank_pattern')
            or adapter.get('alpha_pattern') or adapter.get('use_dora')
            or adapter.get('use_rslora') or adapter.get('bias', 'none') != 'none'):
        raise ValueError('Expected plain rank-one alpha-one dropout-zero adapter.')
    full_targets = {f"model.layers.{i}.{'self_attn' if p in PROJECTIONS[:4] else 'mlp'}.{p}"
                    for i in range(28) for p in PROJECTIONS}
    targets = adapter.get('target_modules')
    if targets != 'all-linear' and set(targets or []) not in (set(PROJECTIONS), full_targets):
        raise ValueError('Adapter targets differ from the 196-module contract.')
    path = directory / 'adapter_model.safetensors'
    _, header = weight_header(path)
    keys, arrays = set(), {}
    spans = []
    for layer in range(28):
        for projection in PROJECTIONS:
            module = f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            ka, kb = module + '.lora_A.weight', module + '.lora_B.weight'
            for key in [ka, kb]:
                if key not in header or header[key].get('dtype') != 'F32':
                    raise ValueError('Missing/non-FP32 rank-one factor mapping.')
                keys.add(key)
                spans.append(header[key]['data_offsets'])
            a, b = read_weight(path, ka), read_weight(path, kb)
            if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != 1 or b.shape[1] != 1
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError('Invalid rank-one factors or nonfinite payload.')
            arrays[module] = (a, b)
    if set(header) - {'__metadata__'} != keys:
        raise ValueError('Unexpected weights outside the 196-module adapter.')
    ordered = sorted(spans)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Overlapping adapter tensor payloads.')
    return arrays


def checkpoint(root, evidence, step, parents):
    if evidence is None:
        return dict(state='not-selected')
    directory = inside(root, evidence['path'])
    needed = [directory / name for name in ['complete.json', 'adapter_config.json', 'adapter_model.safetensors']]
    if not all(p.is_file() for p in needed):
        return dict(state='missing', path=str(directory.relative_to(root)),
                    missing=[p.name for p in needed if not p.is_file()])
    seal = bind(root, needed[0], parents)
    if file_hash(needed[0]) != evidence['complete_sha256']:
        raise ValueError('Checkpoint seal hash mismatch.')
    if 'step' in seal and seal['step'] != int(step):
        raise ValueError('Checkpoint step mismatch.')
    for path in needed[1:]:
        digest = file_hash(path)
        if seal.get('files', {}).get(path.name) != digest:
            raise ValueError('Checkpoint payload hash mismatch.')
        parents[str(path.relative_to(root))] = digest
    if file_hash(needed[2]) != evidence['adapter_sha256']:
        raise ValueError('Prepared adapter hash mismatch.')
    arrays = factors(directory)
    zero = all(not np.any(b) for a, b in arrays.values())
    if step == '0' and not zero:
        raise ValueError('Checkpoint zero has nonzero B factors.')
    if 'construction.json' in seal.get('files', {}):
        construction = directory / 'construction.json'
        bind(root, construction, parents)
        if file_hash(construction) != seal['files']['construction.json']:
            raise ValueError('Random construction hash mismatch.')
    norms = {key: float(np.linalg.norm(a.astype('float64')) * np.linalg.norm(b.astype('float64')))
             for key, (a, b) in arrays.items()}
    return dict(state='verified-adapter-payload', path=str(directory.relative_to(root)),
                modules=len(arrays), b_exact_zero=zero, update_norms=norms,
                verification_scope='used adapter/config/seal; not trainer-state replay')


def audit_source(root, label, path):
    parents = {}
    path = inside(root, path)
    if not path.is_file():
        return dict(state='missing', path=str(path.relative_to(root)), parents={})
    config = bind(root, path, parents)
    expected_protocol = 'pilot3-write-v1' if label == 'sft' else 'pilot4-writes-v1'
    kind = config.get('source_contract', {}).get('kind')
    if (config.get('protocol_version') != expected_protocol
            or (label == 'grpo' and kind != 'grpo')
            or (label == 'random' and kind != 'random-rank1-control')
            or config.get('model') != 'Qwen/Qwen2.5-1.5B-Instruct'
            or config.get('dtype') != 'float32'):
        raise ValueError('Wrong prepared source kind/model/precision for Stage 0.')
    seal = bind(root, path.parent / 'prepare-complete.json', parents)
    if seal.get('config_sha256') != _hash(config):
        raise ValueError('Prepared configuration hash mismatch.')
    relative = str(path.relative_to(root))
    if relative not in seal.get('files', {}):
        raise ValueError('Prepared seal omits its configuration.')
    for value, expected in seal['files'].items():
        payload = inside(root, value)
        if not payload.is_file() or file_hash(payload) != expected:
            raise ValueError('Prepared source payload hash mismatch.')
        parents[str(payload.relative_to(root))] = expected
    evidence = config.get('source_evidence', {})
    identity = {k: config.get(k) for k in ['model', 'model_revision', 'tokenizer_revision', 'dtype']}
    identity['base_sha256'] = evidence.get('base_sha256')
    if not all(identity.values()):
        raise ValueError('Source lacks pinned model/tokenizer/base identity.')
    checkpoints = {step: checkpoint(root, evidence.get('checkpoints', {}).get(step), step, parents)
                   for step in ['0', '64']}
    return dict(state='audited', identity=identity, path=relative, checkpoints=checkpoints,
                available_checkpoint_steps=config.get('checkpoint_steps', []), parents=parents)


def initialization(root, sources):
    paths = []
    for label in ['sft', 'grpo']:
        zero = sources[label].get('checkpoints', {}).get('0', {})
        if zero.get('state') != 'verified-adapter-payload':
            return dict(state='unavailable', equal_a=None, modules_compared=0)
        paths.append(inside(root, zero['path']))
    left, right = map(factors, paths)
    differences = [key for key in left if not np.array_equal(left[key][0], right[key][0])]
    return dict(state='tensor-compared', equal_a=not differences, modules_compared=len(left),
                different_modules=differences,
                comparison='exact FP32 A tensors, not inference from seed')
