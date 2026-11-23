"""Ticket 01 of the joint neuron/adapter audit: a read-only, sealed source inventory.

Binds every intended arm's checkpoints, configs, PEFT key lists and lineage files to
hashes, and states explicitly what is absent, unsupported or unresolved. It never loads
a model, imports Torch, touches the network or edits a source.

Dimensions, layers, rank and target projections are read from the saved payloads; the
28-layer / rank-one / seven-projection Stage 0 contract is reported, not assumed.

Two hash algorithms are kept apart:
- ``sha256`` on a file: SHA-256 of the file's bytes.
- ``parameter_sha256``: ``PARAMETER_HASH_ALGORITHM`` over sorted tensor keys, dtype,
  shape and raw little-endian tensor bytes, independent of header layout/ordering.

Usage:
    python -m pilot_eval.neuron_audit_inventory build --sources <sources.json> \
        --artifact_root <root> --output_root <out> --run_id <id>
    python -m pilot_eval.neuron_audit_inventory verify --run_dir <dir> --artifact_root <root>
"""
import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

from pilot_eval.trace_readout_assets import read_weight, weight_header
from pilot_eval.training import file_hash

SOURCES_SCHEMA = 'joint-neuron-adapter-audit/sources-v1'
INVENTORY_SCHEMA = 'joint-neuron-adapter-audit/inventory-v1'
PARAMETER_HASH_ALGORITHM = 'peft-safetensors-sorted-key-dtype-shape-bytes-sha256-v1'
SEMANTIC_CONFIG_ALGORITHM = 'adapter-config-json-sorted-keys-sorted-target-modules-sha256-v1'
RUN_SUBDIR = 'runs/joint-neuron-adapter-audit/inventory'
PROJECTION_ORDER = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
USED_FILES = ['adapter_model.safetensors', 'adapter_config.json']
KEY = re.compile(r'^base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_(A|B)\.weight$')
DTYPE_BYTES = {'F32': 4, 'F16': 2, 'BF16': 2, 'F64': 8}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _value_hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Source path is outside the artifact root: {value}')
    return path


def _file_record(path):
    return dict(sha256=file_hash(path), size=path.stat().st_size)


def _projection_sort(name):
    return PROJECTION_ORDER.index(name) if name in PROJECTION_ORDER else len(PROJECTION_ORDER)


# --- adapter config -------------------------------------------------------------------------

def _config(path):
    cfg = json.loads(path.read_text())
    semantic = dict(cfg)
    if isinstance(semantic.get('target_modules'), list):
        semantic['target_modules'] = sorted(semantic['target_modules'])
    unsupported = []
    rank, alpha = cfg.get('r'), cfg.get('lora_alpha')
    if type(rank) is not int or rank < 1:
        unsupported.append('r')
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        unsupported.append('lora_alpha')
    if cfg.get('lora_dropout', 0) != 0:
        unsupported.append('lora_dropout')
    if cfg.get('bias', 'none') != 'none':
        unsupported.append('bias')
    for flag in ['use_rslora', 'use_dora', 'use_qalora', 'lora_bias']:
        if cfg.get(flag):
            unsupported.append(flag)
    for pattern in ['rank_pattern', 'alpha_pattern']:
        if cfg.get(pattern):
            unsupported.append(pattern)
    for field in ['layers_to_transform', 'layer_replication', 'modules_to_save', 'target_parameters',
                  'trainable_token_indices', 'exclude_modules']:
        if cfg.get(field):
            unsupported.append(field)
    if cfg.get('peft_type', 'LORA') != 'LORA':
        unsupported.append('peft_type')
    if cfg.get('fan_in_fan_out'):
        unsupported.append('fan_in_fan_out')
    return dict(
        semantic_sha256=_value_hash(semantic), semantic_algorithm=SEMANTIC_CONFIG_ALGORITHM,
        r=rank, lora_alpha=alpha, lora_dropout=cfg.get('lora_dropout'),
        target_modules=semantic.get('target_modules'), peft_version=cfg.get('peft_version'),
        base_model_name_or_path=cfg.get('base_model_name_or_path'), unsupported=unsupported,
        scale=None if unsupported else float(alpha) / rank)


# --- adapter weights ------------------------------------------------------------------------

def _raw_bytes(path, length, row):
    start, end = row['data_offsets']
    with path.open('rb') as stream:
        stream.seek(8 + length + start)
        data = stream.read(end - start)
    if len(data) != end - start:
        raise ValueError('Truncated safetensors payload.')
    return data


def _weights(path):
    length, header = weight_header(path)
    keys = sorted(k for k in header if k != '__metadata__')
    digest = hashlib.sha256()
    modules, unexpected, problems = {}, [], []
    spans = []
    for key in keys:
        row = header[key]
        dtype, shape = row.get('dtype'), row.get('shape')
        raw = _raw_bytes(path, length, row)
        if dtype not in DTYPE_BYTES or len(raw) != DTYPE_BYTES[dtype] * int(np.prod(shape or [0])):
            problems.append(f'unsupported dtype or size for {key}')
        spans.append(tuple(row['data_offsets']))
        for part in (key.encode(), b'\0', str(dtype).encode(), b'\0', _canonical(shape).encode(), b'\0', raw):
            digest.update(part)
        match = KEY.match(key)
        if not match:
            unexpected.append(key)
            continue
        layer, group, projection, side = match.groups()
        module = key[:-len(f'.lora_{side}.weight')]
        entry = modules.setdefault(module, dict(module=module, layer=int(layer), group=group,
                                                projection=projection))
        entry[side] = dict(key=key, dtype=dtype, shape=shape)
        if dtype in ('F32', 'F16', 'BF16'):
            values = read_weight(path, key)
            if not np.isfinite(values).all():
                problems.append(f'nonfinite values in {key}')
            entry[side + '_all_zero'] = bool(not np.any(values))
    ordered = sorted(spans)
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        problems.append('overlapping tensor payloads')
    incomplete = sorted(m for m, e in modules.items() if 'A' not in e or 'B' not in e)
    ranks, dims = set(), {}
    for entry in modules.values():
        if 'A' not in entry or 'B' not in entry:
            continue
        a, b = entry['A']['shape'], entry['B']['shape']
        if len(a) != 2 or len(b) != 2 or a[0] != b[1]:
            problems.append(f"inconsistent factor shapes for {entry['module']}")
            continue
        ranks.add(a[0])
        shape = dict(in_features=a[1], out_features=b[0])
        if dims.setdefault(entry['projection'], shape) != shape:
            problems.append(f"projection {entry['projection']} dimensions differ across layers")
    if len(ranks) > 1:
        problems.append('mixed ranks across modules')
    ordered_modules = sorted(modules.values(), key=lambda e: (e['layer'], _projection_sort(e['projection'])))
    return dict(
        parameter_sha256=digest.hexdigest(), parameter_hash_algorithm=PARAMETER_HASH_ALGORITHM,
        tensor_count=len(keys), module_count=len(modules),
        layers=sorted({e['layer'] for e in modules.values()}),
        projections=sorted({e['projection'] for e in modules.values()}, key=_projection_sort),
        rank=ranks.pop() if len(ranks) == 1 else None,
        dims={p: dims[p] for p in sorted(dims, key=_projection_sort)},
        dtypes=sorted({header[k].get('dtype') for k in keys}),
        zero_b_modules=sum(1 for e in modules.values() if e.get('B_all_zero')),
        zero_a_modules=sum(1 for e in modules.values() if e.get('A_all_zero')),
        unexpected_keys=unexpected, incomplete_modules=incomplete, problems=problems,
    ), dict(keys=keys, modules=[{k: v for k, v in e.items() if not k.endswith('_all_zero')}
                                for e in ordered_modules])


# --- checkpoints ----------------------------------------------------------------------------

def _seal(directory, files):
    path = directory / 'complete.json'
    if not path.is_file():
        return None
    listed = json.loads(path.read_text()).get('files', {})
    mismatched, absent = [], []
    for name, digest in sorted(listed.items()):
        if name in files:
            if files[name]['sha256'] != digest:
                mismatched.append(name)
        else:
            absent.append(name)
    unsealed = sorted(n for n in USED_FILES if n in files and n not in listed)
    return dict(sha256=files['complete.json']['sha256'], mismatched=mismatched,
                absent_locally=absent, unsealed_used_files=unsealed)


def _checkpoint(root, spec, out_dir, arm, step):
    directory = _inside(root, spec['path'])
    record = dict(path=spec['path'], role=spec.get('role', 'audited'), status=None, reasons=[],
                  files={}, seal=None, config=None, weights=None, coverage=None, expected={},
                  key_list_path=None)
    if not (directory / 'adapter_model.safetensors').is_file():
        record['status'] = 'missing'
        record['reasons'].append('adapter_model.safetensors absent')
        for name, digest in (spec.get('expected') or {}).items():
            record['expected'][name] = dict(expected=digest, actual=None, match=False)
        return record, None
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        record['files'][path.name] = _file_record(path)
    files, reasons = record['files'], record['reasons']
    for name in USED_FILES + ['complete.json']:
        if name not in files:
            reasons.append(f'{name} absent')
    record['seal'] = _seal(directory, files)
    if record['seal']:
        for name in record['seal']['mismatched']:
            reasons.append(f'{name} differs from its checkpoint seal')
        for name in record['seal']['unsealed_used_files']:
            reasons.append(f'{name} not listed in its checkpoint seal')
    actual = dict(adapter_sha256=files.get('adapter_model.safetensors', {}).get('sha256'),
                  config_sha256=files.get('adapter_config.json', {}).get('sha256'),
                  seal_sha256=files.get('complete.json', {}).get('sha256'))
    for name, digest in (spec.get('expected') or {}).items():
        if name not in actual:
            raise ValueError(f'Unknown expected hash field: {name}')
        record['expected'][name] = dict(expected=digest, actual=actual[name], match=digest == actual[name])
        if digest != actual[name]:
            reasons.append(f'{name} does not match the lineage-recorded hash')
    weights, keys = _weights(directory / 'adapter_model.safetensors')
    record['weights'] = weights
    reasons.extend(weights['problems'])
    if weights['unexpected_keys']:
        reasons.append('unexpected tensor keys outside the LoRA layer mapping')
    if weights['incomplete_modules']:
        reasons.append('modules missing lora_A or lora_B')
    if weights['module_count'] == 0:
        reasons.append('no LoRA modules found')
    if record['role'] == 'audited' and weights['zero_b_modules']:
        reasons.append(f"{weights['zero_b_modules']} modules have an all-zero lora_B at the audited step; "
                       'unresolved until provenance explains it')
    if 'adapter_config.json' in files:
        config = _config(directory / 'adapter_config.json')
        record['config'] = config
        if config['unsupported']:
            reasons.append('unsupported adapter options: ' + ', '.join(config['unsupported']))
        if weights['rank'] is not None and config['r'] != weights['rank']:
            reasons.append('config rank differs from saved factor rank')
        targets = config['target_modules']
        if targets == 'all-linear' or targets is None:
            not_in_config = []
        else:
            not_in_config = [p for p in weights['projections'] if p not in targets]
        record['coverage'] = dict(
            config_targets=targets, weight_projections=weights['projections'],
            weight_projections_not_in_config=not_in_config,
            config_targets_without_weights=[] if not isinstance(targets, list)
            else sorted(t for t in targets if t not in weights['projections']))
        if not_in_config or record['coverage']['config_targets_without_weights']:
            reasons.append('adapter config targets and saved modules differ')
    rel = Path(RUN_SUBDIR) / out_dir.name / 'keys' / arm / f'step-{step}.json'
    record['key_list_path'] = str(rel)
    record['status'] = 'unresolved' if reasons else 'verified'
    return record, keys


def _arm(root, spec, out_dir, audit_step, init_step):
    arm = spec['arm']
    record = dict(arm=arm, label=spec.get('label', arm), role=spec.get('role', 'trained'),
                  checkpoints={}, lineage=[], consistency={}, reasons=[], restore=spec.get('restore'))
    key_lists = {}
    for step, ck in sorted((spec.get('checkpoints') or {}).items(), key=lambda kv: int(kv[0])):
        record['checkpoints'][step], key_lists[step] = _checkpoint(root, ck, out_dir, arm, step)
    for item in spec.get('lineage') or []:
        path = _inside(root, item['path'])
        present = path.is_file()
        record['lineage'].append(dict(role=item['role'], path=item['path'], present=present,
                                      **(_file_record(path) if present else dict(sha256=None, size=None))))
    audited = record['checkpoints'].get(str(audit_step))
    init = record['checkpoints'].get(str(init_step))
    if record['role'] == 'baseline':
        record['adapted_measurement'] = 'baseline-no-adapter'
    elif audited is None or audited['status'] == 'missing':
        record['adapted_measurement'] = 'missing'
        record['reasons'].append(f'no step-{audit_step} adapter payload')
    else:
        if init and init['weights'] and audited['weights']:
            same_keys = key_lists[str(init_step)]['keys'] == key_lists[str(audit_step)]['keys']
            record['consistency']['same_key_list'] = same_keys
            if not same_keys:
                record['reasons'].append('initialization and audited key lists differ')
            if init['config'] and audited['config']:
                same = init['config']['semantic_sha256'] == audited['config']['semantic_sha256']
                record['consistency']['same_semantic_config'] = same
                if not same:
                    record['reasons'].append('initialization and audited configs differ semantically')
        record['initialization'] = 'missing' if init is None else init['status']
        ok = audited['status'] == 'verified' and not record['reasons']
        record['adapted_measurement'] = 'load-ready' if ok else 'unresolved'
    return record, key_lists


def _base(root, spec):
    record = dict({k: spec.get(k) for k in ['model', 'model_revision', 'tokenizer_revision']},
                  expected_file_sha256=spec.get('expected_file_sha256') or {},
                  expected_base_parameter_sha256=spec.get('base_parameter_sha256'),
                  snapshot_dir=spec.get('snapshot_dir'), local=False, files={}, status='not-local')
    if not spec.get('snapshot_dir'):
        return record
    directory = _inside(root, spec['snapshot_dir'])
    if not directory.is_dir():
        return record
    record['local'] = True
    for name, digest in sorted(record['expected_file_sha256'].items()):
        path = directory / name
        actual = _file_record(path) if path.is_file() else dict(sha256=None, size=None)
        record['files'][name] = dict(actual, expected=digest, match=actual['sha256'] == digest)
    ok = record['files'] and all(f['match'] for f in record['files'].values())
    record['status'] = 'verified' if ok else 'unresolved'
    return record


def _compute(sources_path, root, out_dir):
    sources = json.loads(Path(sources_path).read_text())
    if sources.get('schema') != SOURCES_SCHEMA:
        raise ValueError(f'Expected sources schema {SOURCES_SCHEMA}.')
    audit_step, init_step = sources['audit_step'], sources['init_step']
    names = [a['arm'] for a in sources['arms']]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate arm identifiers.')
    arms, key_lists, restore = {}, {}, []
    for spec in sources['arms']:
        arms[spec['arm']], key_lists[spec['arm']] = _arm(root, spec, out_dir, audit_step, init_step)
        for step, ck in arms[spec['arm']]['checkpoints'].items():
            wanted = USED_FILES + ['complete.json']
            expected = {'adapter_model.safetensors': 'adapter_sha256', 'adapter_config.json': 'config_sha256',
                        'complete.json': 'seal_sha256'}
            for name in wanted:
                if name not in ck['files']:
                    restore.append(dict(arm=spec['arm'], step=step, file=name,
                                        expected_sha256=ck['expected'].get(expected[name], {}).get('expected'),
                                        restore=spec.get('restore')))
    states = [a['adapted_measurement'] for a in arms.values()]
    summary = dict(arms=len(arms), load_ready=states.count('load-ready'),
                   baseline=states.count('baseline-no-adapter'), unresolved=states.count('unresolved'),
                   missing=states.count('missing'))
    resolved = summary['load_ready'] + summary['baseline'] == summary['arms']
    inventory = dict(
        schema=INVENTORY_SCHEMA, sources_sha256=file_hash(sources_path), audit_step=audit_step,
        init_step=init_step, base=_base(root, sources['base']), arms=arms, summary=summary,
        all_listed_arms_resolved=resolved, seven_arm_complete=resolved and summary['arms'] == 7,
        restore_requirements=restore,
        notes=['A load-ready arm has verified payload, config and seal identities only; live PEFT '
               'routing and activity are checked later (03a/03b).',
               'Unresolved scale or targets mean the adapted activations themselves are unmeasured.'])
    return sources, inventory, key_lists


def _status(run_dir, state, **extra):
    _write_json(run_dir / 'meta/status.json', dict(state=state, updated_unix=time.time(), host=socket.gethostname(),
                                                   pid=os.getpid(), **extra))


def build(sources_path, artifact_root, output_root, run_id):
    """Build (or verify and reuse) the sealed inventory for ``run_id``."""
    root = Path(artifact_root).resolve()
    run_dir = Path(output_root).resolve() / RUN_SUBDIR / run_id
    sources, inventory, key_lists = _compute(sources_path, root, run_dir)
    seal_path = run_dir / 'complete.json'
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        if file_hash(run_dir / 'inventory.json') != seal['inventory_sha256']:
            raise ValueError('Sealed inventory failed its integrity check; investigate before rerunning.')
        saved = json.loads((run_dir / 'inventory.json').read_text())
        if saved != json.loads(_canonical(inventory)):
            raise ValueError('Sources changed since the sealed inventory; use a new run_id.')
        print(f'Inventory verified and reused: {run_dir}', flush=True)
        return saved
    _status(run_dir, 'running', run_id=run_id)
    _write_json(run_dir / 'sources.json', sources)
    for arm, steps in key_lists.items():
        for step, keys in steps.items():
            if keys is not None:
                _write_json(run_dir / 'keys' / arm / f'step-{step}.json', keys)
    _write_json(run_dir / 'inventory.json', inventory)
    _write_json(seal_path, dict(
        inventory_sha256=file_hash(run_dir / 'inventory.json'), sources_sha256=file_hash(run_dir / 'sources.json'),
        key_lists={str(p.relative_to(run_dir)): file_hash(p) for p in sorted((run_dir / 'keys').rglob('*.json'))}
        if (run_dir / 'keys').exists() else {}))
    _status(run_dir, 'complete', run_id=run_id, summary=inventory['summary'])
    print(f"Inventory sealed: {run_dir} {inventory['summary']}", flush=True)
    return json.loads((run_dir / 'inventory.json').read_text())


def verify(run_dir, artifact_root):
    """Re-hash sealed outputs and current sources; report differences without raising."""
    run_dir = Path(run_dir).resolve()
    problems = []
    seal = json.loads((run_dir / 'complete.json').read_text())
    if file_hash(run_dir / 'inventory.json') != seal['inventory_sha256']:
        problems.append('inventory.json integrity')
    if file_hash(run_dir / 'sources.json') != seal['sources_sha256']:
        problems.append('sources.json integrity')
    for rel, digest in seal['key_lists'].items():
        if not (run_dir / rel).is_file() or file_hash(run_dir / rel) != digest:
            problems.append(f'{rel} integrity')
    if not problems:
        _, current, _ = _compute(run_dir / 'sources.json', Path(artifact_root).resolve(), run_dir)
        saved = json.loads((run_dir / 'inventory.json').read_text())
        current = json.loads(_canonical(current))
        # sources.json is a copy, so only compare what the sources resolve to.
        current['sources_sha256'] = saved['sources_sha256']
        if current != saved:
            problems.append('current sources differ from the sealed inventory')
    return dict(ok=not problems, problems=problems)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--sources', required=True)
    b.add_argument('--artifact_root', required=True)
    b.add_argument('--output_root', required=True)
    b.add_argument('--run_id', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--run_dir', required=True)
    v.add_argument('--artifact_root', required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        result = build(args.sources, args.artifact_root, args.output_root, args.run_id)
        print(json.dumps(dict(summary=result['summary'], restore_requirements=len(result['restore_requirements'])),
                         indent=2))
        return 0
    result = verify(args.run_dir, args.artifact_root)
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
