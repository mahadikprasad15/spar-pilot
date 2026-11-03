"""Adapter-independent rank-one edits using each base matrix's 50% energy space."""
import re

from pilot_eval.stage0_edit_math import edit_module
from pilot_eval.stage0_spectral_math import decompose

OPERATIONS = {
    'output': 'V1', 'input': 'V2',
    'random0': 'V3-0', 'random1': 'V3-1', 'random2': 'V3-2',
    'output-magnitude': 'V4', 'input-magnitude': 'V4-input',
}


def variant_catalogue(arms):
    arms = list(arms)
    if not arms or len(set(arms)) != len(arms) or any(
        not isinstance(x, str) or not re.fullmatch(r'[a-z0-9_]+', x) for x in arms
    ):
        raise ValueError('Arm names must be unique, nonempty safe identifiers.')
    return {f'{arm}-{op}': {'source_arm': arm, 'operation': op}
            for arm in arms for op in ('original', *OPERATIONS)}


def module_edits(matrix, adapters, *, return_bases=False):
    """Edit one module across arms, computing principal bases once per base matrix."""
    variant_catalogue(adapters)
    arrays, spectrum = decompose(matrix)
    k = spectrum['k50']
    u, v = arrays['u'][:, :k], arrays['v'][:, :k]
    edits, checks = {}, {}
    for arm, factors in adapters.items():
        results, evidence, _ = edit_module(*factors, u, v)
        for operation, key in OPERATIONS.items():
            edits[f'{arm}-{operation}'] = results[key]
            checks[f'{arm}-{operation}'] = evidence[key]
    evidence = {'spectrum': spectrum, 'checks': checks}
    return (edits, evidence, (u, v)) if return_bases else (edits, evidence)


def build_modules(directory, modules, adapters, matrix_loader, *, identity, limit=None):
    """Persist each cut once. Caller must bind identity to verified source files."""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    from pilot_eval.run import _write_json, _write_state
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash, _save_frozen

    directory = Path(directory)
    modules = list(modules)
    variant_catalogue(adapters)
    if (not isinstance(identity, str) or not identity or not modules or len(set(modules)) != len(modules)
            or any(set(a) != set(modules) for a in adapters.values())
            or (limit is not None and (type(limit) is not int or limit < 0))):
        raise ValueError('Invalid build inputs.')
    # Bind the actual input factor bytes as well as the caller's verified base source identity.
    digest = hashlib.sha256()
    for arm in sorted(adapters):
        for module in sorted(modules):
            for factor in adapters[arm][module]:
                digest.update(arm.encode() + b'\0' + module.encode() + b'\0')
                digest.update(str(factor.shape).encode() + factor.dtype.str.encode() + factor.tobytes())
    config = dict(identity=identity, modules=modules, arms=list(adapters), factor_sha256=digest.hexdigest())
    _save_frozen(directory/'build-config.json', config)
    done = 0
    for module in modules:
        key = hashlib.sha256(module.encode()).hexdigest()
        folder = directory/'units'/key
        seal = folder/'complete.json'
        if seal.exists():
            marker = json.loads(seal.read_text())
            expected = dict(config_sha256=_hash(config), module=module,
                            files={name:file_hash(folder/name) for name in ('factors.npz','bases.npz','checks.json')})
            if marker != expected:
                raise ValueError('Saved cut unit changed.')
        elif limit is None or done < limit:
            folder.mkdir(parents=True, exist_ok=True)
            values, checks, (u, v) = module_edits(matrix_loader(module),
                {arm: factors[module] for arm, factors in adapters.items()}, return_bases=True)
            with (folder/'factors.npz.tmp').open('wb') as stream:
                np.savez(stream, **{f'{variant}/{part}': x for variant, pair in values.items()
                                   for part, x in zip(('a','b'),pair)})
            (folder/'factors.npz.tmp').replace(folder/'factors.npz')
            with (folder/'bases.npz.tmp').open('wb') as stream:
                np.savez(stream, u=u, v=v)
            (folder/'bases.npz.tmp').replace(folder/'bases.npz')
            _write_json(folder/'checks.json', checks)
            _write_json(seal, dict(config_sha256=_hash(config), module=module,
                        files={name:file_hash(folder/name) for name in ('factors.npz','bases.npz','checks.json')}))
            done += 1
        else:
            completed = sum((directory/'units'/hashlib.sha256(m.encode()).hexdigest()/'complete.json').exists() for m in modules)
            _write_state(directory, 'incomplete', completed, len(modules))
            return dict(state='incomplete', completed=completed, total=len(modules))
        print(f'Cut build: {module} sealed', flush=True)
    _write_state(directory, 'complete', len(modules), len(modules))
    return dict(state='complete', completed=len(modules), total=len(modules))


def assemble(directory, root, sources, originals, modules, *, identity, load_factors=None):
    """Materialize PEFT adapters from sealed units and verify their serialized factors."""
    import hashlib
    import json
    import shutil
    from pathlib import Path
    import numpy as np
    from pilot_eval.run import _write_json
    from pilot_eval.stage0_edits import _save_tensors
    from pilot_eval.stage0_sources import factors
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash

    directory, root = Path(directory), Path(root).resolve()
    config = json.loads((directory/'build-config.json').read_text())
    if config['identity'] != identity or config['modules'] != list(modules) or config['arms'] != list(sources):
        raise ValueError('Build identity mismatch.')
    for module in modules:
        folder = directory/'units'/hashlib.sha256(module.encode()).hexdigest()
        marker = json.loads((folder/'complete.json').read_text())
        expected = dict(config_sha256=_hash(config), module=module,
                        files={name:file_hash(folder/name) for name in ('factors.npz','bases.npz','checks.json')})
        if marker != expected: raise ValueError('Saved cut unit changed.')
    result = {}
    for name, definition in variant_catalogue(sources).items():
        arm, operation = definition['source_arm'], definition['operation']
        source = Path(sources[arm]); target = directory/'adapters'/name
        target.mkdir(parents=True, exist_ok=True)
        arrays = {module+suffix:factor.copy() for module,pair in originals[arm].items()
                  for suffix,factor in zip(('.lora_A.weight','.lora_B.weight'),pair)}
        if operation != 'original':
            for module in modules:
                unit = directory/'units'/hashlib.sha256(module.encode()).hexdigest()/'factors.npz'
                with np.load(unit,allow_pickle=False) as saved:
                    arrays[module+'.lora_A.weight'] = saved[name+'/a']
                    arrays[module+'.lora_B.weight'] = saved[name+'/b']
        # A previously sealed adapter may never be overwritten, even after interruption.
        complete = target/'complete.json'
        if complete.exists():
            receipt=json.loads(complete.read_text())
            expected=dict(identity=identity,variant=name,files={n:file_hash(target/n) for n in
                ('adapter_config.json','adapter_model.safetensors')})
            if receipt != expected: raise ValueError('Saved adapter changed.')
        else:
            shutil.copyfile(source/'adapter_config.json',target/'adapter_config.json')
            if operation == 'original':
                shutil.copyfile(source/'adapter_model.safetensors',target/'adapter_model.safetensors')
            else:
                _save_tensors(target/'adapter_model.safetensors',arrays)
            actual=(load_factors or factors)(target)
            if set(actual)!=set(originals[arm]) or any(
                not np.array_equal(actual[m][i],arrays[m+suffix])
                for m in actual for i,suffix in enumerate(('.lora_A.weight','.lora_B.weight'))):
                raise ValueError('Serialized adapter mismatch.')
            _write_json(complete,dict(identity=identity,variant=name,files={n:file_hash(target/n) for n in
                ('adapter_config.json','adapter_model.safetensors')}))
        result[name] = dict(adapter_path=str(target.resolve().relative_to(root)),**definition)
    return result
