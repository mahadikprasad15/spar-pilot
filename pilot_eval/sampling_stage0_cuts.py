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


def collect(config, items, directory, runtime, engine, *, limit=None):
    """Singleton generation with independently sealed records and strict resume identity."""
    import hashlib
    import json
    import statistics
    from pathlib import Path
    from pilot_eval.run import _write_json, _write_state
    from pilot_eval.stage0_evaluation import _jsonl
    from pilot_eval.stage0_evaluation_records import OUTPUT_FIELDS, records
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash, _save_frozen

    from pilot_eval.sampling_stage0_evaluation import environment
    runtime = environment(runtime)
    directory = Path(directory)
    variants = config['variant_order']
    if (not variants or len(set(variants)) != len(variants) or set(variants) != set(config['variants'])
            or not items or len({i['id'] for i in items}) != len(items)
            or (limit is not None and (type(limit) is not int or limit < 0))):
        raise ValueError('Invalid evaluation scope.')
    if (directory/'meta/integrity-failure.json').exists():
        raise ValueError('Prior integrity failure; preserve the run.')
    for path,value,label in [(directory/'config.json',config,'config'),(directory/'meta/runtime.json',runtime,'runtime')]:
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Resume {label} differs from saved {label}.')
        _save_frozen(path,value)
    _save_frozen(directory/'inputs/items.json', items)
    inputs=engine.prepare_inputs(items)
    if [r['id'] for r in inputs] != [r['id'] for r in items]: raise ValueError('Input order mismatch.')
    for row,item in zip(inputs,items):
        if not row['input_ids'] or row['attention_mask'] != [1]*len(row['input_ids']):
            raise ValueError('Invalid tokenized input.')
        row['prompt_sha256']=_hash(item['prompt'])
    _save_frozen(directory/'inputs/tokenized-prompts.json',inputs)
    rh=file_hash(directory/'meta/runtime.json'); ih=file_hash(directory/'inputs/tokenized-prompts.json'); items_hash=file_hash(directory/'inputs/items.json')
    all_rows=[]; new=0; total=len(items)*len(variants)
    def check(full=False):
        try:
            proof=engine.check_integrity(full=full)
            if not all(proof.get(k) is True for k in ('base_unchanged','adapters_unchanged')):
                raise ValueError('Model integrity failed.')
            return proof
        except Exception as exc:
            _write_json(directory/'meta/integrity-failure.json',dict(error=str(exc)))
            raise
    for variant in variants:
        for index,item in enumerate(items):
            key=hashlib.sha256((variant+'\0'+str(item['id'])).encode()).hexdigest()
            unit=dict(key=key,variant=variant,ids=[item['id']],seed=config['seed']+index//16)
            folder=directory/'units'/key; seal=folder/'complete.json'; payload=folder/'responses.jsonl'; proof=folder/'integrity.json'
            if seal.exists():
                marker=json.loads(seal.read_text())
                expected=dict(config_sha256=_hash(config),runtime_sha256=rh,input_sha256=ih,items_sha256=items_hash,
                              responses_sha256=file_hash(payload),integrity_sha256=file_hash(proof))
                if marker != expected: raise ValueError('Saved response unit changed.')
                evidence=json.loads(proof.read_text())
                if any(not all(evidence[s].get(k) is True for k in ('base_unchanged','adapters_unchanged')) for s in ('before','after')):
                    raise ValueError('Saved unit integrity failed.')
                saved=[json.loads(line) for line in payload.read_text().splitlines()]
                output=[{k:r[k] for k in OUTPUT_FIELDS} for r in saved]
                if saved != records(config,unit,[item],output,runtime_sha256=rh,input_sha256=ih):
                    raise ValueError('Saved response scoring changed.')
            elif limit is None or new < limit:
                before=check(); output=engine.generate([item],variant,unit['seed']); after=check()
                saved=records(config,unit,[item],output,runtime_sha256=rh,input_sha256=ih)
                _jsonl(payload,saved); _write_json(proof,dict(before=before,after=after))
                _write_json(seal,dict(config_sha256=_hash(config),runtime_sha256=rh,input_sha256=ih,items_sha256=items_hash,
                            responses_sha256=file_hash(payload),integrity_sha256=file_hash(proof)))
                new += 1
                print(f'Cut evaluation: {len(all_rows)+1}/{total} {variant} {item["id"]}',flush=True)
            else:
                _write_state(directory,'incomplete',len(all_rows),total)
                return dict(state='incomplete',completed=len(all_rows),total=total)
            all_rows.extend(saved)
            _write_state(directory,'running',len(all_rows),total)
    check(full=True)
    summaries={}
    for variant in variants:
        rows=[r for r in all_rows if r['variant']==variant]
        summaries[variant]=dict(n=len(rows),accuracy=sum(r['flexible']['correct'] for r in rows)/len(rows),
            strict_accuracy=sum(r['strict']['correct'] for r in rows)/len(rows),
            mean_tokens=statistics.mean(r['token_count'] for r in rows),cap_count=sum(r['capped'] for r in rows))
    result=dict(state='complete',total=total,variants=summaries,batch_size=1)
    response=directory/'results/responses.jsonl'; summary=directory/'results/results.json'; complete=directory/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text()) != dict(config_sha256=_hash(config),items_sha256=items_hash,responses_sha256=file_hash(response),results_sha256=file_hash(summary)):
            raise ValueError('Completed result changed.')
        if [json.loads(x) for x in response.read_text().splitlines()] != all_rows or json.loads(summary.read_text()) != result:
            raise ValueError('Completed aggregate changed.')
    else:
        _jsonl(response,all_rows);_write_json(summary,result)
        _write_json(complete,dict(config_sha256=_hash(config),items_sha256=items_hash,responses_sha256=file_hash(response),results_sha256=file_hash(summary)))
    _write_state(directory,'complete',total,total)
    return result
