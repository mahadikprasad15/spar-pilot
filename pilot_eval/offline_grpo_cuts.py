"""Offline principal cuts, bound to verified Stage 0 source evidence."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.stage0_edit_math import edit_module
from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen

OPERATIONS = {
    'output': 'V1', 'input': 'V2',
    'random0': 'V3-0', 'random1': 'V3-1', 'random2': 'V3-2',
    'output-magnitude': 'V4', 'input-magnitude': 'V4-input',
}
CONDITIONS = ('original', *OPERATIONS)


def make_edits(a, b, u, v):
    """Apply the exact historical rank-one edit and seeded controls."""
    edited, checks, _ = edit_module(a, b, u, v)
    return ({'original': (a.copy(), b.copy()), **{name: edited[key] for name, key in OPERATIONS.items()}},
            {name: checks[key] for name, key in OPERATIONS.items()})


def _identity():
    from pilot_eval import stage0_edit_math, stage0_spectral_math, stage0_evaluation_backend, stage0_evaluation_records, stage0_batching_backend, stage0_batching, scoring
    return {name: file_hash(Path(module.__file__)) for name, module in {
        'workflow': __import__(__name__, fromlist=['']), 'edit_math': stage0_edit_math,
        'spectral_math': stage0_spectral_math, 'evaluation_backend': stage0_evaluation_backend,
        'records':stage0_evaluation_records,'batch_backend':stage0_batching_backend,'batching':stage0_batching,'scorer':scoring}.items()}


def _reference(path, root):
    """Verify the archived cut run and its original sealed source plan."""
    path = inside(root, path)
    reference = json.loads(path.read_text())
    if reference.get('protocol') != 'sampling-principal-cut-evaluation-v2':
        raise ValueError('Expected the sealed Sampling SFT principal-cut evaluation')
    directory = inside(root, reference['run_path'])
    if json.loads((directory/'config.json').read_text()) != reference:
        raise ValueError('Historical cut configuration changed')
    marker = json.loads((directory/'complete.json').read_text())
    if marker['config_sha256'] != _hash(reference):
        raise ValueError('Historical cut completion identity changed')
    for name, key in [('inputs/items.json','items_sha256'), ('meta/final-integrity.json','final_integrity_sha256'),
                      ('results/responses.jsonl','responses_sha256'), ('results/results.json','results_sha256')]:
        if file_hash(directory/name) != marker[key]:
            raise ValueError('Historical cut evidence changed')
    source = json.loads((directory/'cut-sources.json').read_text())
    if source.get('protocol') != 'sampling-principal-cut-sources-v1' or source['plan_sha256'] != reference['plan_sha256']:
        raise ValueError('Historical cut source identity changed')
    plan_path = inside(root, source['plan_path'])
    plan = json.loads(plan_path.read_text())
    if _hash(plan) != source['plan_sha256'] or plan.get('protocol') != 'sampling-principal-cuts-v1':
        raise ValueError('Historical cut plan identity changed')
    historical_code={'sampling_stage0_cuts.py':'06ba73389170885440ddaa06f395cc7a5fc34acf2b1b322631d0315f05fb9f9a',
                     'sampling_stage0_cut_agreement.py':'be42c96985fb00a2110baa9bf3a5461305b4e5b8668b31d523abe1c5c4381b9b'}
    if any(reference['implementation'].get(k)!=v for k,v in historical_code.items()):
        raise ValueError('Historical cut construction or agreement implementation differs')
    for name in ['stage0_edit_math.py', 'stage0_spectral_math.py']:
        if reference['implementation'][name] != file_hash(Path(__file__).with_name(name)):
            raise ValueError('Historical cut numerical implementation differs')
    for arm in ['k0','k2','k5']:
        if {reference['variants'][f'{arm}-{op}']['operation'] for op in CONDITIONS} != set(CONDITIONS):
            raise ValueError('Historical cut condition catalogue differs')
    if not all(json.loads((directory/'meta/final-integrity.json').read_text()).get(k) is True
               for k in ['base_unchanged','adapters_unchanged']):
        raise ValueError('Historical cut model integrity failed')
    import hashlib
    build=directory/'build';build_config=json.loads((build/'build-config.json').read_text())
    if build_config['identity']!=source['plan_sha256']:
        raise ValueError('Historical cut build identity differs')
    omission_path=directory/'meta/omitted-files.json'
    omissions=json.loads(omission_path.read_text()).get('files',{}) if omission_path.exists() else {}
    ranks={}
    for module in build_config['modules']:
        folder=build/'units'/hashlib.sha256(module.encode()).hexdigest()
        seal=json.loads((folder/'complete.json').read_text())
        if seal['config_sha256']!=_hash(build_config) or seal['module']!=module or set(seal['files'])!={'factors.npz','bases.npz','checks.json'}:
            raise ValueError('Historical cut module identity changed')
        for name,digest in seal['files'].items():
            payload=folder/name
            if payload.is_file():
                if file_hash(payload)!=digest:raise ValueError('Historical cut module evidence changed')
            else:
                relative=str(payload.relative_to(directory))
                if name!='bases.npz' or omissions.get(relative)!=digest:
                    raise ValueError('Historical cut module evidence missing without a matching sealed omission')
        ranks[module]=json.loads((folder/'checks.json').read_text())['spectrum']['k50']
    return reference,ranks


def prepare(internals_path, behaviour_path, reference_path, reference_root, output_root, name):
    from pilot_eval import offline_grpo_internals, offline_grpo_behaviour
    from pilot_eval.sft import safe_name
    from pilot_eval.stage0_workflow import _writer
    root = Path(output_root).resolve()
    reference_root = Path(reference_root).resolve()
    safe_name(name)
    internals_path = inside(root, internals_path)
    behaviour_path = inside(root, behaviour_path)
    reference_path = inside(reference_root, reference_path)
    internal, _, _, _, spectral, cache = offline_grpo_internals.load(internals_path, root)
    behaviour, _ = offline_grpo_behaviour.verify(behaviour_path, root)
    reference,reference_ranks = _reference(reference_path, reference_root)
    if reference_ranks!={row['module']:row['k50'] for row in cache['module_results']}:
        raise ValueError('Offline principal ranks differ from sealed Sampling SFT cut plan')
    if internal['source']['training_sha256'] != behaviour['training_sha256']:
        raise ValueError('Behavior and internals training sources differ')
    identity = internal['source']['model_identity']
    if any(reference['model_identity'][k] != identity[k] for k in ['model','model_revision','tokenizer_revision','base_sha256']):
        raise ValueError('Historical cut model differs')
    directory = root/'runs/offline-grpo'/name/'cuts'
    plan_path = root/'plans'/name/'offline.cuts.json'
    plan = dict(protocol='offline-grpo-cuts-v1', name=name, run_path=str(directory.relative_to(root)),
        internals_path=str(internals_path.relative_to(root)), internals_sha256=file_hash(internals_path),
        behaviour_path=str(behaviour_path.relative_to(root)), behaviour_sha256=file_hash(behaviour_path),
        reference_root=str(reference_root), reference_path=str(reference_path.relative_to(reference_root)),
        reference_sha256=file_hash(reference_path), source=internal['source'], runtime=internal['runtime'],
        model_identity=identity, weights_dir=internal['weights_dir'], decoding=internal['decoding'],
        prompt_contract=internal['source']['prompt_contract'], items=behaviour['items'],
        modules=internal['modules'], spectral_path=internal['spectral_path'],
        conditions=list(CONDITIONS), energy_level=50, random_seeds=[1000,1001,1002],
        implementation=_identity(), execution_kind=internal['execution_kind'])
    with _writer(directory):
        _save_frozen(plan_path, plan)
        _save_frozen(directory/'config.json', plan)
        _save_frozen(directory/'meta/run_manifest.json', dict(config_sha256=_hash(plan),
            conditions=8, questions=150, expected_responses=1200, construction_is_behavior=False))
    return plan_path


def load(path, output_root):
    from pilot_eval import offline_grpo_internals, offline_grpo_behaviour
    root = Path(output_root).resolve()
    path = inside(root, path)
    plan = json.loads(path.read_text())
    directory = inside(root, plan['run_path'])
    if (plan.get('protocol') != 'offline-grpo-cuts-v1' or plan['implementation'] != _identity()
            or json.loads((directory/'config.json').read_text()) != plan):
        raise ValueError('Offline cut plan or implementation changed')
    for name in ['internals','behaviour']:
        if file_hash(inside(root, plan[name+'_path'])) != plan[name+'_sha256']:
            raise ValueError('Offline cut parent source changed')
    reference_root = Path(plan['reference_root'])
    reference_path = inside(reference_root, plan['reference_path'])
    if file_hash(reference_path) != plan['reference_sha256']:
        raise ValueError('Historical cut reference changed')
    _,reference_ranks = _reference(reference_path, reference_root)
    internal, _, _, _, spectral, cache = offline_grpo_internals.load(root/plan['internals_path'], root)
    behaviour, _ = offline_grpo_behaviour.verify(root/plan['behaviour_path'], root)
    if reference_ranks!={row['module']:row['k50'] for row in cache['module_results']}:
        raise ValueError('Offline principal ranks differ from sealed historical plan')
    if (internal['source'] != plan['source'] or internal['runtime'] != plan['runtime']
            or behaviour['items'] != plan['items'] or len(plan['items']) != 150
            or len({row['id'] for row in plan['items']}) != 150):
        raise ValueError('Offline source, runtime or paired cohort changed')
    return plan, directory, spectral, cache


def build(path, output_root, *, max_new_units=None):
    import hashlib
    import shutil
    from pilot_eval.stage0_edits import _save_tensors
    from pilot_eval.stage0_sources import factors
    from pilot_eval.stage0_workflow import _writer, _state
    from pilot_eval.run import _write_json
    root = Path(output_root).resolve()
    plan, directory, spectral, cache = load(path, root)
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units < 0):
        raise ValueError('Invalid cut build unit bound')
    source = inside(root, plan['source']['checkpoints']['64']['path'])
    original = factors(source)
    by_module = {row['module']: row for row in cache['module_results']}
    if set(original) != set(plan['modules']) or set(by_module) != set(plan['modules']):
        raise ValueError('Cut module source inventory differs')
    created = 0
    with _writer(directory/'build'):
        for index, module in enumerate(plan['modules']):
            folder = directory/'build/units'/hashlib.sha256(module.encode()).hexdigest()
            marker = folder/'complete.json'
            if marker.exists():
                seal = json.loads(marker.read_text())
                if seal['config_sha256'] != _hash(plan) or seal['module'] != module or any(file_hash(folder/n) != h for n,h in seal['files'].items()):
                    raise ValueError('Cut module seal changed')
                continue
            if max_new_units is not None and created >= max_new_units:
                return dict(complete=False, completed=index, total=len(plan['modules']))
            row = by_module[module]
            with np.load(inside(root, spectral['run_path'])/row['arrays_path'], allow_pickle=False) as data:
                k = row['k50']
                edits, checks = make_edits(*original[module], data['u'][:,:k], data['v'][:,:k])
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder/'factors.tmp.npz'
            np.savez_compressed(temporary, **{f'{condition}/{factor}': value for condition,pair in edits.items() for factor,value in zip(['a','b'],pair)})
            temporary.replace(folder/'factors.npz')
            _write_json(folder/'checks.json', dict(module=module,k=k,checks=checks))
            _write_json(marker, dict(config_sha256=_hash(plan),module=module,
                files={n:file_hash(folder/n) for n in ['factors.npz','checks.json']}))
            created += 1
            _state(directory/'build','running',index+1,len(plan['modules']))
        variants = {}
        for condition in CONDITIONS:
            target = directory/'build/adapters'/condition
            target.mkdir(parents=True,exist_ok=True)
            complete = target/'complete.json'
            arrays = {}
            for module in plan['modules']:
                folder = directory/'build/units'/hashlib.sha256(module.encode()).hexdigest()
                with np.load(folder/'factors.npz', allow_pickle=False) as data:
                    for factor,suffix in [('a','.lora_A.weight'),('b','.lora_B.weight')]:
                        arrays[module+suffix] = data[f'{condition}/{factor}']
            if not complete.exists():
                shutil.copyfile(source/'adapter_config.json',target/'adapter_config.json')
                if condition == 'original':
                    shutil.copyfile(source/'adapter_model.safetensors',target/'adapter_model.safetensors')
                else:
                    _save_tensors(target/'adapter_model.safetensors',arrays)
                actual = factors(target)
                if any(not np.array_equal(actual[m][i], arrays[m+s]) for m in actual
                       for i,s in enumerate(['.lora_A.weight','.lora_B.weight'])):
                    raise ValueError('Serialized cut differs from constructed factors')
                _write_json(complete, dict(config_sha256=_hash(plan),condition=condition,
                    files={n:file_hash(target/n) for n in ['adapter_config.json','adapter_model.safetensors']}))
            seal = json.loads(complete.read_text())
            if seal != dict(config_sha256=_hash(plan),condition=condition,
                            files={n:file_hash(target/n) for n in ['adapter_config.json','adapter_model.safetensors']}):
                raise ValueError('Cut adapter seal changed')
            variants[condition] = dict(adapter_path=str(target.relative_to(root)),operation=condition,source_arm='offline')
        catalogue = dict(variants=variants,construction_complete=True,behavior_measured=False)
        _save_frozen(directory/'build/catalogue.json',catalogue)
        _save_frozen(directory/'build/complete.json',dict(config_sha256=_hash(plan),catalogue_sha256=file_hash(directory/'build/catalogue.json'),
            units={m:file_hash(directory/'build/units'/hashlib.sha256(m.encode()).hexdigest()/'complete.json') for m in plan['modules']},
            adapters={c:file_hash(directory/'build/adapters'/c/'complete.json') for c in CONDITIONS}))
        _state(directory/'build','complete',len(plan['modules']),len(plan['modules']))
        return catalogue


def verify_build(path, output_root, *, deep=False, _loaded=None):
    import hashlib
    root = Path(output_root).resolve()
    plan, directory, spectral, cache = _loaded if _loaded is not None else load(path, root)
    from pilot_eval.stage0_sources import factors
    original=factors(inside(root,plan['source']['checkpoints']['64']['path']))
    rank_rows={row['module']:row for row in cache['module_results']}
    seal = json.loads((directory/'build/complete.json').read_text())
    if (seal['config_sha256'] != _hash(plan) or set(seal['units']) != set(plan['modules'])
            or set(seal['adapters']) != set(CONDITIONS)
            or seal['catalogue_sha256'] != file_hash(directory/'build/catalogue.json')):
        raise ValueError('Cut build completion differs')
    catalogue = json.loads((directory/'build/catalogue.json').read_text())
    if set(catalogue['variants']) != set(CONDITIONS):
        raise ValueError('Cut condition catalogue incomplete')
    for module in plan['modules']:
        folder = directory/'build/units'/hashlib.sha256(module.encode()).hexdigest()
        if file_hash(folder/'complete.json') != seal['units'][module]:
            raise ValueError('Cut module completion changed')
        unit = json.loads((folder/'complete.json').read_text())
        if unit['config_sha256'] != _hash(plan) or unit['module'] != module or set(unit['files']) != {'factors.npz','checks.json'}:
            raise ValueError('Cut module identity changed')
        for name, digest in unit['files'].items():
            if file_hash(folder/name) != digest:
                raise ValueError('Cut module payload changed')
        row=rank_rows[module]
        checks=json.loads((folder/'checks.json').read_text())
        if checks['module']!=module or checks['k']!=row['k50']:
            raise ValueError('Cut principal rank changed')
        with np.load(folder/'factors.npz',allow_pickle=False) as saved:
            if set(saved.files)!={f'{condition}/{key}' for condition in CONDITIONS for key in ['a','b']}:
                raise ValueError('Cut constructed factor inventory changed')
            if any(not np.array_equal(saved[f'original/{key}'],value) for key,value in zip(['a','b'],original[module])):
                raise ValueError('Original cut factors differ from source')
            if deep:
                with np.load(inside(root,spectral['run_path'])/row['arrays_path'],allow_pickle=False) as basis:
                    expected,evidence=make_edits(*original[module],basis['u'][:,:row['k50']],basis['v'][:,:row['k50']])
                if checks['checks']!=evidence:
                    raise ValueError('Cut construction checks changed')
                for condition,pair in expected.items():
                    if any(not np.array_equal(saved[f'{condition}/{key}'],value) for key,value in zip(['a','b'],pair)):
                        raise ValueError('Cut constructed factors changed')
    for condition in CONDITIONS:
        folder = directory/'build/adapters'/condition
        if file_hash(folder/'complete.json') != seal['adapters'][condition]:
            raise ValueError('Cut adapter completion changed')
        unit = json.loads((folder/'complete.json').read_text())
        if unit != dict(config_sha256=_hash(plan), condition=condition,
                        files={n:file_hash(folder/n) for n in ['adapter_config.json','adapter_model.safetensors']}):
            raise ValueError('Cut adapter payload changed')
        if inside(root, catalogue['variants'][condition]['adapter_path']) != folder:
            raise ValueError('Cut adapter path changed')
        actual=factors(folder)
        if set(actual)!=set(plan['modules']):raise ValueError('Cut adapter modules changed')
        for module in plan['modules']:
            unit_folder=directory/'build/units'/hashlib.sha256(module.encode()).hexdigest()
            with np.load(unit_folder/'factors.npz',allow_pickle=False) as saved:
                if any(not np.array_equal(actual[module][i],saved[f'{condition}/{key}']) for i,key in enumerate(['a','b'])):
                    raise ValueError('Cut adapter differs from constructed factors')
    return catalogue


def _engine_config(plan, directory, root, catalogue):
    return dict(plan, variants=catalogue['variants'], variant_order=list(CONDITIONS),
                seed=42, item_ids=[r['id'] for r in plan['items']], decoding={**plan['decoding'], 'use_cache':True},
                adapter_files={name:{n:file_hash(inside(root,v['adapter_path'])/n)
                    for n in ['adapter_model.safetensors','adapter_config.json','complete.json']}
                    for name,v in catalogue['variants'].items()}, scorer='gsm8k-flexible-v3')


def _reservation(config, execution, directory, root, runtime, approval_path):
    from pilot_eval.stage0_evaluation import _reservation as existing_reservation
    return existing_reservation(config, execution, directory, root, runtime, approval_path)


def _execution(path, output_root, stage, *, approval_path=None, dependencies=None, max_new_units=None, reservation_provider=None):
    import time
    import uuid
    from pilot_eval.run import _write_json, _write_state
    from pilot_eval.stage0_qualification_backend import deadline_guard
    from pilot_eval.stage0_workflow import _writer
    from pilot_eval.stage0_batching_backend import BatchDependencies
    root = Path(output_root).resolve()
    loaded = load(path, root)
    plan, directory, _, _ = loaded
    catalogue = verify_build(path, root, _loaded=loaded)
    config = _engine_config(plan, directory, root, catalogue)
    work = directory/stage
    from pilot_eval.stage0_batching import BATCH_GATE
    config['batch_gate'] = BATCH_GATE
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units < 0):
        raise ValueError('Invalid collection unit bound')
    if dependencies is None:
        from pilot_eval.offline_grpo_training import _verify_preflight
        training = json.loads((root/plan['source']['training_path']).read_text())
        _verify_preflight(root/training['run_path'], training)
        dependencies = BatchDependencies()
    with _writer(directory/'execution'):
        if max_new_units==0:
            return dict(complete=False,reason='zero-unit inspection; no model loaded')
        if (work/'complete.json').exists():
            return _verify_stage(config, plan, directory, root, stage)
        runtime = dependencies.runtime()
        if runtime != plan['runtime']:
            raise ValueError('Cut evaluation runtime differs from approved runtime')
        reservation, remaining = (reservation_provider or _reservation)(config,
            {'reservation':{'reservation_id':None,'job_sha256':_hash(config)}},
            directory, root, runtime, approval_path)
        attempt = directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],runtime_sha256=_hash(runtime)))
        started = time.monotonic()
        engine = None
        try:
            with deadline_guard(remaining, plan['execution_kind']=='real-cuda'):
                _save_frozen(work/'meta/runtime.json', runtime)
                engine = dependencies.engine(config, root)
                items = plan['items']
                tokenized = engine.prepare_inputs(items)
                if [r['id'] for r in tokenized] != [r['id'] for r in items]:
                    raise ValueError('Tokenized paired cohort differs')
                for saved,item in zip(tokenized,items):
                    if (not saved.get('input_ids') or saved.get('attention_mask') != [1]*len(saved['input_ids'])
                            or ('prompt_tokens' in item and len(saved['input_ids'])!=item['prompt_tokens'])):
                        raise ValueError('Tokenized paired prompt or mask differs')
                _save_frozen(work/'inputs/tokenized.json', tokenized)
                if stage == 'agreement':
                    result = _agreement(config, work, items, engine, max_new_units, attempt)
                elif stage == 'evaluation':
                    agreement = _verify_stage(config, plan, directory, root, 'agreement')
                    if tokenized!=json.loads((directory/'agreement/inputs/tokenized.json').read_text()):
                        raise ValueError('Evaluation prompts differ from batch agreement')
                    result = _evaluate(config, work, items, engine, agreement, max_new_units, attempt)
                else:
                    raise ValueError('Unsupported cut execution stage')
                integrity = engine.check_integrity(full=True)
                _write_json(attempt/'final-integrity.json', integrity)
                if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):
                    raise ValueError('Cut collection full model integrity failed')
                if result.get('complete'):
                    _save_frozen(work/'complete.json',dict(config_sha256=_hash(config),stage=stage,
                        result_sha256=file_hash(work/'results/results.json'),
                        final_attempt=attempt.name,final_sha256=file_hash(attempt/'final-integrity.json'),
                        agreement_sha256=file_hash(directory/'agreement/complete.json') if stage=='evaluation' else None))
                return result
        except (Exception,KeyboardInterrupt) as exc:
            if engine is not None and 'integrity' not in str(exc).lower() and time.monotonic()-started < remaining:
                try:
                    with deadline_guard(remaining-(time.monotonic()-started),plan['execution_kind']=='real-cuda'):
                        _write_json(attempt/'final-integrity.json',engine.check_integrity(full=True))
                except Exception:
                    pass
            _write_state(work,'failed',0,1,error=str(exc))
            raise
        finally:
            if engine is not None:
                engine.close()
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,
                reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),
                final_integrity_sha256=file_hash(attempt/'final-integrity.json') if (attempt/'final-integrity.json').exists() else None))
            _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def _unit(work, config, key):
    from pilot_eval.stage0_evaluation import _attempt_timing
    folder = work/'units'/key
    marker = folder/'complete.json'
    if not marker.exists():
        return None
    seal = json.loads(marker.read_text())
    if seal['config_sha256'] != _hash(config) or seal['result_sha256'] != file_hash(folder/'result.json'):
        raise ValueError('Cut unit seal changed')
    value = json.loads((folder/'result.json').read_text())
    attempt = inside(work.parent/'attempts', value['attempt_id'])
    final = attempt/'final-integrity.json'
    if not all(value.get('integrity',{}).get(k) is True for k in ['base_unchanged','adapters_unchanged']):
        raise ValueError('Cut unit integrity proof missing')
    if (not final.exists() or not all(json.loads(final.read_text()).get(k) is True
            for k in ['base_unchanged','adapters_unchanged'])):
        raise ValueError('Cut attempt lacks successful full integrity proof')
    timing = _attempt_timing(attempt)
    if json.loads((attempt/'start.json').read_text()).get('runtime_sha256')!=_hash(config['runtime']):
        raise ValueError('Cut attempt runtime identity changed')
    if timing['final_integrity_sha256'] != file_hash(final) or value['attempt_start_sha256'] != file_hash(attempt/'start.json'):
        raise ValueError('Cut attempt timing or identity changed')
    return value


def _save_unit(work, config, key, value, attempt):
    from pilot_eval.run import _write_json
    folder = work/'units'/key
    value = dict(value, attempt_id=attempt.name, attempt_start_sha256=file_hash(attempt/'start.json'))
    _write_json(folder/'result.json', value)
    _write_json(folder/'complete.json',dict(config_sha256=_hash(config),result_sha256=file_hash(folder/'result.json')))
    return value


def _agreement(config, work, items, engine, limit, attempt):
    from pilot_eval.run import _write_json
    from pilot_eval.stage0_batching import BATCH_GATE, numeric_ok
    selected = items[:16]
    eos = config['decoding']['eos_token_id']
    token_rows = []
    for row in selected:
        prompt = engine.inputs[row['id']]
        full = prompt + engine.tokenizer.encode(row['gold'],add_special_tokens=False) + [eos]
        token_rows.append(dict(id=row['id'],input_ids=full,attention_mask=[1]*len(full),
                               prompt_tokens=len(prompt),answer_positions=list(range(len(prompt)-1,len(full)-1))))
    values = {}
    created = 0
    for condition in CONDITIONS:
        existing = _unit(work,config,condition)
        if existing is not None:
            values[condition] = existing
            continue
        if limit is not None and created >= limit:
            return dict(complete=False,completed=len(values),total=len(CONDITIONS))
        numeric = [engine.compare(selected[:size],condition,token_rows) for size in [16,6]]
        passed = all(numeric_ok(result,BATCH_GATE) for result in numeric)
        replay = None
        checks = []
        if condition == 'output':
            references = [engine.generate([row],condition,42)[0] for row in selected]
            for size in [16,6]:
                responses = engine.generate(selected[:size],condition,42)
                for index,(reference,actual) in enumerate(zip(references[:size],responses)):
                    exact = (reference['generated_token_ids']==actual['generated_token_ids']
                             and reference['stop_reason']==actual['stop_reason'])
                    evidence = None
                    near = False
                    if not exact:
                        evidence = engine.divergence(selected[:size],condition,index,reference['generated_token_ids'],responses)
                        near = bool(evidence.get('near_tie')) and reference['stop_reason']==actual['stop_reason']
                    checks.append(dict(item_id=selected[index]['id'],reference=reference,actual=actual,exact=exact,near_tie=near,evidence=evidence,stop_matches=reference['stop_reason']==actual['stop_reason']))
            replay = all(row['exact'] or row['near_tie'] for row in checks)
        integrity = engine.check_integrity()
        if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):
            raise ValueError('Cut agreement model integrity failed')
        values[condition] = _save_unit(work,config,condition,dict(numeric=numeric,numeric_passed=passed,replay_passed=replay,replay_checks=checks,integrity=integrity),attempt)
        created += 1
    replay_passed = values['output']['replay_passed']
    result = dict(complete=True, batch_gate=BATCH_GATE, sizes=[16,6],
        batch_sizes={condition:16 if values[condition]['numeric_passed'] and replay_passed else 1 for condition in CONDITIONS},
        note='Batching requires all-condition teacher-forced agreement and output-cut replay; otherwise singleton.')
    _write_json(work/'results/results.json',result)
    return result


def _evaluate(config, work, items, engine, agreement, limit, attempt):
    from pilot_eval.run import _write_json, _write_state
    from pilot_eval.stage0_evaluation_records import records
    rows = []
    created = 0
    runtime_hash = file_hash(work/'meta/runtime.json')
    input_hash = file_hash(work/'inputs/tokenized.json')
    for condition in CONDITIONS:
        size = agreement['batch_sizes'][condition]
        for start in range(0,len(items),16):
            selected = items[start:start+16]
            key = f'{condition}-{start:04d}'
            unit = dict(key=key,variant=condition,ids=[r['id'] for r in selected],seed=42+start//16)
            unit['item_seeds'] = dict.fromkeys(unit['ids'],unit['seed'])
            value = _unit(work,config,key)
            if value is None:
                if limit is not None and created >= limit:
                    return dict(complete=False,completed=len(rows),total=1200)
                outputs = []
                for offset in range(0,len(selected),size):
                    outputs.extend(engine.generate(selected[offset:offset+size],condition,unit['seed']))
                scored = records(config,unit,selected,outputs,runtime_sha256=runtime_hash,input_sha256=input_hash)
                integrity = engine.check_integrity()
                if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):
                    raise ValueError('Cut evaluation model integrity failed')
                value = _save_unit(work,config,key,dict(records=scored,integrity=integrity),attempt)
                created += 1
            outputs = [{k:r[k] for k in ['text','token_ids','generated_token_ids','stop_reason']} for r in value['records']]
            if value['records'] != records(config,unit,selected,outputs,runtime_sha256=runtime_hash,input_sha256=input_hash):
                raise ValueError('Cut response identity or score changed')
            rows.extend(value['records'])
            _write_state(work,'running',len(rows),1200)
            print(f'Offline cuts: {len(rows)}/1200 paired responses',flush=True)
    if len(rows)!=1200 or len({r['record_id'] for r in rows})!=1200:
        raise ValueError('Cut paired response coverage incomplete')
    from pilot_eval.stage0_evaluation import _jsonl
    _jsonl(work/'results/responses.jsonl',rows)
    result = dict(complete=True,responses=1200,conditions=list(CONDITIONS),batch_sizes=agreement['batch_sizes'])
    _write_json(work/'results/results.json',result)
    return result


def verify_stage(path, output_root, stage):
    from pilot_eval.stage0_batching import BATCH_GATE
    root = Path(output_root).resolve()
    loaded = load(path, root)
    plan, directory, _, _ = loaded
    catalogue = verify_build(path, root, _loaded=loaded)
    config = _engine_config(plan, directory, root, catalogue)
    config['batch_gate'] = BATCH_GATE
    return _verify_stage(config, plan, directory, root, stage)


def _verify_stage(config, plan, directory, root, stage):
    from pilot_eval.stage0_batching import BATCH_GATE, numeric_ok
    from pilot_eval.stage0_evaluation_records import records
    work = directory/stage
    marker = json.loads((work/'complete.json').read_text())
    final = inside(directory/'attempts',marker['final_attempt'])/'final-integrity.json'
    if (marker['config_sha256'] != _hash(config) or marker['stage'] != stage
            or marker['result_sha256'] != file_hash(work/'results/results.json')
            or marker['final_sha256'] != file_hash(final)
            or not all(json.loads(final.read_text()).get(k) is True for k in ['base_unchanged','adapters_unchanged'])
            or json.loads((work/'meta/runtime.json').read_text()) != plan['runtime']):
        raise ValueError('Cut stage completion or runtime changed')
    result = json.loads((work/'results/results.json').read_text())
    tokenized=json.loads((work/'inputs/tokenized.json').read_text())
    if [r['id'] for r in tokenized]!=[r['id'] for r in plan['items']]:
        raise ValueError('Cut tokenized cohort changed')
    for saved,item in zip(tokenized,plan['items']):
        if (not saved.get('input_ids') or saved.get('attention_mask')!=[1]*len(saved['input_ids'])
                or ('prompt_tokens' in item and len(saved['input_ids'])!=item['prompt_tokens'])):
            raise ValueError('Cut tokenized prompt or mask changed')
    if stage == 'agreement':
        values = {condition:_unit(work,config,condition) for condition in CONDITIONS}
        if any(value is None for value in values.values()):
            raise ValueError('Cut agreement units incomplete')
        checks=values['output']['replay_checks']
        expected_ids=[item['id'] for item in plan['items'][:16]]+[item['id'] for item in plan['items'][:6]]
        if len(checks)!=22 or [c['item_id'] for c in checks]!=expected_ids:
            raise ValueError('Cut greedy replay cohort changed')
        for condition in CONDITIONS:
            numeric=values[condition]['numeric']
            if len(numeric)!=2 or any([row['id'] for row in measurement['examples']]!=[item['id'] for item in plan['items'][:size]] for measurement,size in zip(numeric,[16,6])):
                raise ValueError('Cut numeric agreement cohort changed')
        if any((c['stop_matches']!=(c['reference']['stop_reason']==c['actual']['stop_reason'])
                or c['exact']!=(c['stop_matches'] and c['reference']['generated_token_ids']==c['actual']['generated_token_ids'])
                or (not c['stop_matches'] and (c['exact'] or c['near_tie']))) for c in checks):
            raise ValueError('Cut greedy replay evidence incomplete')
        if any(c['near_tie'] and (not c['evidence'] or not c['evidence'].get('near_tie')
                or c['evidence']['measured_error']>BATCH_GATE
                or c['evidence']['top2_margin']>=c['evidence']['measured_error']
                or c['evidence']['chosen_gap']>=c['evidence']['measured_error']) for c in checks):
            raise ValueError('Cut greedy near-tie evidence changed')
        if values['output']['replay_passed']!=all(c['exact'] or c['near_tie'] for c in checks):
            raise ValueError('Cut replay decision changed')
        expected = {condition:16 if all(numeric_ok(v,BATCH_GATE) for v in values[condition]['numeric'])
                    and values['output']['replay_passed'] else 1 for condition in CONDITIONS}
        if result['batch_gate'] != BATCH_GATE or result['sizes'] != [16,6] or result['batch_sizes'] != expected:
            raise ValueError('Cut batch agreement result changed')
    elif stage == 'evaluation':
        if marker['agreement_sha256']!=file_hash(directory/'agreement/complete.json'):
            raise ValueError('Cut agreement completion changed')
        agreement = _verify_stage(config,plan,directory,root,'agreement')
        if tokenized!=json.loads((directory/'agreement/inputs/tokenized.json').read_text()):
            raise ValueError('Evaluation prompts differ from batch agreement')
        runtime_hash = file_hash(work/'meta/runtime.json')
        input_hash = file_hash(work/'inputs/tokenized.json')
        all_rows = []
        for condition in CONDITIONS:
            for start in range(0,150,16):
                selected = plan['items'][start:start+16]
                key = f'{condition}-{start:04d}'
                unit = dict(key=key,variant=condition,ids=[r['id'] for r in selected],seed=42+start//16)
                unit['item_seeds'] = dict.fromkeys(unit['ids'],unit['seed'])
                value = _unit(work,config,key)
                if value is None:
                    raise ValueError('Cut response unit missing')
                outputs = [{k:r[k] for k in ['text','token_ids','generated_token_ids','stop_reason']} for r in value['records']]
                if value['records'] != records(config,unit,selected,outputs,runtime_sha256=runtime_hash,input_sha256=input_hash):
                    raise ValueError('Cut response scores changed')
                all_rows.extend(value['records'])
        saved = [json.loads(line) for line in (work/'results/responses.jsonl').read_text().splitlines()]
        if (len(all_rows)!=1200 or len({r['record_id'] for r in all_rows})!=1200 or saved!=all_rows
                or result['batch_sizes']!=agreement['batch_sizes']):
            raise ValueError('Cut response cohort or batch identity changed')
    else:
        raise ValueError('Unsupported cut execution stage')
    return result


def agreement(path, output_root, **kwargs):
    return _execution(path,output_root,'agreement',**kwargs)


def evaluate(path, output_root, **kwargs):
    return _execution(path,output_root,'evaluation',**kwargs)


def report(path, output_root):
    from pilot_eval.stage0_evaluation_records import summarize
    from pilot_eval.run import _write_json
    from pilot_eval.stage0_workflow import _writer
    root = Path(output_root).resolve()
    from pilot_eval.stage0_batching import BATCH_GATE
    loaded=load(path,root)
    plan,directory,_,_=loaded
    catalogue=verify_build(path,root,_loaded=loaded)
    config=_engine_config(plan,directory,root,catalogue)
    config['batch_gate']=BATCH_GATE
    _verify_stage(config,plan,directory,root,'evaluation')
    rows = [json.loads(line) for line in (directory/'evaluation/results/responses.jsonl').read_text().splitlines()]
    summaries = summarize(config,rows)
    for row in summaries['variants'].values():
        row['reference']='original'
    by_condition = {condition:{r['item_id']:r for r in rows if r['variant']==condition} for condition in CONDITIONS}
    ids = [r['id'] for r in plan['items']]
    rng = np.random.default_rng(42)
    draws = rng.integers(0,150,size=(10000,150))
    paired = {}
    transitions = []
    for condition in CONDITIONS:
        delta = np.array([by_condition[condition][i]['reward']-by_condition['original'][i]['reward'] for i in ids])
        paired[condition] = dict(estimate=float(delta.mean()),interval_95=np.percentile(delta[draws].mean(axis=1),[2.5,97.5]).tolist())
        transitions.extend(dict(condition=condition,item_id=i,original=by_condition['original'][i]['reward'],
                                edited=by_condition[condition][i]['reward']) for i in ids)
    result = dict(protocol='offline-grpo-cut-report-v1',execution_kind=plan['execution_kind'],summaries=summaries,
                  paired=paired,transitions=transitions,original_endpoint_reused=False,
                  note='Paired item bootstrap intervals are pointwise. Cut construction alone is not behavioral evidence.')
    with _writer(directory/'report'):
        _save_frozen(directory/'report/results.json',result)
        lines=['# Offline principal cuts','','Paired comparisons against the original adapter.','',
               '| Condition | Change in accuracy | 95% interval |','|---|---:|---|']
        lines.extend(f"| {condition} | {paired[condition]['estimate']:.3f} | {paired[condition]['interval_95']} |" for condition in CONDITIONS)
        report_path=directory/'report/report.md'
        text='\n'.join(lines)+'\n'
        if report_path.exists() and report_path.read_text()!=text:
            raise ValueError('Cut report changed')
        report_path.write_text(text)
        _save_frozen(directory/'report/complete.json',dict(config_sha256=_hash(plan),
            evaluation_sha256=file_hash(directory/'evaluation/complete.json'),
            results_sha256=file_hash(directory/'report/results.json'),markdown_sha256=file_hash(report_path)))
    return result


def verify_report(path, output_root):
    root=Path(output_root).resolve()
    from pilot_eval.stage0_batching import BATCH_GATE
    loaded=load(path,root)
    plan,directory,_,_=loaded
    catalogue=verify_build(path,root,_loaded=loaded)
    config=_engine_config(plan,directory,root,catalogue)
    config['batch_gate']=BATCH_GATE
    _verify_stage(config,plan,directory,root,'evaluation')
    marker=json.loads((directory/'report/complete.json').read_text())
    expected=dict(config_sha256=_hash(plan),evaluation_sha256=file_hash(directory/'evaluation/complete.json'),
                  results_sha256=file_hash(directory/'report/results.json'),markdown_sha256=file_hash(directory/'report/report.md'))
    if marker!=expected:
        raise ValueError('Cut report seal changed')
    return dict(complete=True,execution_kind=plan['execution_kind'])
