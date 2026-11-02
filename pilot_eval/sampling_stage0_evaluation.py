"""Source-bound singleton evaluation of the three sampling-SFT endpoints.

This is a new measurement, not a Ticket 04 numerical qualification receipt.
"""
import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_evaluation_records import OUTPUT_FIELDS, records
from pilot_eval.stage0_sources import inside, checkpoint
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen

ARMS = ('k0', 'k2', 'k5')


def snapshot_files(directory):
    """Check the local model manifest and all shards; loaded base hash is the weight identity."""
    directory=Path(directory); config=directory/'config.json'; index=directory/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text()).get('weight_map',{})
        if not isinstance(mapping,dict) or not mapping:raise ValueError('Empty model weight index.')
        names=sorted(set(mapping.values()))
    else:
        names=['model.safetensors']
    if any(not isinstance(n,str) or Path(n).name!=n or not n.endswith('.safetensors') for n in names):
        raise ValueError('Unsafe or invalid model weight shard name.')
    sizes={n:(directory/n).stat().st_size for n in names}
    if any(size<=0 for size in sizes.values()):raise ValueError('Empty model weight shard.')
    return dict(config_sha256=file_hash(config),index_sha256=file_hash(index) if index.exists() else None,weight_sizes=sizes)


def _integrity(engine, full=False):
    proof = engine.check_integrity(full=full)
    if not all(proof.get(k) is True for k in ('base_unchanged', 'adapters_unchanged')):
        raise ValueError('Base or adapter integrity failed.')
    return proof


def _completion(config, directory):
    value=dict(config_sha256=_hash(config), responses_sha256=file_hash(directory/'results/responses.jsonl'),
               results_sha256=file_hash(directory/'results/results.json'))
    if config.get('execution_kind')=='real-cuda':
        value['runtime_evidence']={name:file_hash(directory/name) for name in
            ['meta/runtime.json','meta/direct-smoke.json','meta/loaded-adapters.json','meta/final-integrity.json']}
    return value


def _units(config, items):
    size=config.get('unit_size',1)
    if type(size) is not int or not 1<=size<=150:
        raise ValueError('Durable unit size must be between 1 and 150.')
    for start in range(0,len(items),size):
        selected=items[start:start+size]
        for arm in ARMS:
            seeds={row['id']:config['seed']+start+i for i,row in enumerate(selected)}
            yield dict(key=f'{start:04d}-{arm}',variant=arm,ids=[row['id'] for row in selected],
                       seed=config['seed']+start,item_seeds=seeds),selected


def collect(config, items, directory, runtime, engine, *, max_new_units=None):
    """Collect singleton calls in durable groups; caller owns the run lock."""
    directory = Path(directory)
    if (directory/'meta/integrity-failure.json').exists():
        raise ValueError('Prior integrity failure; preserve this run for investigation.')
    def check(full=False):
        try:
            return _integrity(engine, full=full)
        except Exception as exc:
            _write_json(directory/'meta/integrity-failure.json', {'error': str(exc)})
            raise
    if max_new_units is not None and (type(max_new_units) is not int or max_new_units < 0):
        raise ValueError('max_new_units must be a nonnegative integer.')
    ids = [row['id'] for row in items]
    if not ids or len(ids) != len(set(ids)) or list(config['variant_order']) != list(ARMS):
        raise ValueError('Invalid cohort or ordered sampling arms.')
    runtime_path = directory/'meta/runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError('Resume runtime differs from the saved runtime.')
    _save_frozen(directory/'config.json', config)
    _save_frozen(runtime_path, runtime)
    inputs = engine.prepare_inputs(items)
    if [r['id'] for r in inputs] != ids:
        raise ValueError('Tokenized input order mismatch.')
    for row, item in zip(inputs, items):
        tokens = row['input_ids']
        if (not tokens or row['attention_mask'] != [1]*len(tokens)
                or ('prompt_tokens' in item and len(tokens) != item['prompt_tokens'])):
            raise ValueError('Prompt tokenization mismatch.')
        row['prompt_sha256'] = _hash(item['prompt'])
    input_path = directory/'inputs/tokenized-prompts.json'
    _save_frozen(input_path, inputs)
    runtime_hash = file_hash(runtime_path); input_hash = file_hash(input_path)
    all_rows = []; new = 0; total = len(items)*len(ARMS)
    for unit, selected in _units(config,items):
        folder = directory/'units'/unit['key']; payload=folder/'responses.jsonl'; seal=folder/'complete.json'; metadata=folder/'integrity.json'
        if seal.exists():
            marker=json.loads(seal.read_text())
            if marker != {'config_sha256': _hash(config), 'runtime_sha256': runtime_hash,
                          'input_sha256': input_hash, 'responses_sha256': file_hash(payload),
                          'integrity_sha256': file_hash(metadata)}:
                raise ValueError('Saved unit seal changed.')
            proof=json.loads(metadata.read_text())
            if any(not all(proof[stage].get(k) is True for k in ['base_unchanged','adapters_unchanged']) for stage in ['before','after']):
                raise ValueError('Saved unit integrity failed.')
            saved=[json.loads(line) for line in payload.read_text().splitlines()]
            outputs=[{key: row[key] for key in OUTPUT_FIELDS} for row in saved]
            if saved != records(config, unit, selected, outputs, runtime_sha256=runtime_hash, input_sha256=input_hash):
                raise ValueError('Saved unit record mismatch.')
            all_rows.extend(saved)
            continue
        if max_new_units is not None and new >= max_new_units:
            continue
        before=check()
        outputs=[]
        for item in selected:
            outputs.extend(engine.generate([item], unit['variant'], unit['item_seeds'][item['id']]))
            print(f'Sampling Stage 0: {unit["key"]} {len(outputs)}/{len(selected)} generated; pending seal.', flush=True)
        batch=records(config, unit, selected, outputs, runtime_sha256=runtime_hash, input_sha256=input_hash)
        after=check()
        _jsonl(payload, batch)
        _write_json(metadata,dict(before=before,after=after))
        _write_json(seal, dict(config_sha256=_hash(config), runtime_sha256=runtime_hash,
                               input_sha256=input_hash, responses_sha256=file_hash(payload),
                               integrity_sha256=file_hash(metadata)))
        all_rows.extend(batch); new += 1
        _write_state(directory, 'running', len(all_rows), total)
        print(f'Sampling Stage 0: {len(all_rows)}/{total} responses; {unit["key"]}', flush=True)
    final_integrity=check(full=True)
    if len(all_rows) != total:
        _write_state(directory, 'incomplete', len(all_rows), total)
        return {'state': 'incomplete', 'completed': len(all_rows), 'total': total}
    completion=directory/'complete.json'
    if completion.exists():
        marker=json.loads(completion.read_text())
        if marker != _completion(config,directory):
            raise ValueError('Completed result seal changed.')
        saved=[json.loads(line) for line in (directory/'results/responses.jsonl').read_text().splitlines()]
        if saved != all_rows:raise ValueError('Completed response aggregate mismatch.')
        return json.loads((directory/'results/results.json').read_text())
    if config.get('execution_kind')=='real-cuda':
        _write_json(directory/'meta/final-integrity.json',final_integrity)
    _jsonl(directory/'results/responses.jsonl', all_rows)
    summary={}
    for arm in ARMS:
        rows=[r for r in all_rows if r['variant']==arm]
        summary[arm]=dict(n=len(rows), accuracy=sum(r['flexible']['correct'] for r in rows)/len(rows),
                          mean_tokens=statistics.mean(r['token_count'] for r in rows),
                          median_tokens=statistics.median(r['token_count'] for r in rows),
                          cap_count=sum(r['capped'] for r in rows))
    result={'state':'complete', 'total':total, 'variants':summary, 'runtime':runtime,
            'limitations':['automatic scores; no human review', 'GPU execution requires independent qualification']}
    _write_json(directory/'results/results.json',result)
    _write_json(directory/'complete.json',_completion(config,directory))
    _write_state(directory,'complete',total,total)
    return result


def prepare(manifest_path, output_root, name, *, unit_size=8):
    """Freeze verified sources. All paths must be contained in output_root."""
    from pilot_eval import sampling_sft_training as training
    root=Path(output_root).resolve(); safe_name(name)
    if type(unit_size) is not int or not 1<=unit_size<=150:
        raise ValueError('Durable unit size must be between 1 and 150.')
    manifest_path=inside(root,manifest_path); manifest=json.loads(manifest_path.read_text())
    if set(manifest['training_plans']) != set(ARMS):
        raise ValueError('Exactly k0, k2, and k5 training arms are required.')
    gold_path=inside(root,manifest['gold_config']); gold=json.loads(gold_path.read_text())
    items_path=inside(root,gold['evaluation_items_path']); items=json.loads(items_path.read_text())
    if len(items)!=150 or len({r['id'] for r in items})!=150 or _hash(items)!=gold['evaluation_items_sha256']:
        raise ValueError('Original 150-item evaluation cohort identity changed.')
    decoding=gold['source_config']['decoding']
    if (decoding.get('do_sample') is not False or decoding.get('num_beams')!=1
            or decoding.get('num_return_sequences')!=1 or decoding.get('max_new_tokens')!=1024):
        raise ValueError('Original greedy decoding contract changed.')
    parents={str(p.relative_to(root)):file_hash(p) for p in [manifest_path,gold_path,items_path]}
    variants={}; adapter_files={}; base=None
    for arm in ARMS:
        path=inside(root,manifest['training_plans'][arm])
        result=training.verify(path,root)
        _,source,folder=training.load(path,root)
        if source['k'] != int(arm[1:]):raise ValueError('Training arm identity mismatch.')
        if inside(root,source['gold_config']) != gold_path:
            raise ValueError('Training arm is bound to a different Gold configuration.')
        if result.get('execution_kind')!='real-cuda':raise ValueError('Training source is not real CUDA evidence.')
        if base is None:base=source['expected_base_sha256']
        if source['expected_base_sha256']!=base or result.get('base_sha256_before')!=base:
            raise ValueError('Training base identity mismatch.')
        cp=folder/'checkpoints/checkpoint-64'; seal=cp/'complete.json'
        payload=cp/'adapter_model.safetensors'
        evidence=dict(path=str(cp.relative_to(root)), complete_sha256=file_hash(seal),adapter_sha256=file_hash(payload))
        checked=checkpoint(root,evidence,'64',parents)
        if checked['state']!='verified-adapter-payload':raise ValueError('Endpoint checkpoint is unavailable.')
        parents[str(path.relative_to(root))]=file_hash(path)
        variants[arm]=dict(adapter_path=evidence['path'], source_arm=arm, operation='original')
        adapter_files[arm]={n:file_hash(cp/n) for n in ['complete.json','adapter_config.json','adapter_model.safetensors']}
    weights=inside(root,manifest['weights_dir'])
    for name in ['tokenizer.json','tokenizer_config.json']:
        asset=weights/name
        parents[str(asset.relative_to(root))]=file_hash(asset)
    config=dict(protocol='sampling-stage0-evaluation-v1',name=name,run_path=f'runs/stage0/{name}/sampling-behaviour',
        parents=parents,cohort_path=str(items_path.relative_to(root)),cohort_sha256=_hash(items),
        weights_dir=str(inside(root,manifest['weights_dir']).relative_to(root)),model_snapshot=snapshot_files(weights),
        model_identity=dict(model=gold['model'],model_revision=gold['model_revision'],
                            tokenizer_revision=gold['tokenizer_revision'],base_sha256=base),
        prompt_contract={'chat_template_sha256':gold['source_config']['chat_template_sha256']},
        decoding={**decoding,'use_cache':True},scorer='gsm8k-flexible-v3',seed=42,
        variants=variants,variant_order=list(ARMS),adapter_files=adapter_files,execution_kind='real-cuda',
        training_plans=manifest['training_plans'],
        batch_size=1,unit_size=unit_size,qualification='direct PEFT only; Stage 0 numerical qualification not established',
        implementation={n:file_hash(Path(__file__).with_name(n)) for n in ['sampling_stage0_evaluation.py','sampling_stage0_direct_backend.py','stage0_evaluation_backend.py','stage0_evaluation_records.py','scoring.py']})
    path=root/'plans'/name/'sampling-stage0.evaluation.json'
    _save_frozen(path,config); _save_frozen(root/config['run_path']/'config.json',config)
    _save_frozen(root/config['run_path']/'meta/run_manifest.json',dict(config_sha256=_hash(config),
        scope='three original sampling-SFT endpoint adapters, singleton greedy, no historical generation'))
    if not (root/config['run_path']/'meta/status.json').exists():
        _write_state(root/config['run_path'],'prepared',0,450)
    return path


def run(config_path, output_root, *, dependencies=None, max_new_units=None):
    from pilot_eval.sampling_stage0_direct_backend import DirectDependencies
    root=Path(output_root).resolve(); path=inside(root,config_path); config=json.loads(path.read_text())
    for relative,digest in config['parents'].items():
        if file_hash(inside(root,relative))!=digest:raise ValueError('Source evidence changed.')
    for name,digest in config['implementation'].items():
        if file_hash(Path(__file__).with_name(name))!=digest:raise ValueError('Implementation changed.')
    if 'model_snapshot' in config and snapshot_files(inside(root,config['weights_dir']))!=config['model_snapshot']:
        raise ValueError('Model snapshot inputs changed.')
    items=json.loads(inside(root,config['cohort_path']).read_text())
    if _hash(items)!=config['cohort_sha256']:raise ValueError('Cohort changed.')
    if 'training_plans' in config:
        from pilot_eval.sampling_sft_training import verify as verify_training
        for arm in ARMS:verify_training(inside(root,config['training_plans'][arm]),root)
    directory=inside(root,config['run_path']); deps=dependencies or DirectDependencies()
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        runtime=deps.runtime()
        if runtime.get('execution_kind')!='real-cuda' or 'A100' not in runtime.get('device',''):
            raise ValueError('This named direct-PEFT plan requires an A100 CUDA runtime.')
        runtime_path=directory/'meta/runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text())!=runtime:
            raise ValueError('Resume runtime differs from saved runtime.')
        deps.preflight(config,root)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(config_sha256=_hash(config),runtime=runtime))
        started=time.monotonic();engine=None
        try:
            engine=deps.engine(config,root)
            checks=engine.loaded_adapter_checks
            if (len(checks)!=3 or {c['variant'] for c in checks}!=set(ARMS)
                    or any(c.get('tensor_count')!=392 or c.get('exact_payload_match') is not True for c in checks)):
                raise ValueError('In-memory adapter payload verification failed.')
            _save_frozen(directory/'meta/loaded-adapters.json',checks)
            engine.prepare_inputs(items)
            smoke=engine.smoke(items[:2])
            if len(smoke)!=6 or any(r.get('exact_repeat') is not True for r in smoke):
                raise ValueError('Direct-PEFT finite repeat smoke failed.')
            _save_frozen(directory/'meta/direct-smoke.json',dict(runtime=runtime,observations=smoke,
                scope='Two prompts per arm; finite FP32 repeated direct PEFT logits, not merged-loader or batching qualification.'))
            return collect(config,items,directory,runtime,engine,max_new_units=max_new_units)
        except BaseException as exc:
            completed=sum(len(rows) for unit,rows in _units(config,items)
                          if (directory/'units'/unit['key']/'complete.json').exists())
            _write_state(directory,'failed',completed,
                         len(items)*len(ARMS),error=str(exc))
            raise
        finally:
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,
                usd_cost='unknown; provider billing not observed',rate_usd_per_hour=None))
            if engine is not None:engine.close()


def verify(config_path, output_root):
    root=Path(output_root).resolve(); config=json.loads(inside(root,config_path).read_text())
    for relative,digest in config['parents'].items():
        if file_hash(inside(root,relative))!=digest:raise ValueError('Source evidence changed.')
    for name,digest in config['implementation'].items():
        if file_hash(Path(__file__).with_name(name))!=digest:raise ValueError('Implementation changed.')
    if 'model_snapshot' in config and snapshot_files(inside(root,config['weights_dir']))!=config['model_snapshot']:
        raise ValueError('Model snapshot inputs changed.')
    items=json.loads(inside(root,config['cohort_path']).read_text())
    if _hash(items)!=config['cohort_sha256']:raise ValueError('Cohort changed.')
    if 'training_plans' in config:
        from pilot_eval.sampling_sft_training import verify as verify_training
        for arm in ARMS:verify_training(inside(root,config['training_plans'][arm]),root)
    directory=inside(root,config['run_path'])
    if (directory/'meta/integrity-failure.json').exists():raise ValueError('Prior integrity failure.')
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('Run config changed.')
    runtime=directory/'meta/runtime.json'; inputs=directory/'inputs/tokenized-prompts.json'
    token_rows=json.loads(inputs.read_text())
    if [r['id'] for r in token_rows]!=[r['id'] for r in items]:raise ValueError('Tokenized cohort changed.')
    for token,item in zip(token_rows,items):
        if token['prompt_sha256']!=_hash(item['prompt']) or token['attention_mask']!=[1]*len(token['input_ids']):
            raise ValueError('Tokenized prompt changed.')
    rh=file_hash(runtime); ih=file_hash(inputs); all_rows=[]
    for unit,selected in _units(config,items):
        folder=directory/'units'/unit['key']; payload=folder/'responses.jsonl'
        seal=json.loads((folder/'complete.json').read_text())
        integrity_path=folder/'integrity.json'
        if seal!=dict(config_sha256=_hash(config),runtime_sha256=rh,input_sha256=ih,responses_sha256=file_hash(payload),integrity_sha256=file_hash(integrity_path)):
            raise ValueError('Unit evidence changed.')
        proof=json.loads(integrity_path.read_text())
        if any(not all(proof[stage].get(k) is True for k in ['base_unchanged','adapters_unchanged']) for stage in ['before','after']):
            raise ValueError('Unit integrity evidence changed.')
        saved=[json.loads(line) for line in payload.read_text().splitlines()]
        outputs=[{key:row[key] for key in OUTPUT_FIELDS} for row in saved]
        if saved!=records(config,unit,selected,outputs,runtime_sha256=rh,input_sha256=ih):
            raise ValueError('Unit scoring changed.')
        all_rows.extend(saved)
    marker=json.loads((directory/'complete.json').read_text()); response=directory/'results/responses.jsonl'; result=directory/'results/results.json'
    if marker!=_completion(config,directory):
        raise ValueError('Completed result changed.')
    if config.get('execution_kind')=='real-cuda':
        actual_runtime=json.loads(runtime.read_text())
        smoke=json.loads((directory/'meta/direct-smoke.json').read_text())
        observations=smoke['observations']
        expected={(arm,row['id']) for arm in ARMS for row in items[:2]}
        if (smoke['runtime']!=actual_runtime or len(observations)!=len(expected)
                or {(r.get('variant'),r.get('id')) for r in observations}!=expected
                or any(r.get('exact_repeat') is not True for r in observations)):
            raise ValueError('Direct smoke evidence changed.')
        checks=json.loads((directory/'meta/loaded-adapters.json').read_text())
        if (len(checks)!=3 or {c['variant'] for c in checks}!=set(ARMS)
                or any(c.get('tensor_count')!=392 or c.get('exact_payload_match') is not True for c in checks)):
            raise ValueError('Loaded adapter evidence changed.')
        integrity=json.loads((directory/'meta/final-integrity.json').read_text())
        if not all(integrity.get(k) is True for k in ['base_unchanged','adapters_unchanged']):
            raise ValueError('Final integrity evidence changed.')
    if [json.loads(line) for line in response.read_text().splitlines()]!=all_rows:raise ValueError('Aggregated responses changed.')
    return json.loads(result.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',required=True);p.add_argument('--output-root',required=True);p.add_argument('--name',required=True);p.add_argument('--unit-size',type=int,default=8)
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--output-root',required=True);p.add_argument('--max-new-units',type=int)
    p=sub.add_parser('verify');p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    args=parser.parse_args()
    result=prepare(args.manifest,args.output_root,args.name,unit_size=args.unit_size) if args.command=='prepare' else (run(args.config,args.output_root,max_new_units=args.max_new_units) if args.command=='run' else verify(args.config,args.output_root))
    print(result)


if __name__=='__main__':main()
