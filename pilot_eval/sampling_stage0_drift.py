"""Fixed-context KL and effective writes for the three sampling-SFT endpoints.

The original Stage 0 instrument, unchanged: the same 150 GSM8K + 150 FineWeb
sequences (pinned by file hash), singleton FP32 decoder, FP64 output-head KL
(variant || untuned) in 32-position chunks, direct LoRA-branch write hooks and
the same per-unit checks and summaries (stage0_drift_backend / _math). Controls
are the untuned model and the shared Gold checkpoint-0 every arm started from.
Only the plan and run bookkeeping are new: no Colab budget or session binding.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import read_artifact, safe_name
from pilot_eval.stage0_drift import _agrees, _factor_norms
from pilot_eval.stage0_drift_math import check, positions, summarize
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval import sampling_stage0_evaluation as evaluation

ARMS = evaluation.ARMS
ZERO = 'k-zero'
CONTROLS = ['untuned', ZERO]
# The exact fixed-context payload the original Stage 0 drift run measured.
STAGE0_ITEMS_FILE_SHA256 = '5e617d5434306a7b995351746b75fb115380c55e0e88706d5aa74f5d8a2d57d8'
IMPLEMENTATION = ['sampling_stage0_drift.py','sampling_stage0_evaluation.py','stage0_drift.py','stage0_drift_math.py',
                  'stage0_drift_backend.py','stage0_evaluation_backend.py','token_measurement_math.py','activation_profile.py']
UNIT_FILES = {'arrays.npz','result.json','attempt.json'}


def prepare(behaviour_plan, inputs_path, output_root, name, *, items_file_sha256=STAGE0_ITEMS_FILE_SHA256,
            batches_per_unit=8, context_chunk=32, workspace_bytes=536870912):
    root=Path(output_root).resolve();safe_name(name)
    plan=inside(root,behaviour_plan);behaviour=json.loads(plan.read_text())
    if behaviour.get('protocol')!='sampling-stage0-evaluation-v2':raise ValueError('Expected the sampling behaviour plan.')
    evaluation.check_sources(behaviour,root)
    inputs=inside(root,inputs_path);prepared=json.loads(inputs.read_text())
    for k in ['model','model_revision','tokenizer_revision']:
        if prepared.get(k)!=behaviour['model_identity'].get(k):raise ValueError('Fixed-context model/tokenizer identity changed.')
    if prepared.get('dtype')!='float32':raise ValueError('Fixed contexts were prepared for a different precision.')
    items=inside(root,prepared['items_path'])
    if file_hash(items)!=items_file_sha256:raise ValueError('Not the original Stage 0 fixed-context payload.')
    rows=read_artifact(root,prepared['items_path'])
    if (_hash(rows)!=prepared['items_sha256'] or len(rows)!=300 or len({r['id'] for r in rows})!=300
            or {c:sum(r['corpus']==c for r in rows) for c in ['gsm8k','fineweb']}!={'gsm8k':150,'fineweb':150}):
        raise ValueError('Expected the original 150 GSM8K and 150 FineWeb measurement sequences.')
    positions(rows)
    for value in [batches_per_unit,context_chunk,workspace_bytes]:
        if type(value) is not int or value<1:raise ValueError('Positive integer save group/chunk/workspace required.')
    zero=behaviour['zero_adapter']
    variants={**behaviour['variants'],ZERO:dict(adapter_path=zero['path'],source_arm='gold-checkpoint-0',operation='zero-control')}
    adapter_files={};factor_norm_sums={};modules=None
    for variant,data in variants.items():
        folder=inside(root,data['adapter_path']);norms=_factor_norms(folder)
        if modules is not None and modules!=list(norms):raise ValueError('Adapter module mapping differs.')
        modules=list(norms);factor_norm_sums[variant]=float(sum(norms.values()))
        adapter_files[variant]={n:file_hash(folder/n) for n in ['adapter_config.json','adapter_model.safetensors','complete.json']}
    if adapter_files[ZERO]['adapter_model.safetensors']!=zero['adapter_sha256']:raise ValueError('Zero control changed.')
    factor_norm_sums['untuned']=0.
    diagnostic_ids=[r['id'] for c in ['gsm8k','fineweb'] for r in [r for r in rows if r['corpus']==c][:2]]
    units=[dict(key='control-'+v,variant=v,ids=diagnostic_ids,diagnostic=True) for v in CONTROLS]
    for start in range(0,len(rows),batches_per_unit):
        for arm in ARMS:
            units.append(dict(key=f'batch-{start//batches_per_unit:04d}-{arm}',variant=arm,
                ids=[r['id'] for r in rows[start:start+batches_per_unit]],diagnostic=start==0))
    config=dict(schema_version=1,protocol='sampling-stage0-fixed-context-v1',name=name,
        run_path=f'runs/stage0/{name}/fixed-context',behaviour_plan=str(plan.relative_to(root)),behaviour_sha256=file_hash(plan),
        parents=behaviour['parents'],training_plans=behaviour['training_plans'],cohort_path=behaviour['cohort_path'],
        cohort_sha256=behaviour['cohort_sha256'],model_snapshot=behaviour['model_snapshot'],
        implementation={n:file_hash(Path(__file__).with_name(n)) for n in IMPLEMENTATION},
        inputs_path=str(inputs.relative_to(root)),inputs_sha256=file_hash(inputs),items_path=prepared['items_path'],
        items_file_sha256=items_file_sha256,items_sha256=prepared['items_sha256'],item_ids=[r['id'] for r in rows],
        execution_kind='real-cuda',gpu=behaviour['gpu'],weights_dir=behaviour['weights_dir'],
        model_identity=behaviour['model_identity'],decoding=behaviour['decoding'],prompt_contract=behaviour['prompt_contract'],
        variants=variants,variant_order=[*ARMS,ZERO],adapter_files={a:behaviour['adapter_files'][a] for a in ARMS},
        drift_adapter_files=adapter_files,controls=CONTROLS,modules=modules,factor_norm_sums=factor_norm_sums,
        batch_size=1,batches_per_unit=batches_per_unit,context_chunk=context_chunk,workspace_bytes=workspace_bytes,
        units=units,seed=42,output_head_precision='float64',decoder_precision='float32',
        kl_direction='variant||untuned',runtime=None,execution_sha256=file_hash(plan),
        historical_values_used=False,
        comparison='Original Stage 0 measured V0a/V0b with this instrument and payload on a Colab L4; joins disclose any environment difference.')
    path=root/'plans'/name/'sampling-stage0.drift.json';directory=inside(root,config['run_path'])
    _save_frozen(path,config);_save_frozen(directory/'config.json',config)
    if not (directory/'meta/status.json').exists():_write_state(directory,'prepared',0,len(units))
    return path


def _load(path,root):
    root=Path(root).resolve();path=inside(root,path);config=json.loads(path.read_text())
    if config.get('protocol')!='sampling-stage0-fixed-context-v1':raise ValueError('Unexpected drift plan.')
    evaluation.check_sources({**config,'adapter_files':config['adapter_files']},root)
    if file_hash(inside(root,config['behaviour_plan']))!=config['behaviour_sha256']:raise ValueError('Behaviour plan changed.')
    if (file_hash(inside(root,config['inputs_path']))!=config['inputs_sha256']
            or file_hash(inside(root,config['items_path']))!=config['items_file_sha256']):
        raise ValueError('Fixed-context inputs changed.')
    for variant,files in config['drift_adapter_files'].items():
        folder=inside(root,config['variants'][variant]['adapter_path'])
        if any(file_hash(folder/n)!=h for n,h in files.items()):raise ValueError(f'Adapter payload changed: {variant}')
    directory=inside(root,config['run_path'])
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('Drift run config changed.')
    return root,config,directory,read_artifact(root,config['items_path'])


def _sealed(config,directory,unit,by_id):
    folder=directory/'units'/unit['key'];marker=folder/'complete.json'
    if not marker.exists():return None
    seal=json.loads(marker.read_text())
    if (seal.get('identity')!=dict(config_sha256=_hash(config),runtime_sha256=file_hash(directory/'meta/runtime.json'),key=unit['key'])
            or set(seal.get('files',{}))!=UNIT_FILES or any(file_hash(folder/p)!=d for p,d in seal['files'].items())):
        raise ValueError(f'Drift unit evidence changed: {unit["key"]}')
    attempt=json.loads((folder/'attempt.json').read_text())['integrity']
    if attempt.get('base_unchanged') is not True or attempt.get('adapters_unchanged') is not True:
        raise ValueError('Drift unit integrity evidence changed.')
    value=json.loads((folder/'result.json').read_text())
    with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:value['arrays']={k:arrays[k] for k in arrays.files}
    return check(value,[by_id[i] for i in unit['ids']],config,unit['variant'],unit['diagnostic'])


def _summary(config,directory,rows,saved):
    runtime=json.loads((directory/'meta/runtime.json').read_text())
    result=summarize({**config,'runtime':runtime},saved,rows)
    result['module_factor_norms_path']='inputs/module-factor-norms.json'
    return result


def run(path,output_root,*,dependencies=None,max_new_units=None):
    root,config,directory,rows=_load(path,output_root)
    from pilot_eval.stage0_drift_backend import DriftDependencies
    by_id={r['id']:r for r in rows}
    with run_lock(directory):
        if (directory/'complete.json').exists():return verify(path,root)
        deps=dependencies or DriftDependencies();runtime=deps.runtime()
        if runtime.get('execution_kind')!='real-cuda' or config['gpu'] not in runtime.get('device',''):
            raise ValueError(f"Fixed-context collection requires a {config['gpu']} CUDA runtime.")
        env=evaluation.environment(runtime);runtime_path=directory/'meta/runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text())!=env:
            raise ValueError('Resume runtime differs from the saved runtime.')
        _save_frozen(runtime_path,env)
        saved=[_sealed(config,directory,u,by_id) for u in config['units']]
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(config_sha256=_hash(config),runtime=runtime))
        started=time.monotonic();engine=None;new=0
        try:
            engine=deps.engine(config,root)
            checks=getattr(engine,'loaded_adapter_checks',[])
            if ({c['variant'] for c in checks}!=set(config['variants']) or len(checks)!=len(config['variants'])
                    or any(c.get('tensor_count')!=392 or c.get('exact_payload_match') is not True for c in checks)):
                raise ValueError('Every drift adapter needs exact in-memory checks of all 392 factors.')
            _save_frozen(directory/'meta/loaded-adapters.json',dict(execution_kind='real-cuda',checks=checks))
            _save_frozen(directory/'inputs/module-factor-norms.json',
                {v:_factor_norms(inside(root,d['adapter_path'])) for v,d in config['variants'].items()})
            for i,unit in enumerate(config['units']):
                if saved[i] is not None:continue
                if max_new_units is not None and new>=max_new_units:break
                begin=time.monotonic();selected=[by_id[j] for j in unit['ids']]
                result=engine.measure(selected,unit['variant'],diagnostic=unit['diagnostic'])
                evidence=attempt/'observations'/unit['key'];evidence.mkdir(parents=True,exist_ok=True)
                with (evidence/'arrays.npz').open('wb') as stream:np.savez_compressed(stream,**result['arrays'])
                _write_json(evidence/'result.json',{k:v for k,v in result.items() if k!='arrays'})
                result=check(result,selected,config,unit['variant'],unit['diagnostic'])
                integrity=engine.check_integrity()
                if integrity.get('base_unchanged') is not True or integrity.get('adapters_unchanged') is not True:
                    raise ValueError('Drift model-integrity guard failed.')
                folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True)
                with (folder/'arrays.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**result['arrays'])
                (folder/'arrays.npz.tmp').replace(folder/'arrays.npz')
                _write_json(folder/'result.json',{k:v for k,v in result.items() if k!='arrays'})
                _write_json(folder/'attempt.json',dict(attempt_id=attempt.name,integrity=integrity,
                    seconds=time.monotonic()-begin,code_git_commit=runtime.get('git_commit')))
                _write_json(folder/'complete.json',dict(identity=dict(config_sha256=_hash(config),
                    runtime_sha256=file_hash(runtime_path),key=unit['key']),
                    files={p:file_hash(folder/p) for p in sorted(UNIT_FILES)}))
                saved[i]=result;new+=1
                _write_state(directory,'running',sum(s is not None for s in saved),len(saved))
                print(f'Sampling fixed context: {sum(s is not None for s in saved)}/{len(saved)} sealed; {unit["key"]}',flush=True)
            integrity=engine.check_integrity(full=True)
            if integrity.get('base_unchanged') is not True or integrity.get('adapters_unchanged') is not True:
                raise ValueError('Full drift integrity failed; preserve this attempt.')
            if any(s is None for s in saved):
                _write_state(directory,'incomplete',sum(s is not None for s in saved),len(saved))
                return dict(state='incomplete',completed=sum(s is not None for s in saved),total=len(saved))
            _write_json(directory/'meta/final-integrity.json',integrity)
            result=_summary(config,directory,rows,saved)
            _jsonl(directory/'results/per-document.jsonl',result['per_document'])
            _write_json(directory/'results/results.json',result)
            files=['config.json','meta/runtime.json','meta/loaded-adapters.json','meta/final-integrity.json',
                   'inputs/module-factor-norms.json','results/results.json','results/per-document.jsonl']
            files+=[f'units/{u["key"]}/complete.json' for u in config['units']]
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={p:file_hash(directory/p) for p in files}))
            _write_state(directory,'complete',len(saved),len(saved))
            return result
        except BaseException as exc:
            _write_state(directory,'failed',sum(s is not None for s in saved),len(saved),error=str(exc))
            raise
        finally:
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,newly_sealed=new))
            if engine is not None:engine.close()


def verify(path,output_root):
    root,config,directory,rows=_load(path,output_root)
    seal=json.loads((directory/'complete.json').read_text())
    if seal.get('config_sha256')!=_hash(config):raise ValueError('Drift completion identity mismatch.')
    for p,digest in seal['files'].items():
        if file_hash(directory/p)!=digest:raise ValueError(f'Drift completion payload changed: {p}')
    final=json.loads((directory/'meta/final-integrity.json').read_text())
    if final.get('base_unchanged') is not True or final.get('adapters_unchanged') is not True:
        raise ValueError('Missing final model integrity.')
    by_id={r['id']:r for r in rows};saved=[_sealed(config,directory,u,by_id) for u in config['units']]
    if any(s is None for s in saved):raise ValueError('Drift collection is incomplete.')
    result=_summary(config,directory,rows,saved)
    records=[json.loads(s) for s in (directory/'results/per-document.jsonl').read_text().splitlines()]
    if (not _agrees(json.loads((directory/'results/results.json').read_text()),json.loads(json.dumps(result)))
            or not _agrees(records,json.loads(json.dumps(result['per_document'])))):
        raise ValueError('Drift aggregates disagree with sealed per-position measurements.')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--behaviour-plan',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--name',required=True)
    for command in ['run','verify']:
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=str(prepare(args.behaviour_plan,args.inputs,args.output_root,args.name))
    elif args.command=='run':result={k:v for k,v in run(args.config,args.output_root).items() if k!='per_document'}
    else:result={k:v for k,v in verify(args.config,args.output_root).items() if k!='per_document'}
    print(json.dumps(result,indent=2,sort_keys=True,default=str))


if __name__=='__main__':main()
