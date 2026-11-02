"""Batch-versus-singleton agreement for the three sampling-SFT endpoints.

The same check the original Stage 0 batch16 path passed before production
(ADR 0014): on the fixed first 16 cohort questions, for the full 16 and tail 6
batch shapes, (a) teacher-forced FP32 answer-position logits and the cached
decode step against singleton inference, within the unchanged 1e-3 gate, and
(b) left-padded batched greedy replay against singleton greedy references:
exact token IDs and stop reason, or a measured near tie. The singleton
references are generated here because none exist for these adapters. Every
arm is replayed (the original replayed V1/V5 only). An arm that fails is
recorded, not raised: its behaviour run then generates one question at a time.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.stage0_batching import BATCH_GATE, numeric_ok
from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval import sampling_stage0_evaluation as evaluation

ARMS = evaluation.ARMS
SIZES = (16, 6) # 150 = 9 x 16 + 6
IMPLEMENTATION = ['sampling_stage0_agreement.py','sampling_stage0_evaluation.py','stage0_batching.py',
                  'stage0_batching_backend.py','stage0_evaluation_backend.py']
SOURCE_FIELDS = ['parents','cohort_path','cohort_sha256','weights_dir','model_snapshot','model_identity',
                 'prompt_contract','decoding','variants','variant_order','adapter_files','training_plans']


def units(config):
    ids=config['sequence_ids'];result=[]
    for arm in ARMS:
        result += [dict(key=f'reference-{arm}-{i:02d}',kind='reference',variant=arm,ids=[q]) for i,q in enumerate(ids)]
    for arm in ARMS:
        for size in SIZES:
            result.append(dict(key=f'numeric-b{size}-{arm}',kind='numeric',variant=arm,ids=ids[:size]))
            result.append(dict(key=f'greedy-b{size}-{arm}',kind='greedy',variant=arm,ids=ids[:size]))
    return result


def prepare(manifest_path, output_root, name, *, gpu='L4'):
    root=Path(output_root).resolve();safe_name(name)
    source,items=evaluation.sources(manifest_path,root)
    config=dict(protocol='sampling-stage0-batch-agreement-v1',name=name,
        run_path=f'runs/stage0/{name}/batch-agreement',**{k:source[k] for k in SOURCE_FIELDS},
        sequence_ids=[r['id'] for r in items[:16]],sizes=list(SIZES),batch_gate=BATCH_GATE,seed=42,
        execution_kind='real-cuda',gpu=gpu,implementation={n:file_hash(Path(__file__).with_name(n)) for n in IMPLEMENTATION},
        scope='Fixed first 16 questions; all three arms; numeric logits and cached decode; greedy replay at 16 and tail 6',
        decision='automatic: an arm batches only if every numeric and greedy unit passes; otherwise singleton')
    config['units']=units(config)
    path=root/'plans'/name/'sampling-stage0.agreement.json';directory=inside(root,config['run_path'])
    _save_frozen(path,config);_save_frozen(directory/'config.json',config)
    if not (directory/'meta/status.json').exists():_write_state(directory,'prepared',0,len(config['units']))
    return path


def _load(path,root):
    root=Path(root).resolve();config=json.loads(inside(root,path).read_text())
    if config.get('protocol')!='sampling-stage0-batch-agreement-v1' or config['units']!=units(config):
        raise ValueError('Unexpected agreement plan.')
    items=evaluation.check_sources(config,root)
    if [r['id'] for r in items[:16]]!=config['sequence_ids'] or config['batch_gate']!=BATCH_GATE:
        raise ValueError('Agreement questions or numerical gate changed.')
    directory=inside(root,config['run_path'])
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('Agreement run config changed.')
    return root,config,items,directory


def _sealed(config,directory,unit):
    folder=directory/'units'/unit['key']
    if not (folder/'complete.json').exists():return None
    if json.loads((folder/'complete.json').read_text())!=dict(config_sha256=_hash(config),unit=unit,
            runtime_sha256=file_hash(directory/'meta/runtime.json'),sha256=file_hash(folder/'result.json')):
        raise ValueError(f'Changed agreement unit: {unit["key"]}')
    return json.loads((folder/'result.json').read_text())


def token_rows(engine,rows):
    """Teacher-forced gold answers exactly as the original qualification built them."""
    eos=engine.tokenizer.eos_token_id;result=[]
    for row in rows:
        prompt=engine.inputs[row['id']];full=prompt+engine.tokenizer.encode(row['gold'],add_special_tokens=False)+[eos]
        result.append(dict(id=row['id'],input_ids=full,attention_mask=[1]*len(full),prompt_tokens=len(prompt),
                           answer_positions=list(range(len(prompt)-1,len(full)-1))))
    return result


def replay(config,unit,rows,responses,references,engine):
    eos=config['decoding']['eos_token_id'];checks=[]
    for j,(row,response) in enumerate(zip(rows,responses)):
        reference=references[row['id']]['generated_token_ids'];actual=response['generated_token_ids']
        expected_stop='eos' if reference and reference[-1]==eos else 'cap'
        actual_stop='eos' if actual and actual[-1]==eos else 'cap'
        check=dict(id=row['id'],exact=actual==reference and response['stop_reason']==expected_stop,near_tie=False,
                   stop_matches=response['stop_reason']==expected_stop,stop_consistent=response['stop_reason']==actual_stop,
                   singleton_stop_reason=expected_stop,batch_stop_reason=response['stop_reason'])
        if actual!=reference:check.update(engine.divergence(rows,unit['variant'],j,reference,responses))
        if not check['stop_consistent']:check.update(exact=False,near_tie=False)
        checks.append(check)
    return dict(responses=[dict(**r,id=row['id']) for row,r in zip(rows,responses)],checks=checks,
                acceptable=all(c['exact'] or c['near_tie'] for c in checks))


def assess(config,values):
    by_key={u['key']:v for u,v in zip(config['units'],values)};arms={}
    for arm in ARMS:
        numeric=[by_key[f'numeric-b{s}-{arm}'] for s in SIZES];greedy=[by_key[f'greedy-b{s}-{arm}'] for s in SIZES]
        checks=[c for v in greedy for c in v['checks']]
        arms[arm]=dict(numeric_passed=all(numeric_ok(v,config['batch_gate']) for v in numeric),
            greedy_passed=all(v['acceptable'] for v in greedy),
            max_absolute_error=max(v['max_absolute_error'] for v in numeric),
            exact_answers=sum(c['exact'] for c in checks),answers=len(checks),
            near_ties=sum(not c['exact'] and c['near_tie'] for c in checks))
        arms[arm]['batching_qualified']=arms[arm]['numeric_passed'] and arms[arm]['greedy_passed']
    return dict(arms=arms,batch_sizes={a:16 if arms[a]['batching_qualified'] else 1 for a in ARMS},
        batch_gate=config['batch_gate'],sizes=config['sizes'],
        decision='automatic from sealed units; batching where both checks passed, otherwise singleton')


def run(path,output_root,*,dependencies=None):
    root,config,items,directory=_load(path,output_root)
    from pilot_eval.stage0_batching_backend import BatchDependencies
    with run_lock(directory):
        evaluation.no_prior_integrity_failure(directory)
        if (directory/'complete.json').exists():return verify(path,root)
        deps=dependencies or BatchDependencies();runtime=deps.runtime()
        if runtime.get('execution_kind')!='real-cuda' or config['gpu'] not in runtime.get('device',''):
            raise ValueError(f"Batch agreement requires a {config['gpu']} CUDA runtime.")
        env=evaluation.environment(runtime);runtime_path=directory/'meta/runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text())!=env:
            raise ValueError('Resume runtime differs from the saved runtime.')
        _save_frozen(runtime_path,env)
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(config_sha256=_hash(config),runtime=runtime))
        values=[_sealed(config,directory,u) for u in config['units']]
        started=time.monotonic();engine=None
        try:
            engine=deps.engine(dict(variants=config['variants'],variant_order=list(ARMS),weights_dir=config['weights_dir'],
                model_identity=config['model_identity'],seed=config['seed'],prompt_contract=config['prompt_contract'],
                decoding=config['decoding'],execution_kind='real-cuda',batch_gate=config['batch_gate']),root)
            checks=engine.loaded_adapter_checks
            if ({c['variant'] for c in checks}!=set(ARMS) or len(checks)!=3
                    or any(c.get('tensor_count')!=392 or c.get('exact_payload_match') is not True for c in checks)):
                raise ValueError('In-memory adapter payload verification failed.')
            _save_frozen(directory/'meta/loaded-adapters.json',checks)
            by_id={r['id']:r for r in items};selected=[by_id[i] for i in config['sequence_ids']]
            engine.prepare_inputs(selected)
            tokens=token_rows(engine,selected);_save_frozen(directory/'inputs/token-rows.json',tokens)
            for i,unit in enumerate(config['units']):
                if values[i] is not None:continue
                rows=[by_id[q] for q in unit['ids']];begin=time.monotonic()
                print(f'Sampling batch agreement {i+1}/{len(values)}: {unit["key"]}',flush=True)
                if unit['kind']=='reference':
                    response=engine.generate(rows,unit['variant'],config['seed'])
                    if len(response)!=1:raise ValueError('Wrong singleton response count.')
                    value=dict(id=unit['ids'][0],**response[0])
                elif unit['kind']=='numeric':
                    value=engine.compare(rows,unit['variant'],tokens);numeric_ok(value,config['batch_gate'])
                else:
                    references={}
                    for j,u in enumerate(config['units']):
                        if u['kind']=='reference' and u['variant']==unit['variant']:references[u['ids'][0]]=values[j]
                    responses=engine.generate(rows,unit['variant'],config['seed'])
                    if len(responses)!=len(rows):raise ValueError('Wrong batched response count.')
                    value=replay(config,unit,rows,responses,references,engine)
                value['seconds']=time.monotonic()-begin
                evaluation.require_intact(directory,engine.check_integrity(),message='Model integrity failed during batch agreement.')
                folder=directory/'units'/unit['key'];_write_json(folder/'result.json',value)
                _write_json(folder/'complete.json',dict(config_sha256=_hash(config),unit=unit,
                    runtime_sha256=file_hash(runtime_path),sha256=file_hash(folder/'result.json')))
                values[i]=value
                _write_state(directory,'running',sum(v is not None for v in values),len(values))
            proof=evaluation.require_intact(directory,engine.check_integrity(full=True),message='Final model integrity failed.')
            _write_json(directory/'meta/final-integrity.json',proof)
            result={**assess(config,values),'environment':env}
            _write_json(directory/'results/agreement.json',result)
            files=['config.json','meta/runtime.json','meta/loaded-adapters.json','meta/final-integrity.json',
                   'inputs/token-rows.json','results/agreement.json']+[f'units/{u["key"]}/complete.json' for u in config['units']]
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={f:file_hash(directory/f) for f in files}))
            _write_state(directory,'complete',len(values),len(values))
            return result
        except BaseException as exc:
            _write_state(directory,'failed',sum(v is not None for v in values),len(values),error=str(exc))
            raise
        finally:
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started))
            if engine is not None:engine.close()


def verify(path,output_root):
    root,config,items,directory=_load(path,output_root);evaluation.no_prior_integrity_failure(directory)
    seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Agreement completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(directory/name)!=digest:raise ValueError(f'Agreement evidence changed: {name}')
    values=[_sealed(config,directory,u) for u in config['units']]
    if any(v is None for v in values):raise ValueError('Agreement evidence is incomplete.')
    result=json.loads((directory/'results/agreement.json').read_text())
    if {k:v for k,v in result.items() if k!='environment'}!=assess(config,values):
        raise ValueError('Agreement summary differs from sealed units.')
    if result['environment']!=json.loads((directory/'meta/runtime.json').read_text()):
        raise ValueError('Agreement environment evidence changed.')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--name',required=True);p.add_argument('--gpu',default='L4')
    for command in ['run','verify']:
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=str(prepare(args.manifest,args.output_root,args.name,gpu=args.gpu))
    elif args.command=='run':result=run(args.config,args.output_root)
    else:result=verify(args.config,args.output_root)
    print(json.dumps(result,indent=2,sort_keys=True,default=str))


if __name__=='__main__':main()
