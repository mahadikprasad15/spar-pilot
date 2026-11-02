"""Base-model teacher-forced scoring of the Gold and k0/k2/k5 training answers.

The original Stage 0 E0.6a instrument, unchanged (stage0_distance_backend /
_math): one untuned FP32 forward per unique answer, full-vocabulary FP64
surprise, entropy and rank at every supervised answer token, prompt masked,
the supervised end-of-turn included. Answers are the exact token IDs each
adapter trained on (sampling training export) and the Gold SFT rows.

Summaries per dataset, on each arm's covered problems and on the problems all
three arms cover: token-weighted NLL, entropy, excess surprise (NLL minus
entropy; descriptive, not KL), top-1 fraction, answer-weighted NLL/entropy,
first32/middle/last32 windows, and paired k-minus-Gold differences on the
common problems, with 10,000 whole-problem bootstrap draws (seed 42).
"""
import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

import numpy as np

from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.stage0_distance_math import check_arrays, position_windows
from pilot_eval.stage0_evaluation import _jsonl
from pilot_eval.stage0_sources import inside
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen
from pilot_eval import sampling_stage0_evaluation as evaluation

ARMS = evaluation.ARMS
DATASETS = ('gold', *ARMS)
COMMON_PROBLEMS = 431 # accepted coverage: problems every arm covers
IMPLEMENTATION = ['sampling_stage0_dataset.py','sampling_stage0_evaluation.py','stage0_distance_backend.py',
                  'stage0_distance_math.py','stage0_rollout_backend.py']
BOOTSTRAP = dict(draws=10000,seed=42,percentiles=[2.5,97.5],unit='whole problem')


def answers_from_rows(datasets,eos):
    """Unique (prompt, target) answers and per-dataset memberships; masks and end-of-turn checked."""
    answers={};memberships={}
    for name,rows in datasets.items():
        memberships[name]=[];seen=set()
        for row in rows:
            boundary=row['prompt_tokens'];prompt=row['input_ids'][:boundary];targets=row['input_ids'][boundary:]
            if (row['labels']!=[-100]*boundary+targets or not prompt or not targets or targets[-1]!=eos
                    or eos in targets[:-1] or any(type(t) is not int or t<0 for t in prompt+targets)):
                raise ValueError(f'{name} answer mask/end-of-turn contract differs: {row["id"]}')
            if row['id'] in seen:raise ValueError(f'Duplicate problem in {name}: {row["id"]}')
            seen.add(row['id']);identifier=_hash([prompt,targets])
            answers.setdefault(identifier,dict(answer_id=identifier,input_ids=prompt+targets,prompt_tokens=len(prompt),
                target_ids=targets,positions=list(range(len(prompt)-1,len(prompt)+len(targets)-1))))
            memberships[name].append(dict(problem_id=row['id'],answer_id=identifier))
    gold={m['problem_id'] for m in memberships['gold']}
    if any(m['problem_id'] not in gold for a in ARMS for m in memberships[a]):raise ValueError('Arm problem outside the Gold set.')
    return list(answers.values()),memberships


def prepare(behaviour_plan,output_root,name,*,context_chunk=16,workspace_bytes=536870912,unit_size=32,load_rows=None):
    root=Path(output_root).resolve();safe_name(name)
    plan=inside(root,behaviour_plan);behaviour=json.loads(plan.read_text())
    if behaviour.get('protocol')!='sampling-stage0-evaluation-v2':raise ValueError('Expected the sampling behaviour plan.')
    evaluation.check_sources(behaviour,root)
    datasets,parents=(load_rows or training_rows)(root,behaviour)
    answers,memberships=answers_from_rows(datasets,behaviour['decoding']['eos_token_id'])
    common=sorted(set.intersection(*[{m['problem_id'] for m in memberships[a]} for a in ARMS]))
    if len(common)!=COMMON_PROBLEMS:raise ValueError(f'Expected {COMMON_PROBLEMS} commonly covered problems, found {len(common)}.')
    for value in [context_chunk,workspace_bytes,unit_size]:
        if type(value) is not int or value<1:raise ValueError('Positive integer chunk/workspace/unit size required.')
    units=[dict(key=f'answers-{i//unit_size:04d}',ids=[a['answer_id'] for a in answers[i:i+unit_size]]) for i in range(0,len(answers),unit_size)]
    config=dict(protocol='sampling-stage0-dataset-v1',name=name,run_path=f'runs/stage0/{name}/dataset',
        behaviour_plan=str(plan.relative_to(root)),behaviour_sha256=file_hash(plan),
        **{k:behaviour[k] for k in ['parents','training_plans','cohort_path','cohort_sha256','weights_dir','model_snapshot',
                                    'variants','adapter_files','model_identity','gpu']},
        data_parents=parents,implementation={n:file_hash(Path(__file__).with_name(n)) for n in IMPLEMENTATION},
        execution_kind='real-cuda',context_chunk=context_chunk,workspace_bytes=workspace_bytes,units=units,
        answers_sha256=_hash(answers),memberships_sha256=_hash(memberships),common_problem_ids=common,
        scored_eos_token_id=behaviour['decoding']['eos_token_id'],bootstrap=BOOTSTRAP,
        mask='stored prompt masked; supervised answer tokens through the single final end-of-turn',
        units_of_measure='nats per answer token',excess_surprise='NLL minus model entropy; descriptive, not KL',
        rank='1 + full-vocabulary logits strictly greater than the target; ties share rank',
        score_model='untuned audited FP32 base, the same weights every arm and Gold trained from')
    path=root/'plans'/name/'sampling-stage0.dataset.json';directory=inside(root,config['run_path'])
    _save_frozen(path,config);_save_frozen(directory/'config.json',config)
    _jsonl(directory/'inputs/answers.jsonl',answers);_save_frozen(directory/'inputs/memberships.json',memberships)
    if not (directory/'meta/status.json').exists():_write_state(directory,'prepared',0,len(units))
    return path


def training_rows(root,behaviour):
    """Gold SFT rows and each arm's exported training rows, through their own verifiers."""
    from pilot_eval import sampling_sft_training_data as data
    from pilot_eval.sft import load_sft
    from pilot_eval.sampling_sft_generation import read_lines
    configs=[json.loads(inside(root,p).read_text()) for p in behaviour['training_plans'].values()]
    if len({c['data_plan'] for c in configs})!=1 or len({c['gold_config'] for c in configs})!=1:
        raise ValueError('Arms disagree on training data or Gold config.')
    data_plan=configs[0]['data_plan'];data.verify(inside(root,data_plan),root)
    folder=inside(root,json.loads(inside(root,data_plan).read_text())['run_path'])
    _,gold,_=load_sft(inside(root,configs[0]['gold_config']),root)
    rows={'gold':gold,**{a:read_lines(folder/f'arms/{a}/training.jsonl') for a in ARMS}}
    parents={data_plan:file_hash(inside(root,data_plan)),configs[0]['gold_config']:file_hash(inside(root,configs[0]['gold_config']))}
    parents|={str((folder/f'arms/{a}/training.jsonl').relative_to(root)):file_hash(folder/f'arms/{a}/training.jsonl') for a in ARMS}
    return rows,parents


def _load(path,root):
    root=Path(root).resolve();config=json.loads(inside(root,path).read_text())
    if config.get('protocol')!='sampling-stage0-dataset-v1':raise ValueError('Unexpected dataset plan.')
    evaluation.check_sources(config,root)
    if file_hash(inside(root,config['behaviour_plan']))!=config['behaviour_sha256']:raise ValueError('Behaviour plan changed.')
    for relative,digest in config['data_parents'].items():
        if file_hash(inside(root,relative))!=digest:raise ValueError(f'Training answers changed: {relative}')
    directory=inside(root,config['run_path'])
    answers=[json.loads(l) for l in (directory/'inputs/answers.jsonl').read_text().splitlines()]
    memberships=json.loads((directory/'inputs/memberships.json').read_text())
    if (json.loads((directory/'config.json').read_text())!=config or _hash(answers)!=config['answers_sha256']
            or _hash(memberships)!=config['memberships_sha256']):
        raise ValueError('Frozen dataset answers or memberships changed.')
    return root,config,directory,answers,memberships


def _sealed(config,directory,unit,by_id):
    folder=directory/'units'/unit['key']
    if not (folder/'complete.json').exists():return None
    if (json.loads((folder/'complete.json').read_text())!=dict(config_sha256=_hash(config),unit=unit,
            runtime_sha256=file_hash(directory/'meta/runtime.json'),sha256=file_hash(folder/'tokens.npz'),
            integrity_sha256=file_hash(folder/'integrity.json'))
            or json.loads((folder/'integrity.json').read_text()).get('base_unchanged') is not True):
        raise ValueError(f'Dataset unit changed: {unit["key"]}')
    result={}
    with np.load(folder/'tokens.npz',allow_pickle=False) as saved:
        for i,identifier in enumerate(unit['ids']):
            result[identifier]=check_arrays({k:saved[f'{k}_{i}'] for k in ['surprise','entropy','rank']},by_id[identifier])
    return result


def run(path,output_root,*,dependencies=None,max_new_units=None):
    root,config,directory,answers,memberships=_load(path,output_root)
    from pilot_eval.stage0_distance_backend import DistanceDependencies
    by_id={a['answer_id']:a for a in answers}
    with run_lock(directory):
        evaluation.no_prior_integrity_failure(directory)
        if (directory/'complete.json').exists():return verify(path,root)
        deps=dependencies or DistanceDependencies();runtime=deps.runtime()
        if runtime.get('execution_kind')!='real-cuda' or config['gpu'] not in runtime.get('device',''):
            raise ValueError(f"Dataset scoring requires a {config['gpu']} CUDA runtime.")
        env=evaluation.environment(runtime);runtime_path=directory/'meta/runtime.json'
        if runtime_path.exists() and json.loads(runtime_path.read_text())!=env:raise ValueError('Resume runtime differs from the saved runtime.')
        _save_frozen(runtime_path,env)
        saved=[_sealed(config,directory,u,by_id) for u in config['units']]
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(config_sha256=_hash(config),runtime=runtime))
        started=time.monotonic();engine=None;new=0
        try:
            engine=deps.engine(dict(weights_dir=config['weights_dir'],model_identity=config['model_identity'],
                context_chunk=config['context_chunk'],workspace_bytes=config['workspace_bytes'],execution_kind='real-cuda'),root)
            _save_frozen(directory/'meta/vocabulary.json',engine.vocabulary())
            for i,unit in enumerate(config['units']):
                if saved[i] is not None:continue
                if max_new_units is not None and new>=max_new_units:break
                arrays={};values={}
                for j,identifier in enumerate(unit['ids']):
                    scored=check_arrays(engine.score(by_id[identifier]),by_id[identifier]);values[identifier]=scored
                    arrays.update({f'{k}_{j}':v for k,v in scored.items()})
                integrity=evaluation.require_intact(directory,engine.check_integrity(),('base_unchanged',),
                    message='Base-model integrity failed during scoring.')
                folder=directory/'units'/unit['key'];folder.mkdir(parents=True,exist_ok=True)
                with (folder/'tokens.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
                (folder/'tokens.npz.tmp').replace(folder/'tokens.npz');_write_json(folder/'integrity.json',integrity)
                _write_json(folder/'complete.json',dict(config_sha256=_hash(config),unit=unit,runtime_sha256=file_hash(runtime_path),
                    sha256=file_hash(folder/'tokens.npz'),integrity_sha256=file_hash(folder/'integrity.json')))
                saved[i]=values;new+=1
                _write_state(directory,'running',sum(s is not None for s in saved),len(saved))
                print(f'Sampling dataset scoring: {sum(s is not None for s in saved)}/{len(saved)} units sealed',flush=True)
            integrity=evaluation.require_intact(directory,engine.check_integrity(full=True),('base_unchanged',),
                message='Full base-model integrity failed.')
            if any(s is None for s in saved):
                _write_state(directory,'incomplete',sum(s is not None for s in saved),len(saved))
                return dict(state='incomplete',completed=sum(s is not None for s in saved),total=len(saved))
            _write_json(directory/'meta/final-integrity.json',integrity)
            return _finish(config,directory,answers,memberships,saved)
        except BaseException as exc:
            _write_state(directory,'failed',sum(s is not None for s in saved),len(saved),error=str(exc))
            raise
        finally:
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-started,newly_sealed=new))
            if engine is not None and hasattr(engine,'close'):engine.close()


def _ratio_ci(totals,draws,numerator,denominator):
    return [float(x) for x in np.percentile(numerator(totals[draws].sum(1))/denominator(totals[draws].sum(1)),BOOTSTRAP['percentiles'])]


def summarize(config,answers,memberships,arrays):
    """Token-weighted dataset estimates; whole-problem bootstrap, same draws for paired differences."""
    def problem_sums(name,problems):
        rows={m['problem_id']:arrays[m['answer_id']] for m in memberships[name] if m['problem_id'] in problems}
        return {p:np.array([len(a['surprise']),a['surprise'].sum(),a['entropy'].sum(),(a['rank']==1).sum(),1.,
                            a['surprise'].mean(),a['entropy'].mean()]) for p,a in rows.items()},rows
    summaries=[];windows=[];paired=[];common=set(config['common_problem_ids'])
    for scope in ['covered','common']:
        for name in DATASETS:
            problems={m['problem_id'] for m in memberships[name]} if scope=='covered' else common
            sums,rows=problem_sums(name,problems);ids=sorted(sums);totals=np.array([sums[i] for i in ids]);t=totals.sum(0)
            draws=np.random.default_rng(BOOTSTRAP['seed']).integers(0,len(ids),size=(BOOTSTRAP['draws'],len(ids)))
            summaries.append(dict(dataset=name,scope=scope,n_problems=len(ids),n_tokens=int(t[0]),nll=float(t[1]/t[0]),
                entropy=float(t[2]/t[0]),excess_surprise=float((t[1]-t[2])/t[0]),top1_fraction=float(t[3]/t[0]),
                answer_weighted_nll=float(t[5]/t[4]),answer_weighted_entropy=float(t[6]/t[4]),
                answer_weighted_excess_surprise=float((t[5]-t[6])/t[4]),
                answer_weighted_nll_ci=_ratio_ci(totals,draws,lambda s:s[:,5],lambda s:s[:,4]),
                answer_weighted_entropy_ci=_ratio_ci(totals,draws,lambda s:s[:,6],lambda s:s[:,4]),
                answer_weighted_excess_surprise_ci=_ratio_ci(totals,draws,lambda s:s[:,5]-s[:,6],lambda s:s[:,4]),
                n_answers_under_64=sum(len(a['surprise'])<64 for a in rows.values()),
                **{f'{w}_tokens':sum(position_windows(len(a['surprise'])).count(w) for a in rows.values())
                   for w in ['first32','middle','last32']},
                median_answer_tokens=float(statistics.median(len(a['surprise']) for a in rows.values())),
                nll_ci=_ratio_ci(totals,draws,lambda s:s[:,1],lambda s:s[:,0]),
                excess_surprise_ci=_ratio_ci(totals,draws,lambda s:s[:,1]-s[:,2],lambda s:s[:,0])))
            for window in ['first32','middle','last32']:
                n=s=e=0.
                for a in rows.values():
                    mask=np.array([w==window for w in position_windows(len(a['surprise']))])
                    n+=mask.sum();s+=a['surprise'][mask].sum();e+=a['entropy'][mask].sum()
                windows.append(dict(dataset=name,scope=scope,window=window,n_tokens=int(n),
                    nll=float(s/n) if n else None,entropy=float(e/n) if n else None,
                    excess_surprise=float((s-e)/n) if n else None)) # zero-count windows stay visible
    ids=sorted(common);draws=np.random.default_rng(BOOTSTRAP['seed']).integers(0,len(ids),size=(BOOTSTRAP['draws'],len(ids)))
    gold=np.array([problem_sums('gold',common)[0][i] for i in ids])
    for arm in ARMS:
        k=np.array([problem_sums(arm,common)[0][i] for i in ids])
        for metric,num in [('nll',lambda s:s[...,1]),('entropy',lambda s:s[...,2]),('excess_surprise',lambda s:s[...,1]-s[...,2])]:
            point=num(k.sum(0))/k.sum(0)[0]-num(gold.sum(0))/gold.sum(0)[0]
            ks,gs=k[draws].sum(1),gold[draws].sum(1)
            ci=np.percentile(num(ks)/ks[:,0]-num(gs)/gs[:,0],BOOTSTRAP['percentiles'])
            paired.append(dict(arm=arm,minus='gold',metric=metric,n_problems=len(ids),difference=float(point),ci=[float(x) for x in ci]))
    return dict(schema_version=1,analysis_complete=True,real_gpu_measurement=config['execution_kind']=='real-cuda',
        summaries=summaries,position_windows=windows,paired_differences=paired,bootstrap=BOOTSTRAP,
        unique_answers=len(answers),membership_counts={k:len(v) for k,v in memberships.items()},
        limitations=['Training answers scored under the untuned base; describes the datasets, not the trained adapters.',
            'Arms cover different problem subsets; the common scope (431 problems) is the paired comparison.',
            'Intervals resample whole problems and are pointwise; excess surprise is descriptive, not KL.'])


def _finish(config,directory,answers,memberships,saved):
    arrays={k:v for unit in saved for k,v in unit.items()}
    result=summarize(config,answers,memberships,arrays)
    _write_json(directory/'results/summary.json',result)
    files=['config.json','inputs/answers.jsonl','inputs/memberships.json','meta/runtime.json','meta/vocabulary.json',
           'meta/final-integrity.json','results/summary.json']+[f'units/{u["key"]}/complete.json' for u in config['units']]
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={f:file_hash(directory/f) for f in files}))
    _write_state(directory,'complete',len(saved),len(saved))
    return result


def verify(path,output_root):
    root,config,directory,answers,memberships=_load(path,output_root);evaluation.no_prior_integrity_failure(directory)
    seal=json.loads((directory/'complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Dataset completion identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(directory/name)!=digest:raise ValueError(f'Dataset evidence changed: {name}')
    by_id={a['answer_id']:a for a in answers};saved=[_sealed(config,directory,u,by_id) for u in config['units']]
    if any(s is None for s in saved):raise ValueError('Dataset scoring is incomplete.')
    result=summarize(config,answers,memberships,{k:v for unit in saved for k,v in unit.items()})
    if json.loads((directory/'results/summary.json').read_text())!=json.loads(json.dumps(result)):
        raise ValueError('Dataset summary differs from sealed token scores.')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--behaviour-plan',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--name',required=True)
    for command in ['run','verify']:
        p=sub.add_parser(command);p.add_argument('--config',required=True);p.add_argument('--output-root',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':result=str(prepare(args.behaviour_plan,args.output_root,args.name))
    elif args.command=='run':result=run(args.config,args.output_root)
    else:result=verify(args.config,args.output_root)
    print(json.dumps(result,indent=2,sort_keys=True,default=str))


if __name__=='__main__':main()
