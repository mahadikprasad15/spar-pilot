"""Pilot 4 disposable GPU evidence and reviewed immutable execution plans."""
import datetime
import json
import math
from pathlib import Path

from pilot_eval.grpo_prepare import load_grpo_prepared, _within
from pilot_eval.run import _write_json, _write_state
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash, run_lock
from pilot_eval.workflow import _hash, _save_frozen


def _implementation():
    from pilot_eval import grpo_algorithm, grpo_gpu, scoring
    return {k:file_hash(Path(v.__file__)) for k,v in
            [('workflow',__import__(__name__,fromlist=[''])),('algorithm',grpo_algorithm),
             ('gpu',grpo_gpu),('scorer',scoring)]}


def _seal(directory, config):
    files={str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*')
           if p.is_file() and p.name not in ['complete.json','.lock'] and '/meta/' not in str(p)
           and '/checkpoints/progress.json' not in str(p) and '/logs/' not in str(p)}
    _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files=files))


def _verify(directory, config):
    marker=json.loads((directory/'complete.json').read_text())
    if marker['config_sha256']!=_hash(config):raise ValueError('preflight/freeze config identity mismatch')
    if not {'config.json','results/results.json','runtime.json'}.issubset(marker['files']):
        raise ValueError('incomplete preflight/freeze seal')
    for rel,digest in marker['files'].items():
        p=(directory/rel).resolve()
        if not p.is_relative_to(directory.resolve()) or file_hash(p)!=digest:
            raise ValueError('preflight/freeze evidence hash mismatch')
    return marker


def verify_controls(root,name):
    from pilot_eval import grpo_controls,grpo_algorithm,grpo_checks
    directory=root/'runs/diagnostics/pilot-4'/safe_name(name)
    config=json.loads((directory/'config.json').read_text())
    identities={'task_sha256':file_hash(grpo_controls.TASK_PATH),
                'algorithm_sha256':file_hash(Path(grpo_algorithm.__file__)),
                'controls_sha256':file_hash(Path(grpo_controls.__file__)),
                'checks_sha256':file_hash(Path(grpo_checks.__file__))}
    if any(config.get(k)!=v for k,v in identities.items()):raise ValueError('controls implementation changed; rerun under a new name')
    marker=json.loads((directory/'complete.json').read_text())
    if marker['config_sha256']!=_hash(config) or not {'config.json','results/results.json'}.issubset(marker['files']):raise ValueError('control config mismatch')
    for rel,digest in marker['files'].items():
        if file_hash(_within(root,directory/rel))!=digest:raise ValueError('control evidence mismatch')
    result=json.loads((directory/'results/results.json').read_text())
    if not result.get('passed') or not result.get('algorithm_checks',{}).get('passed'):
        raise ValueError('algorithm and two-seed learning controls must pass')
    if {(r['seed'],r['sign']) for r in result['runs']}!={(42,1),(42,-1),(43,1),(43,-1)} or any(
            not r['passed'] or r['signed_change']<.2 for r in result['runs']):
        raise ValueError('incomplete learning controls')
    return dict(passed=True,algorithm_checks=dict(passed=True),evidence_sha256=file_hash(directory/'complete.json'))


def _baseline(root,path,prepared):
    from pilot_eval.grpo_baseline import _summary
    path=_within(root,path);config=json.loads(path.read_text())
    if config['source_sha256']!=file_hash(prepared) or config['source_path']!=str(prepared.relative_to(root)):
        raise ValueError('sampling baseline belongs to another preparation')
    directory=_within(root,config['run_path'])
    if json.loads((directory/'config.json').read_text())!=config:raise ValueError('baseline config mismatch')
    records=[]
    for i,batch in enumerate(config['batches']):
        shard=directory/'batches'/f'{i:04d}.json';marker=json.loads(shard.with_suffix('.complete.json').read_text())
        if marker!={'sha256':file_hash(shard),'config_sha256':_hash(config),
                    'runtime_sha256':file_hash(directory/'meta/runtime.json')}:
            raise ValueError('baseline shard mismatch')
        saved=json.loads(shard.read_text())
        if [r['draw_id'] for r in saved]!=[f'{q}:draw:{d}' for q in batch['ids'] for d in range(8)]:
            raise ValueError('baseline draw identity mismatch')
        records.extend(saved)
    summary=_summary(records,config['settings'])
    if summary!=json.loads((directory/'results/results.json').read_text()):raise ValueError('baseline summary differs from verified draws')
    combined=''.join(json.dumps(r,sort_keys=True)+'\n' for r in records)
    if combined!=(directory/'results/responses.jsonl').read_text():raise ValueError('baseline combined draws mismatch')
    if summary['p99_censored']:raise ValueError('censored baseline p99 cannot justify final cap; collect a named extension')
    return config,summary,dict(config=str(path.relative_to(root)),config_sha256=file_hash(path),
                              runtime_sha256=file_hash(directory/'meta/runtime.json'),
                              responses_sha256=file_hash(directory/'results/responses.jsonl'))


def _settings(value, summary):
    fields={'scorer','loss','cap_rule','generation_groups','backward_groups','evaluation_batch_size',
            'gradient_checkpointing','diagnostic_seed','memory_margin_gib','top_p','top_k'}
    if set(value)!=fields or any(v is None for v in value.values()):raise ValueError('resolve every preflight setting explicitly: '+', '.join(sorted(fields)))
    if value['scorer']!='gsm8k-flexible-v3' or value['loss']!='dapo':raise ValueError('explicit validated scorer/loss required')
    rule=value['cap_rule']
    if set(rule)!={'method','extra_tokens'} or rule['method']!='ceil-p99-plus' or type(rule['extra_tokens']) is not int or rule['extra_tokens']<1:
        raise ValueError('cap rule requires explicit positive extra_tokens above ceil(p99)')
    for key in ['generation_groups','backward_groups']:
        if type(value[key]) is not int or value[key] not in [1,2,4,8]:raise ValueError('memory batches must preserve complete groups and divide eight prompts')
    if type(value['evaluation_batch_size']) is not int or value['evaluation_batch_size']<1:raise ValueError('explicit evaluation batch required')
    if type(value['diagnostic_seed']) is not int or value['diagnostic_seed']<0 or type(value['gradient_checkpointing']) is not bool:raise ValueError('invalid seed/checkpointing setting')
    if (type(value['memory_margin_gib']) not in [int,float] or not math.isfinite(value['memory_margin_gib']) or value['memory_margin_gib']<=0
        or type(value['top_p']) not in [int,float] or not 0<value['top_p']<=1 or type(value['top_k']) is not int or value['top_k']<0):raise ValueError('explicit finite memory margin and generation filters required')
    return {**value,'completion_limit':math.ceil(summary['length_percentiles']['99'])+rule['extra_tokens']}


def _finite(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite preflight evidence')
    if isinstance(value,dict):
        for v in value.values():_finite(v)
    if isinstance(value,list):
        for v in value:_finite(v)



def _cached_unit(directory, identity, execute):
    marker=directory/'unit.complete.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        if value['identity']!=identity:raise ValueError('preflight unit identity changed')
        for relative,digest in value['files'].items():
            path=(directory/relative).resolve()
            if not path.is_relative_to(directory.resolve()) or file_hash(path)!=digest:
                raise ValueError('preflight unit evidence changed')
        print('preflight: reusing verified unit '+directory.name,flush=True)
        return json.loads((directory/'unit-result.json').read_text())
    result=execute();_finite(result)
    _write_json(directory/'unit-result.json',result)
    files={str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file() and p.name!='unit.complete.json'}
    _write_json(marker,dict(identity=identity,files=files))
    return result


def collect_preflight(config_path,baseline_path,controls_name,settings_path,root,name,dependencies=None):
    root=Path(root).resolve();path=_within(root,config_path)
    plan,training,evaluation,_=load_grpo_prepared(path,root)
    baseline,summary,baseline_evidence=_baseline(root,baseline_path,path)
    settings=_settings(json.loads(Path(settings_path).read_text()),summary)
    if any(settings[k]!=baseline['settings'][k] for k in ['top_p','top_k']):raise ValueError('baseline and training sampling filters must match; collect a named matched baseline')
    controls=(dependencies.verify_controls(root,controls_name) if dependencies is not None else verify_controls(root,controls_name))
    if not controls.get('passed') or not controls.get('algorithm_checks',{}).get('passed'):raise ValueError('controls must pass before GPU preflight')
    groups=[plan['optimizer_prompt_order'][i:i+8] for i in range(0,512,8)]
    directory=root/'runs/pilot-4'/plan['model'].replace('/','--')/'preflight'/safe_name(name)
    config=dict(protocol_version='pilot4-preflight-v1',source_path=str(path.relative_to(root)),source_sha256=file_hash(path),
                baseline=baseline_evidence,controls_name=controls_name,controls=controls,settings=settings,
                optimizer_groups=groups,approved=plan['approved'],optimizer=plan['optimizer'],
                run_path=str(directory.relative_to(root)),implementation=_implementation(),
                prompt_template=plan['source_prompt_contract'],model=plan['model'],model_revision=plan['model_revision'])
    by_id={r['id']:r for r in training};trials=[by_id[q] for group in groups[:2] for q in group]
    with run_lock(directory):
        _save_frozen(directory/'config.json',config)
        prepared=root/'plans'/safe_name(name)/'grpo.preflight.json';_save_frozen(prepared,config)
        if (directory/'complete.json').exists():
            _verify(directory,config)
            return json.loads((directory/'results/results.json').read_text())
        engine=None
        try:
            _write_state(directory,'running',0,4)
            from pilot_eval.grpo_gpu import GPUPreflightDependencies
            engine=(dependencies or GPUPreflightDependencies()).load_preflight(plan,settings,directory)
            runtime=engine.runtime();_finite(runtime)
            if runtime.get('dtype')!='float32' or runtime.get('adapter_dtype')!='float32' or not runtime.get('deterministic_algorithms') or runtime.get('tf32') is not False:
                raise ValueError('live FP32/deterministic/TF32-off policy required')
            _save_frozen(directory/'runtime.json',runtime)
            print('preflight: matched untuned and zero-adapter greedy outputs',flush=True)
            identity=_hash(dict(config=config,runtime=runtime))
            zero=_cached_unit(directory/'zero',identity,lambda:engine.zero_equivalence(evaluation,settings))
            _write_json(directory/'results/zero-equivalence.json',zero)
            if [r['id'] for r in zero]!=[r['id'] for r in evaluation] or any(r['baseline_token_ids']!=r['zero_token_ids'] for r in zero):raise ValueError('zero-adapter greedy output mismatch')
            _write_state(directory,'running',1,4)
            results=[]
            for n in range(2):
                print(f'preflight: disposable two-step trial {n+1}/2',flush=True)
                unit=directory/f'trial-{n}'
                result=_cached_unit(unit,identity,lambda:engine.trial(trials,settings,settings['diagnostic_seed'],unit))
                _finite(result);_write_json(directory/f'trial-{n}/results.json',result)
                if (not result['zero_initialized'] or result['base_before']!=result['base_after'] or len(result['steps'])!=2 or len(result['responses'])!=128):raise ValueError('trial initialization/base integrity/workload mismatch')
                results.append(result);_write_state(directory,'running',n+2,4)
            # Timing is deliberately excluded; identical execution state and draws are required.
            if (any(results[0][k]!=results[1][k] for k in ['state_sha256','responses']) or
                [(x['loss'],x['gradient_norm']) for x in results[0]['steps']] !=
                [(x['loss'],x['gradient_norm']) for x in results[1]['steps']]):
                raise ValueError('disposable repeatability mismatch')
            print('preflight: final-cap capacity probe',flush=True)
            longest=sorted(training,key=lambda r:r['prompt_tokens'],reverse=True)[:8]
            capacity=_cached_unit(directory/'capacity',identity,lambda:engine.capacity(longest,settings,directory/'capacity'));_finite(capacity)
            _write_json(directory/'capacity/results.json',capacity)
            if (not capacity['base_unchanged'] or not capacity['synthetic_full_cap'] or capacity['completions']!=64 or capacity['completion_tokens']!=settings['completion_limit']):raise ValueError('incomplete final-cap capacity/integrity evidence')
            if capacity['free_memory_bytes']<settings['memory_margin_gib']*2**30:raise ValueError('measured capacity has insufficient reviewed memory margin')
            maximum_step=max(x['seconds'] for r in results for x in r['steps'])
            inclusive_step=max(r['elapsed_seconds']/2 for r in results)
            zero_seconds=getattr(engine,'zero_elapsed_seconds',None)
            timing_path=directory/'results/zero-timing.json'
            if timing_path.exists():zero_seconds=json.loads(timing_path.read_text())['seconds']
            workload=dict(training_completions=4096,greedy_evaluation_responses=750,sampled_evaluation_responses=2400,
                          zero_equivalence_responses=300,checkpoint_recovery_max_redo_steps=32)
            result=dict(passed=True,runtime=runtime,capacity=capacity,workload=workload,
                        observed_step_seconds=maximum_step,projection=dict(training_seconds=64*inclusive_step,
                          basis='two-step inclusive elapsed cost; zero-output timings and forced-cap generation; conditional estimates',
                          greedy_evaluation_seconds=None if zero_seconds is None else zero_seconds*2.5,
                          sampled_evaluation_full_cap_seconds=capacity['generation_seconds']*(2400/64),
                          model_loading_seconds=(json.loads((directory/'results/loading-timing.json').read_text())['seconds']
                              if (directory/'results/loading-timing.json').exists() else None),
                          worst_full_cap_training_seconds=64*capacity['elapsed_seconds'],max_redo_seconds=32*capacity['elapsed_seconds']),
                        note='Measured diagnostic estimates, not guarantees. Synthetic full-cap stress is not scientific training. Actual future length distribution is unknown.')
            _write_json(directory/'results/results.json',result);_seal(directory,config);_write_state(directory,'completed',4,4)
            return result
        except BaseException as exc:
            _write_state(directory,'failed',0,4,str(exc));raise
        finally:
            if engine is not None:engine.close()


def load_preflight(path,root):
    root=Path(root).resolve();path=_within(root,path);config=json.loads(path.read_text())
    if config['protocol_version']!='pilot4-preflight-v1' or config['implementation']!=_implementation():raise ValueError('preflight implementation/config changed')
    if file_hash(_within(root,config['source_path']))!=config['source_sha256']:raise ValueError('preflight source changed')
    load_grpo_prepared(root/config['source_path'],root)
    _,_,baseline=_baseline(root,root/config['baseline']['config'],root/config['source_path'])
    if baseline!=config['baseline']:raise ValueError('baseline identity changed')
    if verify_controls(root,config['controls_name'])!=config['controls']:raise ValueError('control evidence changed')
    directory=_within(root,config['run_path']);_verify(directory,config)
    result=json.loads((directory/'results/results.json').read_text())
    if not result['passed']:raise ValueError('preflight did not pass')
    return config,result,directory


def freeze_protocol(config_path,review_path,root,name):
    root=Path(root).resolve();flight,result,directory=load_preflight(config_path,root)
    review=json.loads(Path(review_path).read_text());_finite(review)
    if set(review)!={'reviewed','notes','margin','preregistration','monitors'} or review['reviewed'] is not True or not review['notes'].strip():raise ValueError('explicit evidence review and notes required')
    if type(review['margin']) not in [int,float] or not 0<review['margin']<1:raise ValueError('explicit non-inferiority margin required')
    prereg=review['preregistration']
    if set(prereg)!={'author','written_at','prediction','falsifier'} or any(not isinstance(v,str) or not v.strip() for v in prereg.values()):raise ValueError('user-authored dated prediction and falsifier required')
    timestamp=datetime.datetime.fromisoformat(prereg['written_at'])
    if timestamp.tzinfo is None or timestamp>datetime.datetime.now(datetime.timezone.utc):raise ValueError('valid past/current timezone-aware preregistration date required')
    if set(review['monitors'])!={'cap_fraction','dead_group_fraction','length_ratio_change_8_steps'}:raise ValueError('all monitor policies required')
    for v in review['monitors'].values():
        if set(v)!={'threshold','action'} or type(v['threshold']) not in [int,float] or not 0<v['threshold']<=1 or v['action'] not in ['flag','pause','stop']:raise ValueError('explicit monitor threshold/action required')
    config=dict(protocol_version='pilot4-frozen-v1',state='frozen',preflight_path=str(_within(root,config_path).relative_to(root)),
                preflight_sha256=file_hash(config_path),preflight_marker_sha256=file_hash(directory/'complete.json'),
                settings=flight['settings'],optimizer_groups=flight['optimizer_groups'],optimizer=flight['optimizer'],
                approved=flight['approved'],runtime=result['runtime'],controls=flight['controls'],
                controls_name=flight['controls_name'],review=review,source_path=flight['source_path'],
                source_sha256=flight['source_sha256'],implementation=flight['implementation'])
    target=root/'plans'/safe_name(name)/'grpo.frozen.json'
    with run_lock(target.parent):
        _save_frozen(target,config)
        _write_json(target.parent/'freeze-complete.json',dict(config_sha256=file_hash(target)))
    return target


def load_frozen(path,root):
    root=Path(root).resolve();path=_within(root,path);config=json.loads(path.read_text())
    if config.get('protocol_version')!='pilot4-frozen-v1' or config.get('state')!='frozen':raise ValueError('frozen GRPO execution protocol required')
    marker=json.loads((path.parent/'freeze-complete.json').read_text())
    if marker['config_sha256']!=file_hash(path):raise ValueError('frozen protocol changed')
    if file_hash(root/config['preflight_path'])!=config['preflight_sha256']:raise ValueError('preflight plan changed')
    flight,_,directory=load_preflight(root/config['preflight_path'],root)
    if file_hash(directory/'complete.json')!=config['preflight_marker_sha256']:raise ValueError('preflight seal changed')
    return config
