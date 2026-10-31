"""Bounded cache on/off diagnostic on existing, sealed T3 inputs."""
import argparse
import json
import sys
import threading
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_instrument as instrument
from pilot_eval import sampling_sft_jobs as jobs
from pilot_eval import sampling_sft_budget as budget
from pilot_eval.sampling_sft_sampler import scored
from pilot_eval.sft import safe_name
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash


def compare_scores(fixtures,on,off,hf):
    if not (len(fixtures)==len(on)==len(off)==len(hf)) or not fixtures:
        raise ValueError('Fixture and scorer cardinalities differ.')
    for fixture,*values in zip(fixtures,on,off,hf):
        for value in values:scored(value,fixture)
    result={}
    for name,left,right in [('on_vs_off',on,off),('on_vs_hf',on,hf),('off_vs_hf',off,hf)]:
        errors=[];means=[]
        for fixture,a,b in zip(fixtures,left,right):
            av=a['logprobs'];bv=b['logprobs']
            means.append(abs(sum(av)/len(av)-sum(bv)/len(bv)))
            errors.extend((abs(x-y),fixture['trace_id'],i,x,y) for i,(x,y) in enumerate(zip(av,bv)))
        err,trace,index,a,b=max(errors)
        result[name]=dict(max_abs_error=err,max_mean_error=max(means),
            tokens_over_0_01=sum(e[0]>0.01 for e in errors),tokens=len(errors),
            traces_over_mean_limit=sum(e>0.001 for e in means),
            passed=max(means)<=0.001 and err<=0.01,
            worst=dict(trace_id=trace,response_index=index,left=a,right=b))
    return result


def run(root,name,volume,rate,reviewer):
    root=Path(root).resolve();safe_name(name)
    if not reviewer.strip() or not volume.strip():raise ValueError('Reviewer and volume required.')
    budget.number(rate,True)
    old=root/'plans/a40-q-v1-scores/sampling-sft.instrument.json'
    old_oracle=root/'plans/a40-q-v1-oracle/sampling-sft.instrument.json'
    _,old_plan,_=instrument.load(old,root)
    _,oracle_plan,_=instrument.load(old_oracle,root)
    old_result=instrument.verify(old,root)['result']
    oracle_result=instrument.verify(old_oracle,root)['result']
    if old_plan['settings']['mode']!='scores' or oracle_plan['settings']['mode']!='oracle':
        raise ValueError('Expected saved vLLM and HF scoring plans.')
    if old_plan['runtime_hash']!=oracle_plan['runtime_hash'] or old_result['fixture_sha256']!=oracle_result['fixture_sha256']:
        raise ValueError('Original runtime or fixture identity differs.')
    if old_result['execution_kind']!='real-cuda' or oracle_result['execution_kind']!='real-cuda':
        raise ValueError('Diagnostic inputs must be real CUDA evidence.')
    fixture_path=root/old_plan['settings']['fixtures']
    if file_hash(fixture_path)!=old_result['fixture_sha256']:
        raise ValueError('Saved fixture hash mismatch.')
    fixtures=json.loads(fixture_path.read_text())
    instrument.validate_fixtures(fixtures,old_plan['settings']['runtime']['actual_stop_ids'],old_plan['problems'])
    source_plan=root/old_plan['source_plan'];_,parent,_=source.load(source_plan,root)
    control=root/parent['run_path']/'diagnostics'/name
    control.mkdir(parents=True,exist_ok=True)
    source.frozen(control/'inputs.json',dict(score_plan=str(old.relative_to(root)),
        score_plan_sha256=file_hash(old),oracle_plan=str(old_oracle.relative_to(root)),
        oracle_plan_sha256=file_hash(old_oracle),fixture_sha256=file_hash(fixture_path),
        old_score_complete_sha256=file_hash(root/old_plan['run_path']/'complete.json'),
        oracle_complete_sha256=file_hash(root/oracle_plan['run_path']/'complete.json'),
        diagnostic_code_sha256=file_hash(Path(__file__)),
        description='Cache on/off diagnosis; no production qualification or threshold change.'))
    results={}; evidence={}
    for enabled in [True,False]:
        label='on' if enabled else 'off';stage=f'{name}-{label}'
        settings=json.loads(json.dumps(old_plan['settings']))
        settings['runtime']['prefix_caching']=enabled
        source.frozen(control/f'settings-{label}.json',settings)
        plan=instrument.prepare(source_plan,root,control/f'settings-{label}.json',stage)
        cfg=json.loads(plan.read_text());job_path=root/parent['run_path']/'jobs'/stage/'job.plan.json'
        request=dict(job_id=stage,kind='gpu',stage='sampling',mode='runpod',rate_usd_per_hour=str(rate),
            estimated_hours='0.25',setup_hours='0',network_volume_id=volume,
            command=[sys.executable,'-m','pilot_eval','sampling-sft-instrument-run','--config',str(plan),
                     '--job',str(job_path),'--output-root',str(root)],
            outputs=[cfg['run_path']+'/'+p for p in ['complete.json','results.json','env.json']])
        source.frozen(control/f'request-{label}.json',request)
        job_path=jobs.prepare(source_plan,root,control/f'request-{label}.json')
        ledger=budget.initialize(source_plan,root);approval=control/f'approval-{label}.json'
        if not approval.exists():
            source.frozen(approval,dict(reviewer=reviewer,notes='User requested bounded cache diagnostic; old charges remain unreconciled.',
                paid_execution_approved=True,preflight_reviewed=True,waive_budget_limits=True,
                budget_sha256=ledger['sha256'],request_sha256=_hash(request),
                source_audit_sha256=file_hash(root/parent['run_path']/'results/source-audit.json')))
        done=threading.Event();log=job_path.parent/'logs/child.console.log'
        def follow():
            offset=0
            while True:
                if log.exists():
                    with log.open(errors='replace') as f:f.seek(offset);chunk=f.read();offset=f.tell()
                    if chunk:print(chunk,end='',flush=True)
                if done.wait(0.3):break
        watcher=threading.Thread(target=follow,daemon=True);watcher.start()
        print(f'Starting fixed-fixture scoring: prefix caching {label}.',flush=True)
        try:status=jobs.run(job_path,root,approval)
        finally:done.set();watcher.join(timeout=5)
        if not status['workload_success']:raise ValueError('Diagnostic job failed: '+str(log))
        result=instrument.verify(plan,root)['result']
        if result['execution_kind']!='real-cuda':raise ValueError('Real GPU evidence required.')
        results[label]=result['scores']
        evidence[label]=dict(plan=str(plan.relative_to(root)),plan_sha256=file_hash(plan),
            runtime_hash=cfg['runtime_hash'],results=str((root/cfg['run_path']/'results.json').relative_to(root)),
            results_sha256=file_hash(root/cfg['run_path']/'results.json'),
            completion_sha256=file_hash(root/cfg['run_path']/'complete.json'))
    comparisons=compare_scores(fixtures,results['on'],results['off'],oracle_result['scores'])
    comparisons['fresh_on_vs_original_vllm']=compare_scores(fixtures,results['on'],old_result['scores'],oracle_result['scores'])['on_vs_off']
    report=dict(state='completed',scope='diagnostic only; original T3 failure and production gate unchanged',
        fixture_sha256=file_hash(fixture_path),old_runtime_hash=old_plan['runtime_hash'],
        comparisons=comparisons,evidence=evidence,inputs=str((control/'inputs.json').relative_to(root)))
    source.frozen(control/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--name',required=True)
    p.add_argument('--volume-id',required=True);p.add_argument('--rate',required=True);p.add_argument('--reviewer',required=True)
    a=p.parse_args()
    try: print(json.dumps(run(a.output_root,a.name,a.volume_id,a.rate,a.reviewer),indent=2))
    except (ValueError,OSError,KeyError) as exc:
        print('Diagnostic stopped:',str(exc),file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
