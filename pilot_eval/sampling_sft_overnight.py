"""Resume-first approved L4 scoring and three-arm training; no external export."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pilot_eval import sampling_sft as source, sampling_sft_gate as gate
from pilot_eval import sampling_sft_jobs as jobs, sampling_sft_pipeline as pipeline
from pilot_eval import sampling_sft_review as review, sampling_sft_training as training
from pilot_eval.sampling_sft_provider import RunPodProvider
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash

LIMITS={'0':34,'2':35,'5':40}
DIFFERENCES={'0-2':49,'0-5':54,'2-5':49}
FLAGS=sorted([f'k{k}:omissions-require-T16' for k in LIMITS]+['uncovered-set-difference:'+p for p in DIFFERENCES])


def validate_coverage(report):
    c=report['coverage'];ids=c['uncovered_ids'];pairs=c['pairwise_flags']
    if (c['blocked'] or set(ids)!=set(LIMITS) or any(len(ids[k])!=n or len(set(ids[k]))!=n for k,n in LIMITS.items())
            or set(pairs)!=set(DIFFERENCES) or any(not pairs[k]['flagged'] or pairs[k]['count']!=n or
                len(set(ids[k.split('-')[0]]) ^ set(ids[k.split('-')[1]]))!=n for k,n in DIFFERENCES.items())
            or c['t16_required']!={k:True for k in LIMITS}):
        raise ValueError('Coverage differs from the explicitly accepted three-arm cohort.')


def orchestrate(folder,ops):
    """Completed stages have durable pointers; failed stages are never marked complete."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def status(stage):
        source.save(folder/'status.json',{'state':stage,'external_copy_verified':False})
        print('Overnight stage:',stage,flush=True)
    status('preflight');ops.preflight()
    score=folder/'score.complete.json'
    if not score.exists():
        status('gate-scoring')
        ops.score();source.frozen(score,{'state':'verified'})
    selection_file=folder/'selection.complete.json'
    if not selection_file.exists():
        status('gate-report-and-selection')
        selection=ops.report_selection();source.frozen(selection_file,{'selection':selection})
    selection=json.loads(selection_file.read_text())['selection']
    status('training');result=ops.training(selection)
    source.frozen(folder/'complete.json',{'state':'training-verified','inventory':result,'external_copy_verified':False})
    status('training-verified')
    return result


class Operations:
    def __init__(self,config):
        self.cfg=config;self.root=Path(config['root']).resolve();self.gate=inside(self.root,config['gate']);self.job=inside(self.root,config['score_job']);self.data=inside(self.root,config['data'])

    def preflight(self):
        for name,digest in self.cfg['input_sha256'].items():
            if file_hash(inside(self.root,name))!=digest:raise ValueError('Overnight input changed: '+name)
        for name,digest in self.cfg['code_sha256'].items():
            if file_hash(Path(__file__).with_name(name))!=digest:raise ValueError('Overnight implementation changed: '+name)
        subprocess.run(['runpodctl','--version'],check=True,capture_output=True,timeout=30)
        provider=RunPodProvider();pod=provider.own_pod_id();observed=provider.inspect(pod)
        if observed.get('id')!=pod or observed.get('storageVolumeId',observed.get('networkVolumeId'))!=self.cfg['volume_id']:
            raise ValueError('Pod or persistent volume identity mismatch.')
        if observed.get('storageMountPath') and not self.root.is_relative_to(Path(observed['storageMountPath'])):
            raise ValueError('Artifact root is not on the verified persistent volume.')
        from pilot_eval import sampling_sft_budget as budget
        if budget.number(observed.get('costPerHr'),True)>budget.number(self.cfg['rate'],True):raise ValueError('Live rate exceeds approved estimate.')
        _,g,_=gate.load(self.gate,self.root);_,j,_=jobs._load(self.job,self.root)
        pipeline.validate_gate_job(self.root,self.gate,self.job,j)
        if j['request']['network_volume_id']!=self.cfg['volume_id']:raise ValueError('Gate job volume mismatch.')
        if g['protocol']!='sampling-sft-gate-l4-v1':raise ValueError('Expected independently bound L4 oracle.')
        from pilot_eval.sampling_sft_l4_oracle import capture
        if capture()!=g['oracle_runtime']:raise ValueError('Current L4 runtime does not match the gate.')
        rp=json.loads((self.root/g['review_plan']).read_text())
        if rp.get('coverage_policy')!=review.AUTO_POLICY or file_hash(self.root/rp['generation_plan'])!=review.AUTO_POLICY['generation_sha256']:
            raise ValueError('Not the specifically accepted generation and automatic coverage policy.')
        pipeline.training_ready(self.data,self.root)
        data=json.loads(self.data.read_text())
        if data['review_plan']!=g['review_plan']:raise ValueError('Gate and training use different reviews.')
        cov=review.verify(self.root/g['review_plan'],self.root)
        validate_coverage({'coverage':{'blocked':cov['coverage_blocked'],
            'uncovered_ids':{k:cov['arms'][k]['uncovered_ids'] for k in LIMITS},
            'pairwise_flags':cov['pairwise_uncovered_differences'],
            't16_required':{k:cov['arms'][k]['t16_required'] for k in LIMITS}}})
        export=json.loads((inside(self.root,data['run_path'])/'results/export.json').read_text())
        if any(export['arms'][k]['missing']!=n or export['arms'][k]['examples']!=512-n for k,n in LIMITS.items()):
            raise ValueError('Training exports do not match accepted coverage counts.')
        # The training backend loads by Hub ID: verify the pinned cache is present before any scoring spend.
        from huggingface_hub import try_to_load_from_cache
        from pilot_eval import sampling_sft as sampling
        audit=sampling.verify(self.root/g['source_plan'],self.root)['audit'];identity=audit['training']['identity']
        model_root=inside(self.root,g['settings']['local_model'])
        for name in ('config.json','generation_config.json','merges.txt','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json'):
            cached=try_to_load_from_cache(identity['model'],name,revision=identity['model_revision'])
            if not isinstance(cached,str) or not Path(cached).is_file() or file_hash(cached)!=file_hash(model_root/name):
                raise ValueError('Pinned HF training cache missing or mismatched: '+name)

    def score(self):
        print('Starting or verifying independent L4 gate scoring.',flush=True)
        pipeline.gate_score(self.gate,self.root,self.job,self.cfg['reviewer'])

    def report_selection(self):
        print('Verifying report and accepted coverage.',flush=True)
        pipeline.gate_report(self.gate,self.root)
        report=gate.verify(self.gate,self.root)
        if report['execution_kind']!='real-cuda':raise ValueError('Real CUDA gate evidence required.')
        validate_coverage(report)
        c=self.cfg;forecast={'arms':[0,2,5],'per_arm':{k:{'rate_usd_per_hour':c['rate'],'estimated_hours':c['hours'][k],
            'setup_hours':c['setup_hours']} for k in LIMITS},'notes':'User approved all three arms and stated cost waiver for this run.'}
        target=inside(self.root,c['state_dir'])/'forecast.json';source.frozen(target,forecast)
        request=gate.request(self.gate,self.root,target)
        if request['required_acknowledgments']!=FLAGS or request['coverage_blocked']:raise ValueError('Unexpected gate flags; stop for review.')
        approval={'request_path':request['request_path'],'request_sha256':request['request_sha256'],
            'report_sha256':request['report_sha256'],'budget_sha256':request['budget_snapshot']['sha256'],
            'reviewer':c['reviewer'],'notes':'User accepted k0/k2/k5 omissions 34/35/40 and all three arms for this run; pairwise differences 49/54/49.',
            'gate_approved':True,'acknowledged_coverage_flags':FLAGS,'review_80_percent':True,'waive_budget_limits':True}
        approval_path=inside(self.root,c['state_dir'])/'gate-approval.json';source.frozen(approval_path,approval)
        return gate.approve(self.gate,self.root,approval_path)['selection_path']

    def training(self,selection):
        print('Verifying selection and running k0, k2, k5 training sequentially.',flush=True)
        c=self.cfg;training.selection(self.root,self.root/selection,json.loads(self.data.read_text()))
        entries=pipeline.prepare_training_jobs(self.data,self.root/selection,self.root,c['name']+'-train',
            c['volume_id'],c['rate'],c['reviewer'],c['hours'],c['setup_hours'])
        return pipeline.training(self.data,self.root,entries)['inventory']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','gate','score-job','data','name','volume-id','rate','reviewer','hours-k0','hours-k2','hours-k5','setup-hours'):
        p.add_argument('--'+n,required=True)
    a=p.parse_args(argv);root=Path(a.root).resolve()
    from pilot_eval.sft import safe_name
    safe_name(a.name)
    state_dir='runs/sampling-sft/overnight/'+a.name
    cfg={'root':str(root),'gate':str(inside(root,a.gate).relative_to(root)),'score_job':str(inside(root,a.score_job).relative_to(root)),
        'data':str(inside(root,a.data).relative_to(root)),'name':a.name,'state_dir':state_dir,'volume_id':a.volume_id,
        'rate':a.rate,'reviewer':a.reviewer,'hours':{'0':a.hours_k0,'2':a.hours_k2,'5':a.hours_k5},'setup_hours':a.setup_hours}
    cfg['input_sha256']={cfg[k]:file_hash(root/cfg[k]) for k in ('gate','score_job','data')}
    names=['sampling_sft_overnight.py','sampling_sft_training_backend.py','sampling_sft_pipeline.py','sampling_sft_gate.py','sampling_sft_l4_oracle.py']
    cfg['code_sha256']={name:file_hash(Path(__file__).with_name(name)) for name in names}
    folder=inside(root,state_dir)
    try:
        with run_lock(folder):
            source.frozen(folder/'config.json',cfg)
            result=orchestrate(folder,Operations(cfg))
        print(json.dumps({'state':'training-verified','inventory':result,'external_copy_verified':False},indent=2))
        return 0
    except (ValueError,OSError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
        source.save(folder/'status.json',{'state':'stopped','error':str(exc),'external_copy_verified':False})
        print('Overnight workflow stopped:',exc,file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
