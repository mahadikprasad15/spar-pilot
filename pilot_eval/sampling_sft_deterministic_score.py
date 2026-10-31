"""Bounded, isolated V1 deterministic-scheduling score investigation.

Two real jobs, each starting a fresh engine and scoring the same frozen fixture
batch twice. This descriptive diagnostic cannot qualify sampling or relax gates.
"""
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_instrument as instrument
from pilot_eval import sampling_sft_jobs as jobs
from pilot_eval import sampling_sft_budget as budget
from pilot_eval.sampling_sft_sampler import scored
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.sft import safe_name
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash

ENV_NAME = 'VLLM_ENABLE_V1_MULTIPROCESSING'
ENV_VALUE = '0'
LANES = ('a', 'b')


def compare_scores(fixtures, left, right):
    if len(left) != len(fixtures) or len(right) != len(fixtures):
        raise ValueError('Scoring cardinality mismatch.')
    differences = []
    for req, a, b in zip(fixtures, left, right):
        scored(a, req); scored(b, req)
        if a['token_ids'] != b['token_ids'] or a['positions'] != b['positions']:
            raise ValueError('Score alignment mismatch.')
        for i, (x, y) in enumerate(zip(a['logprobs'], b['logprobs'])):
            if x != y:
                differences.append(dict(problem_id=req['problem_id'], token_index=i,
                    token_id=a['token_ids'][i], position=a['positions'][i], left=x, right=y,
                    abs_difference=abs(x-y)))
    return dict(compared_values=sum(len(a['logprobs']) for a in left),
        different_values=len(differences), max_abs_difference=max((v['abs_difference'] for v in differences),default=0),
        first_difference=differences[0] if differences else None, differences=differences)


def prepare(score_config, output_root, name):
    root, parent, folder = instrument.load(score_config, output_root)
    safe_name(name)
    prior = instrument.verify(score_config, root)['result']
    if parent['settings']['mode'] != 'scores' or parent['settings']['runtime']['prefix_caching'] is not False:
        raise ValueError('Require completed no-prefix-cache score plan.')
    fixture = inside(root,parent['settings']['fixtures'])
    if prior['fixture_sha256'] != file_hash(fixture): raise ValueError('Fixture identity mismatch.')
    path = root/'plans'/name/'sampling-sft.deterministic-score.json'
    target = folder/'diagnostics'/name
    overlay = dict(base_runtime=parent['settings']['runtime'], environment={ENV_NAME:ENV_VALUE})
    plan = dict(protocol='sampling-sft-deterministic-score-v1', name=name,
        parent=str(inside(root,score_config).relative_to(root)), parent_sha256=file_hash(inside(root,score_config)),
        fixture=str(fixture.relative_to(root)), fixture_sha256=file_hash(fixture),
        runtime=overlay, runtime_hash=_hash(overlay), implementation=file_hash(Path(__file__)),
        run_path=str(target.relative_to(root)), production_ready=False)
    if path.exists() or target.exists(): raise ValueError('Name already exists; preserve the prior attempt.')
    source.frozen(path,plan)
    source.frozen(target/'plan-seal.json', {'config_sha256':_hash(plan)})
    return path


def load(config, output_root):
    root=Path(output_root).resolve(); config=inside(root,config)
    plan=json.loads(config.read_text()); folder=inside(root,plan['run_path'])
    if (plan['protocol']!='sampling-sft-deterministic-score-v1' or plan['implementation']!=file_hash(Path(__file__))
        or plan['runtime_hash']!=_hash(plan['runtime']) or plan['runtime']['environment']!={ENV_NAME:ENV_VALUE}
        or json.loads((folder/'plan-seal.json').read_text())!={'config_sha256':_hash(plan)}
        or file_hash(root/plan['parent'])!=plan['parent_sha256'] or file_hash(root/plan['fixture'])!=plan['fixture_sha256']):
        raise ValueError('Diagnostic plan, implementation or parent identity changed.')
    _, parent, _ = instrument.load(root/plan['parent'],root)
    prior=instrument.verify(root/plan['parent'],root)['result']
    if (parent['settings']['mode']!='scores' or parent['settings']['runtime']!=plan['runtime']['base_runtime']
        or parent['settings']['fixtures']!=plan['fixture'] or prior['fixture_sha256']!=plan['fixture_sha256']):
        raise ValueError('Diagnostic parent/fixture changed.')
    return root,plan,folder,parent


def lane_result(folder,plan,lane):
    if lane not in LANES: raise ValueError('Unknown lane.')
    path=folder/lane
    if not (path/'complete.json').exists(): return None
    seal=json.loads((path/'complete.json').read_text())
    if seal!={'config_sha256':_hash(plan),'results_sha256':file_hash(path/'results.json')}:
        raise ValueError('Diagnostic lane seal changed.')
    return json.loads((path/'results.json').read_text())


def run(config, output_root, lane, job, *, dependencies=None):
    root,plan,folder,parent=load(config,output_root)
    existing=lane_result(folder,plan,lane)
    if existing is not None:return existing
    if dependencies is None:
        instrument.authorize(root,config,job,expected_command='sampling-sft-deterministic-score-run')
        _, bound_job, _ = jobs._load(job,root)
        argv=bound_job['request']['command']
        if '--lane' not in argv or argv[argv.index('--lane')+1]!=lane:
            raise ValueError('Lane does not match active job.')
    path=folder/lane
    with _writer(path):
        if (path/'started.json').exists():raise ValueError('Failed or interrupted lane preserved; use a new diagnostic name.')
        source.frozen(path/'started.json',dict(config_sha256=_hash(plan),lane=lane))
        source.save(path/'status.json',dict(state='running'))
        try:
            if 'vllm' in sys.modules:raise ValueError('vLLM already imported before setting diagnostic environment.')
            os.environ[ENV_NAME]=ENV_VALUE
            if os.environ.get(ENV_NAME)!=ENV_VALUE:raise ValueError('Diagnostic environment not applied.')
            if dependencies is None:
                from pilot_eval.sampling_sft_engine import open_engine
                engine,tokenizer,runtime=open_engine(parent,root)
                from vllm import envs as vllm_envs
                if vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING is not False:
                    raise ValueError('vLLM effective V1 multiprocessing is not disabled.')
                runtime={**runtime,'vllm_effective_multiprocessing':vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING}
                for problem in parent['problems']:
                    actual=tokenizer.apply_chat_template([dict(role='user',content=problem['user_message'])],
                        tokenize=True,add_generation_prompt=True)
                    if list(actual)!=problem['scorer_prompt_ids']:raise ValueError('Scorer prompt retokenization changed.')
            else:
                engine,tokenizer,runtime=dependencies.open(parent,root)
                runtime={**runtime,'execution_kind':'cpu-fixture'}
            fixtures=json.loads((root/plan['fixture']).read_text())
            first=engine.score(fixtures);second=engine.score(fixtures)
            summary=compare_scores(fixtures,first,second)
            result=dict(lane=lane,fixture_sha256=plan['fixture_sha256'],runtime_hash=plan['runtime_hash'],
                environment={ENV_NAME:os.environ.get(ENV_NAME)},worker=runtime,
                completed_utc=datetime.now(timezone.utc).isoformat(),host=platform.node(),pid=os.getpid(),first=first,second=second,
                within_process=summary,production_ready=False)
            source.save(path/'results.json',result)
            source.save(path/'status.json',dict(state='completed'))
            source.frozen(path/'complete.json',dict(config_sha256=_hash(plan),results_sha256=file_hash(path/'results.json')))
            return result
        except BaseException as exc:
            source.save(path/'status.json',dict(state='failed',error=str(exc),type=type(exc).__name__))
            raise


def report(config, output_root):
    root,plan,folder,_=load(config,output_root)
    a,b=[lane_result(folder,plan,lane) for lane in LANES]
    if a is None or b is None:raise ValueError('Both sealed diagnostic lanes required.')
    if any(x['runtime_hash']!=plan['runtime_hash'] or x['fixture_sha256']!=plan['fixture_sha256']
           or x['environment']!={ENV_NAME:ENV_VALUE} for x in (a,b)):
        raise ValueError('Lane evidence differs from diagnostic identity.')
    fixtures=json.loads((root/plan['fixture']).read_text())
    result=dict(scope='20 frozen fixtures; repeated scoring, not pilot replay or qualification',
        runtime_hash=plan['runtime_hash'], fixture_sha256=plan['fixture_sha256'],
        processes={lane:dict(host=value['host'],pid=value['pid'],execution_kind=value['worker']['execution_kind'])
                   for lane,value in zip(LANES,(a,b))},
        within_a=compare_scores(fixtures,a['first'],a['second']),
        within_b=compare_scores(fixtures,b['first'],b['second']),
        between_first=compare_scores(fixtures,a['first'],b['first']),
        between_second=compare_scores(fixtures,a['second'],b['second']),
        parents={lane:file_hash(folder/lane/'complete.json') for lane in LANES},production_ready=False)
    source.frozen(folder/'summary.json',result)
    return result


def stage(config, output_root, lane, volume_id, rate, reviewer, *, execute=False, estimated_hours='0.5'):
    root,plan,folder,parent=load(config,output_root)
    if lane not in LANES or not volume_id.strip() or not reviewer.strip():raise ValueError('Lane, volume and reviewer required.')
    budget.number(rate,True);budget.number(estimated_hours,True)
    source_config=root/parent['source_plan']
    source_plan=json.loads(source_config.read_text())
    name=plan['name']+'-'+lane
    control=folder/'jobs'/lane
    job=root/source_plan['run_path']/'jobs'/name/'job.plan.json'
    base=str((folder/lane).relative_to(root))
    request=dict(job_id=name,kind='gpu',stage='sampling',mode='runpod',rate_usd_per_hour=str(rate),
        estimated_hours=str(estimated_hours),setup_hours='0',network_volume_id=volume_id,
        command=[sys.executable,'-m','pilot_eval','sampling-sft-deterministic-score-run',
            '--config',str(inside(root,config)),'--job',str(job),'--lane',lane,'--output-root',str(root)],
        outputs=[base+'/'+n for n in ['results.json','status.json','complete.json']])
    source.frozen(control/'request.json',request)
    jobs.prepare(source_config,root,control/'request.json')
    approval=control/'approval.json'
    if not approval.exists():
        current=budget.initialize(source_config,root)
        source.frozen(approval,dict(reviewer=reviewer,notes='User explicitly approved bounded diagnostic and waived budget limits; actual cost remains unreconciled.',
            paid_execution_approved=True,preflight_reviewed=True,waive_budget_limits=True,
            request_sha256=_hash(request),budget_sha256=current['sha256'],
            source_audit_sha256=file_hash(root/source_plan['run_path']/'results/source-audit.json')))
    if not execute:return dict(state='prepared',job=str(job),approval=str(approval),production_ready=False)
    result=jobs.run(job,root,approval)
    if not result['workload_success']:raise ValueError('Diagnostic job failed; inspect its retained execution and child log.')
    verified=lane_result(folder,plan,lane)
    if not verified or verified['worker'].get('execution_kind')!='real-cuda':
        raise ValueError('Real CUDA diagnostic evidence required.')
    return dict(state='completed',job=str(job),production_ready=False)
