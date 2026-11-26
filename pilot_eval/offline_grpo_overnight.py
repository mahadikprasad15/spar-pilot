"""Resumable, source-verified offline GRPO orchestration.

No model, network or repository write occurs during import. Native workflows
remain responsible for their own scientific gates and checkpoint recovery.
"""
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pilot_eval.run import _write_json
from pilot_eval.sft import safe_name
from pilot_eval.stage0_sources import inside
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash, _save_frozen


STAGES = {
    'pool': ('offline-pool-inventory','offline-pool-prepare','offline-pool-route','offline-pool-fresh-prepare','offline-pool-fresh-run','offline-pool-import'),
    'spectral': ('stage0-spectral-prepare','stage0-spectral-run'),
    'dataset': ('offline-grpo-dataset-prepare','offline-grpo-dataset-run','offline-grpo-dataset-report'),
    'diagnostics': ('offline-grpo-train-prepare','offline-grpo-train-check-diagnostics'),
    'train': ('offline-grpo-train-run',),
    'behaviour': ('offline-grpo-behaviour-prepare','offline-grpo-behaviour-run','offline-grpo-behaviour-report'),
    'geometry': ('offline-grpo-internals-prepare','offline-grpo-internals-geometry'),
    'push': ('offline-grpo-internals-collect',),
    'drift': ('offline-grpo-internals-collect',),
    'internals_report': ('offline-grpo-internals-report',),
    'cut_build': ('offline-grpo-cuts-prepare','offline-grpo-cuts-build'),
    'cut_agreement': ('offline-grpo-cuts-agreement',),
    'cut_evaluation': ('offline-grpo-cuts-evaluate',),
    'cut_report': ('offline-grpo-cuts-report',),
}
# Cheap to rebuild from pinned inputs, so these stages are never exported.
REGENERATED = {'spectral'}
# The approval may set max_seconds; otherwise only this fault guard bounds the run.
DEFAULT_MAX_SECONDS = 30*24*3600.
DEPENDENCIES = {
    'pool': [], 'spectral': [], 'dataset': ['pool'], 'diagnostics': ['pool'], 'train': ['diagnostics'],
    'behaviour': ['train'], 'geometry': ['train','spectral'], 'push': ['train'], 'drift': ['train'],
    'internals_report': ['geometry','push','drift'], 'cut_build': ['behaviour','geometry'],
    'cut_agreement': ['cut_build'], 'cut_evaluation': ['cut_agreement'], 'cut_report': ['cut_evaluation'],
}


def validate_graph(stages):
    seen = set()
    for stage in stages:
        identifier = stage['id']
        if not isinstance(identifier,str) or not identifier or identifier in seen:
            raise ValueError('Stage IDs must be unique and nonempty')
        if any(dependency not in seen for dependency in stage.get('depends_on',[])):
            raise ValueError('Every dependency must name an earlier stage')
        seen.add(identifier)


def _validate_spec(spec, root):
    if spec.get('protocol') != 'offline-grpo-overnight-v1':
        raise ValueError('Unsupported overnight plan')
    safe_name(spec['name'])
    stages = spec['stages']
    validate_graph(stages)
    if {s['id'] for s in stages} != set(STAGES):
        raise ValueError('All required experiment stages must be declared exactly once')
    for stage in stages:
        name = stage['id']
        if stage.get('depends_on') != DEPENDENCIES[name]:
            raise ValueError('Stage dependencies differ from the frozen experiment DAG: '+name)
        commands = stage.get('commands')
        if not isinstance(commands,list) or not commands:
            raise ValueError('Every stage requires explicit project commands')
        recipe=[cmd[0] for cmd in commands if isinstance(cmd,list) and cmd]
        valid_recipes=[list(STAGES[name])]
        if name=='pool':valid_recipes.append(['offline-pool-inventory','offline-pool-prepare','offline-pool-import'])
        if recipe not in valid_recipes:
            raise ValueError('Stage recipe must declare the full supported operation sequence: '+name)
        if commands[-1][0]!=STAGES[name][-1]:
            raise ValueError('Stage recipe must end with its required operation: '+name)
        if any(not isinstance(cmd,list) or not cmd or cmd[0] not in STAGES[name]
               or any(not isinstance(value,str) for value in cmd) for cmd in commands):
            raise ValueError('Stage has an unsupported command')
        for cmd in commands:
            if cmd.count('--output-root') != 1 or Path(cmd[cmd.index('--output-root')+1]).resolve() != root:
                raise ValueError('Child command must bind the selected artifact root')
            if cmd[0]=='offline-grpo-train-check-diagnostics' and cmd.count('--fidelity-tolerance')!=1:
                raise ValueError('Diagnostics require one explicit fidelity tolerance')
            if '--approval' in cmd:
                raise ValueError('Child approvals must use the single overnight approval')
            if '--config' in cmd:
                allowed={inside(root,stage['plan'])}
                allowed.update(inside(root,p) for p in stage.get('additional_plans',[]))
                if cmd.count('--config')!=1 or inside(root,cmd[cmd.index('--config')+1]) not in allowed:
                    raise ValueError('Child command must bind a declared stage plan')
        inside(root,stage['plan'])
    if not spec.get('export',{}).get('repo') or not spec['export'].get('paths'):
        raise ValueError('Private destination and explicit export paths required')
    for name in spec['export']['paths']:
        inside(root,name)
    if (not isinstance(spec.get('model_identity'),dict) or not all(spec['model_identity'].get(k) for k in ['model','model_revision','tokenizer_revision','base_sha256'])
            or type(spec.get('eos_token_id')) is not int or not spec.get('snapshot_hashes') or not spec.get('snapshot_path')):
        raise ValueError('Pinned model identity, EOS and snapshot inventory are required')
    inside(root,spec['snapshot_path'])
    if not spec.get('hf_cache_root') or type(spec.get('minimum_cache_free_bytes')) is not int or spec['minimum_cache_free_bytes']<0:
        raise ValueError('HF model cache root and nonnegative free-space requirement are required')
    if type(spec.get('minimum_free_bytes')) is not int or spec['minimum_free_bytes']<0:
        raise ValueError('Nonnegative free-space requirement required')
    if not spec.get('source_hashes') or not spec.get('software_versions'):
        raise ValueError('Pinned sources and software versions are required')
    by_id={stage['id']:stage for stage in stages}
    for group in [('diagnostics','train'),('geometry','push','drift','internals_report'),('cut_build','cut_agreement','cut_evaluation','cut_report')]:
        if len({str(inside(root,by_id[name]['plan'])) for name in group})!=1:
            raise ValueError('Related stages must bind the same source plan')
    for name in ['push','drift']:
        command=by_id[name]['commands'][-1]
        if command.count('--kind')!=1 or command[command.index('--kind')+1]!=name:
            raise ValueError('Collector kind differs from its DAG stage')
    if not isinstance(spec.get('stage0_runtime'),dict) or not spec['stage0_runtime'].get('session_id') or not isinstance(spec.get('training_runtime'),dict):
        raise ValueError('Explicit reviewed runtime required')
    if any(not isinstance(k,str) or not isinstance(v,str) for k,v in spec.get('source_hashes',{}).items()):
        raise ValueError('Source inventory must contain path and SHA-256 strings')
    for name in spec['source_hashes']:
        inside(root,name)
    serialized=json.dumps(spec,sort_keys=True)
    for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN']:
        token=os.environ.get(key)
        if token and token in serialized:
            raise ValueError('Credentials must not be stored in the plan')


def _code_inventory():
    """Bind tracked release code and this new driver, excluding unrelated untracked files."""
    import subprocess
    project=Path(__file__).parent.parent
    names=subprocess.check_output(['git','ls-files','--','pilot_eval/*.py',
        'tests/test_offline_grpo_engine.py','tests/test_grpo_algorithm.py'],cwd=project,text=True).splitlines()
    required=['pilot_eval/offline_grpo_overnight.py','tests/test_offline_grpo_engine.py','tests/test_grpo_algorithm.py']
    return {name:file_hash(project/name) for name in sorted(set(names+required))}


def prepare(spec_path, output_root):
    root=Path(output_root).resolve()
    spec_path=inside(root,spec_path)
    spec=json.loads(spec_path.read_text())
    _validate_spec(spec,root)
    directory=root/'runs/offline-grpo'/spec['name']/'overnight'
    path=root/'plans'/spec['name']/'offline.overnight.json'
    plan=dict(spec=spec,spec_path=str(spec_path.relative_to(root)),spec_sha256=file_hash(spec_path),
              implementation_sha256=file_hash(Path(__file__)),code_inventory=_code_inventory(),python=sys.executable)
    with _writer(directory):
        _save_frozen(path,plan)
        _save_frozen(directory/'config.json',plan)
        manifest_path=directory/'meta/run_manifest.json'
        if not manifest_path.exists():
            _save_frozen(manifest_path,dict(config_sha256=_hash(plan),created_at=datetime.now(timezone.utc).isoformat(),
                expected_stages=list(STAGES),export_repo=spec['export']['repo']))
        elif json.loads(manifest_path.read_text())['config_sha256']!=_hash(plan):
            raise ValueError('Overnight run manifest changed')
        if not (directory/'meta/status.json').exists():
            _write_json(directory/'meta/status.json',dict(state='prepared',stages={}))
    return path


def _load(path, root):
    root=Path(root).resolve();path=inside(root,path)
    plan=json.loads(path.read_text());spec=plan['spec']
    _validate_spec(spec,root)
    directory=root/'runs/offline-grpo'/spec['name']/'overnight'
    if (plan['implementation_sha256']!=file_hash(Path(__file__)) or plan['code_inventory']!=_code_inventory() or plan['python']!=sys.executable
            or file_hash(inside(root,plan['spec_path']))!=plan['spec_sha256']
            or json.loads((directory/'config.json').read_text())!=plan
            or json.loads((directory/'meta/run_manifest.json').read_text())['config_sha256']!=_hash(plan)):
        raise ValueError('Overnight plan, interpreter or code identity changed')
    return plan,directory


class Dependencies:
    def parity(self):
        import re
        import subprocess
        command=[sys.executable,'-m','pytest','-q','-rs',
                 'tests/test_offline_grpo_engine.py::test_pinned_trl_online_and_offline_update_parity']
        run=subprocess.run(command,capture_output=True,text=True,check=False,cwd=Path(__file__).parent.parent)
        if run.returncode!=0 or not re.search(r'\b1 passed\b',run.stdout) or re.search(r'\b\d+ skipped\b',run.stdout):
            raise ValueError('Pinned TRL gradient/update parity did not pass; skips are not passes')
        return dict(passed=True,test=command[-1],python=sys.executable)

    def runtime(self):
        from pilot_eval.stage0_qualification_backend import QualificationDependencies
        return QualificationDependencies().runtime()

    def hub(self):
        from pilot_eval.sampling_sft_hf import Hub
        return Hub()

    def cache_root(self):
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)

    def cache_lookup(self, model, filename, revision):
        from huggingface_hub import try_to_load_from_cache
        value=try_to_load_from_cache(model,filename,revision=revision,cache_dir=str(self.cache_root()))
        return Path(value) if isinstance(value,str) else None

    def fresh_fetch(self, repo, name, revision, cache_dir):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,
                                    cache_dir=str(cache_dir),force_download=True))

    def invoke(self, arguments, *, reservation_provider=None):
        from pilot_eval.cli import main
        model_commands={'offline-pool-route','offline-pool-fresh-run','offline-grpo-train-check-diagnostics','offline-grpo-train-run',
                        'offline-grpo-behaviour-run','offline-grpo-dataset-run','offline-grpo-internals-collect',
                        'offline-grpo-cuts-agreement','offline-grpo-cuts-evaluate'}
        if arguments[0] in model_commands:
            with _offline_model_scope():
                return main(arguments,reservation_provider=reservation_provider)
        return main(arguments,reservation_provider=reservation_provider)


def preflight(path, output_root, *, dependencies=None):
    root=Path(output_root).resolve();plan,directory=_load(path,root);spec=plan['spec']
    deps=dependencies or Dependencies();issues=[];evidence={}
    def check(name, function):
        try:
            evidence[name]=function()
        except Exception as exc:
            issues.append(dict(check=name,error_type=type(exc).__name__,message=_safe_message(exc)))
    def sources():
        failures=[]
        for relative,digest in spec['source_hashes'].items():
            source=inside(root,relative)
            if not source.is_file():failures.append(relative+': missing')
            elif file_hash(source)!=digest:failures.append(relative+': hash mismatch')
        if failures:raise ValueError('; '.join(failures))
        return dict(verified=len(spec['source_hashes']))
    def runtime():
        actual=deps.runtime()
        if actual!=spec['stage0_runtime']:raise ValueError('Stage 0 runtime differs from approved identity')
        training=spec['training_runtime']
        # Transformers' full determinism sets its own CUBLAS value in training; Stage 0 records
        # the value it requested. Both are deterministic, so that field is not compared.
        for field in ['device','python','cuda','versions','compute_capability']:
            if training.get(field)!=actual.get(field):raise ValueError('Training and Stage 0 runtimes differ on '+field)
        if (training.get('dtype')!='float32' or training.get('adapter_dtype')!='float32' or training.get('tf32') is not False
                or training.get('attention_implementation')!='eager' or training.get('deterministic_algorithms') is not True
                or actual.get('precision')!='float32' or actual.get('tf32') is not False or actual.get('attention')!='eager'):
            raise ValueError('Training or Stage 0 numeric mode differs from FP32 deterministic eager contract')
        return dict(stage0=actual,training=training,training_full_runtime='rechecked by the model loader before training')
    def software():
        if spec['software_versions'].get('trl')!='0.26.2':
            raise ValueError('The pinned TRL 0.26.2 must be declared')
        actual={name:importlib.metadata.version(name) for name in spec['software_versions']}
        if actual!=spec['software_versions']:raise ValueError('Pinned package versions differ')
        return dict(python=sys.executable,versions=actual)
    stages={s['id']:s for s in spec['stages']}
    def option(command,name):
        if command.count(name)!=1:raise ValueError('Missing explicit source option '+name)
        return command[command.index(name)+1]
    def cut_reference():
        from pilot_eval.offline_grpo_cuts import _reference
        command=next(c for c in stages['cut_build']['commands'] if c[0]=='offline-grpo-cuts-prepare')
        reference_root=Path(option(command,'--reference-root')).resolve()
        _reference(option(command,'--reference'),reference_root)
        return dict(verified=True)
    def dataset_source():
        from pilot_eval.offline_grpo_dataset import _legacy
        command=next(c for c in stages['dataset']['commands'] if c[0]=='offline-grpo-dataset-prepare')
        comparison_root=Path(option(command,'--comparison-root')).resolve() if '--comparison-root' in command else root
        source=inside(comparison_root,option(command,'--sft'))
        metadata=json.loads(source.read_text())
        if metadata.get('protocol')=='sampling-stage0-dataset-v1':
            _legacy(source,comparison_root,spec['model_identity'],spec['eos_token_id'])
        else:
            from pilot_eval.sampling_sft_training_data import verify
            verify(source,comparison_root)
        return dict(verified=True)
    def qualification():
        from pilot_eval.stage0_qualification import _load as load_qualification, _verify as verify_qualification
        command=next(c for c in stages['geometry']['commands'] if c[0]=='offline-grpo-internals-prepare')
        config,directory=load_qualification(inside(root,option(command,'--qualification')),root)
        result=verify_qualification(config,directory)
        if config['model_identity']!=spec['model_identity'] or result['runtime']!=spec['stage0_runtime']:
            raise ValueError('Qualification model or runtime differs from the approved plan')
        approved=[]
        for path in (directory/'reviews').glob('*/qualification.json'):
            receipt=json.loads(path.read_text());review=receipt.get('review',{})
            if (receipt.get('qualified') is True and receipt.get('config_sha256')==_hash(config)
                    and path.parent.name==_hash(review) and review.get('diagnostic_sha256')==file_hash(directory/'complete.json')
                    and receipt.get('runtime')==spec['stage0_runtime']):
                approved.append(path)
        if not approved:raise ValueError('Matching approved Stage 0 qualification receipt missing')
        return dict(verified=True)
    def runtime_files():
        training=next(c for c in stages['diagnostics']['commands'] if c[0]=='offline-grpo-train-prepare')
        if json.loads(inside(root,option(training,'--runtime')).read_text())!=spec['training_runtime']:
            raise ValueError('Training runtime file differs from approved identity')
        for name,operation in [('pool','offline-pool-fresh-prepare'),('dataset','offline-grpo-dataset-prepare'),('geometry','offline-grpo-internals-prepare')]:
            for command in stages[name]['commands']:
                if command[0]==operation and json.loads(inside(root,option(command,'--runtime')).read_text())!=spec['stage0_runtime']:
                    raise ValueError(name+' runtime file differs from Stage 0 identity')
        return dict(stage0_and_training='separately bound')
    def snapshot():
        import time
        start=time.monotonic();folder=inside(root,spec['snapshot_path']);total=0
        for name,digest in spec['snapshot_hashes'].items():
            path=inside(folder,name)
            if not path.is_file() or file_hash(path)!=digest:raise ValueError('Model snapshot file missing or changed: '+name)
            total+=path.stat().st_size
        return dict(files=len(spec['snapshot_hashes']),bytes=total,verification_seconds=time.monotonic()-start)
    def model_cache():
        return _verify_model_cache(spec,deps)
    def storage():
        import tempfile
        free=shutil.disk_usage(root).free
        if free<spec['minimum_free_bytes']:raise ValueError('Insufficient free space')
        with tempfile.TemporaryDirectory(prefix='.offline-preflight-',dir=root) as temporary:
            first=Path(temporary)/'first';second=Path(temporary)/'second'
            with first.open('wb') as stream:
                stream.write(b'offline-grpo');stream.flush();os.fsync(stream.fileno())
            os.replace(first,second)
            if second.read_bytes()!=b'offline-grpo':raise ValueError('Storage readback failed')
        return dict(free_bytes=free,root=str(root),atomic_rename=True)
    def hub():
        if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')!='0':
            raise ValueError('Set HF_HUB_ENABLE_HF_TRANSFER=0 for the approved storage environment')
        client=deps.hub();principal=client.api.whoami().get('name');info=client.info(spec['export']['repo'])
        if not principal or info.private is not True:raise ValueError('Authenticated private HF destination required')
        return dict(repo=spec['export']['repo'],principal=principal,private=True,write_access='not yet verified')
    for name,function in [('sources',sources),('cut_reference',cut_reference),('dataset_source',dataset_source),('qualification',qualification),('runtime_files',runtime_files),('snapshot',snapshot),('model_cache',model_cache),('runtime',runtime),('software',software),('parity',deps.parity),('storage',storage),('hf',hub)]:
        check(name,function)
    result=dict(ok=not issues,issues=issues,evidence=evidence,checked_at=datetime.now(timezone.utc).isoformat())
    _write_json(directory/'meta/preflight.json',result)
    return result


def _safe_message(exc):
    message=str(exc)
    for key in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN']:
        token=os.environ.get(key)
        if token:message=message.replace(token,'[redacted]')
    return message


def _native_verify(stage, root):
    """Reopen each native seal; status files are never completion evidence."""
    path=inside(root,stage['plan'])
    name=stage['id']
    if name=='pool':
        from pilot_eval.offline_grpo_pool import verify
        verify(path,root)
    elif name=='spectral':
        from pilot_eval.stage0_spectral import _load,_verify_cache
        config,_=_load(path,root)
        _verify_cache(config,inside(root,config['run_path']))
    elif name=='diagnostics':
        from pilot_eval.offline_grpo_training import _load,_verify_preflight
        config,directory,*_=_load(path,root)
        _verify_preflight(directory,config)
    elif name=='train':
        from pilot_eval.offline_grpo_training import _load,_verify_complete
        config,directory,*_=_load(path,root)
        _verify_complete(directory,root,config,config['optimizer_groups'])
    elif name=='behaviour':
        from pilot_eval.offline_grpo_behaviour import verify
        config,_=verify(path,root)
        report_path=inside(root,stage['report'])
        marker=json.loads((report_path/'complete.json').read_text())
        report_config=json.loads((report_path/'config.json').read_text())
        if (report_config['source_sha256']!=file_hash(path) or marker['config_sha256']!=_hash(report_config)
                or report_config['source_seal_sha256']!=file_hash(root/config['run_path']/'complete.json')):
            raise ValueError('Behavior report identity changed')
        for relative,digest in marker['files'].items():
            if file_hash(inside(report_path,relative))!=digest:raise ValueError('Behavior report payload changed')
    elif name=='dataset':
        from pilot_eval.offline_grpo_dataset import verify
        verify(path,root)
        config=json.loads(path.read_text());directory=inside(root,config['run_path'])
        if not (directory/'report.complete.json').is_file():raise FileNotFoundError('Dataset report missing')
    elif name in ['geometry','push','drift','internals_report']:
        from pilot_eval import offline_grpo_internals as workflow
        if name=='geometry':workflow.verify_geometry(path,root)
        elif name=='internals_report':workflow.verify_report(path,root)
        else:workflow.verify_collection(path,root,name)
    elif name in ['cut_build','cut_agreement','cut_evaluation','cut_report']:
        from pilot_eval import offline_grpo_cuts as workflow
        if name=='cut_build':workflow.verify_build(path,root)
        elif name=='cut_report':workflow.verify_report(path,root)
        else:workflow.verify_stage(path,root,name.removeprefix('cut_'))
    else:
        raise ValueError('Unknown native verifier')
    return True


def _write_state(directory, plan, state):
    from pilot_eval.stage0_workflow import _now
    state['updated_at']=_now()
    state['config_sha256']=_hash(plan)
    _write_json(directory/'meta/status.json',state)
    _write_json(directory/'checkpoints/progress.json',state)
    log=directory/'logs/events.jsonl'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        stream.write(json.dumps(state,sort_keys=True)+'\n')


def _authorization(path, plan, preflight_result, root):
    value=json.loads(inside(root,path).read_text())
    limit=value.get('max_seconds',DEFAULT_MAX_SECONDS)
    if (value.get('protocol')!='offline-grpo-prelaunch-v1' or value.get('approved') is not True
            or value.get('plan_sha256')!=_hash(plan) or value.get('stage0_runtime')!=plan['spec']['stage0_runtime'] or value.get('training_runtime')!=plan['spec']['training_runtime']
            or value.get('repo')!=plan['spec']['export']['repo'] or not value.get('reviewer')
            or value.get('preflight_sha256')!=_hash(preflight_result)
            or type(limit) not in [int,float] or not limit>0):
        raise ValueError('Explicit prelaunch approval must bind this exact plan, preflight, runtime and destination')
    return value


def _snapshot_report(plan, directory, root, state, attempt=None, elapsed=0.):
    """Freeze measured child summaries and wrapper progress for this attempt."""
    measured={}
    locations={
        'behaviour':lambda stage,config:inside(root,stage['report'])/'results/results.json',
        'dataset':lambda stage,config:inside(root,config['run_path'])/'results/results.json',
        'internals_report':lambda stage,config:inside(root,config['run_path'])/'report/results.json',
        'cut_report':lambda stage,config:inside(root,config['run_path'])/'report/results.json',
    }
    for stage in plan['spec']['stages']:
        name=stage['id']
        if name not in locations or state['stages'].get(name,{}).get('state')!='verified':
            continue
        source_plan=inside(root,stage['plan'])
        config=json.loads(source_plan.read_text())
        result_path=locations[name](stage,config)
        value=json.loads(result_path.read_text())
        if name=='behaviour':summary={k:value[k] for k in ['summaries','changes','online_comparison','online_comparison_status','anchor']}
        elif name=='dataset':summary={k:value[k] for k in ['full','paired_common','common_question_ids']}
        elif name=='internals_report':summary={k:value[k] for k in ['adapter_frobenius_norms','drift','preupdate','descriptive_similarities','push_alignment','unavailable','limitations']}
        else:summary={k:value[k] for k in ['paired','summaries','note']}
        measured[name]=dict(plan=str(source_plan.relative_to(root)),plan_sha256=file_hash(source_plan),
                            result=str(result_path.relative_to(root)),result_sha256=file_hash(result_path),summary=summary)
    pool=next(stage for stage in plan['spec']['stages'] if stage['id']=='pool')
    if state['stages'].get('pool',{}).get('state')=='verified':
        config=json.loads(inside(root,pool['plan']).read_text());handoff=inside(root,config['run_path'])/'handoff.json'
        if handoff.is_file():
            value=json.loads(handoff.read_text())
            measured['pool']=dict(result=str(handoff.relative_to(root)),result_sha256=file_hash(handoff),
                summary=dict(route=value['metadata'].get('route'),origin_summary=value.get('origin_summary')))
    result=dict(protocol='offline-grpo-overnight-report-v1',source_plan_sha256=_hash(plan),
        complete=all(state['stages'].get(name,{}).get('state')=='verified' for name in STAGES),
        stages=state['stages'],measurements=measured,
        limitations=['Read child reports for exact cohort, runtime and compatibility.',
          'Pre-update and fixed-context KL have different directions and contexts.',
          'One seed does not estimate run-to-run variability.',
          'The online/offline contrast changes answers, rewards, lengths and dead groups; it is not a pure effect of staleness.',
          'Geometry, overlap and layer-similarity results are descriptive, not causal or behavioral evidence.',
          'Matched-norm KL requires a separate scaled-adapter measurement.'])
    progress=directory/'checkpoints/progress.json'
    if measured.get('pool',{}).get('summary',{}).get('route')=='reuse-historical-128':
        result['limitations'].append('The pool mixes historical and newly sampled answers from two sessions; see the per-origin rates and the route decision.')
    result['progress_sha256']=file_hash(progress) if progress.exists() else None
    omissions=[]
    if 'behaviour' in measured and measured['behaviour']['summary']['online_comparison_status']['state']!='performed':
        omissions.append('Online behaviour comparison: '+measured['behaviour']['summary']['online_comparison_status']['state'])
    if 'internals_report' in measured:omissions.extend(measured['internals_report']['summary']['unavailable'])
    result['omissions']=omissions
    folder=directory/'results/snapshots'/_hash(result)
    _save_frozen(folder/'final-report.json',result)
    lines=['# Offline GRPO research report','','Complete: '+str(result['complete']),'',
           '| Stage | State |','|---|---|']
    lines.extend(f"| {name} | {state['stages'].get(name,{}).get('state','not_started')} |" for name in STAGES)
    lines.append('')
    for name,value in measured.items():
        lines.extend(['## '+name,'',f"Source: {value['result']} (SHA-256 {value['result_sha256']})",'',
                      '```json',json.dumps(value['summary'],indent=2,sort_keys=True),'```',''])
    if omissions:lines.extend(['## Omitted comparisons','']+['- '+x for x in omissions]+[''])
    lines.extend(['## Limitations','']+['- '+x for x in result['limitations']])
    text='\n'.join(lines)+'\n';report_path=folder/'final-report.md'
    if report_path.exists() and report_path.read_text()!=text:raise ValueError('Snapshot report changed')
    report_path.write_text(text)
    wrapper_files=['config.json','meta/run_manifest.json','meta/preflight.json','meta/authorization.json',
                   'meta/status.json','checkpoints/progress.json','logs/events.jsonl']
    wrapper_files += [str(p.relative_to(directory)) for p in sorted((directory/'logs').glob('*.log'))]
    for name in wrapper_files:
        source=directory/name
        if source.exists():
            target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
            data=source.read_bytes()
            if target.exists() and target.read_bytes()!=data:raise ValueError('Snapshot progress changed')
            target.write_bytes(data)
    _snapshot_recovery(directory, root, folder, attempt, elapsed)
    return folder


def _snapshot_recovery(directory, root, folder, attempt, elapsed):
    """Copy the state a resume reads, since these files keep changing after the export."""
    for start in sorted((directory/'attempts').glob('*/start.json')):
        for name in ['start.json','timing.json','timing.complete.json']:
            source=start.parent/name
            if source.is_file():
                target=folder/'attempts'/start.parent.name/name
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    if attempt is not None:
        # This attempt is still running, so the backup records its time so far.
        target=folder/'attempts'/attempt.name
        start=json.loads((attempt/'start.json').read_text())
        _write_json(target/'timing.json',dict(seconds=elapsed,reservation_id=start['reservation_id'],
            start_sha256=file_hash(attempt/'start.json'),final_integrity_sha256=None,usage_reconciled=False))
        _write_json(target/'timing.complete.json',dict(sha256=file_hash(target/'timing.json')))


def restore_wrapper(manifest_path, output_root):
    """After extracting a backup at the same root, put its recovery state back in place."""
    import shutil as copier
    from pilot_eval.sampling_sft_exports import verify_manifest
    root=Path(output_root).resolve();manifest=verify_manifest(inside(root,manifest_path),root)
    snapshot=inside(root,manifest['wrapper_snapshot']);directory=snapshot.parent.parent.parent;restored=0
    for source in sorted(snapshot.rglob('*')):
        relative=source.relative_to(snapshot)
        if not source.is_file() or relative.name.startswith('final-report'):continue
        target=directory/relative
        if target.exists():continue
        target.parent.mkdir(parents=True,exist_ok=True);copier.copyfile(source,target);restored+=1
    return dict(restored_files=restored,partial=manifest['partial'],missing=manifest['missing'],
                note='Resume still requires the same approved runtime session; otherwise use a new plan.')


def _artifact_manifest(plan, root, directory, snapshot, *, partial):
    from pilot_eval.sampling_sft_exports import inventory
    report=json.loads((snapshot/'final-report.json').read_text())
    paths=set();missing=['stage:'+name+':'+entry['state'] for name,entry in report['stages'].items() if entry['state']!='verified' and name not in REGENERATED]
    # Regenerated stages are never backed up, whichever export path would reach them.
    skipped=[]
    for stage in plan['spec']['stages']:
        if stage['id'] in REGENERATED:
            stage_plan=inside(root,stage['plan']);skipped.append(stage_plan)
            if stage_plan.is_file():skipped.append(inside(root,json.loads(stage_plan.read_text())['run_path']))
    def add(target,label):
        target=inside(root,target)
        if not target.exists():
            missing.append(label);return
        if target.is_symlink():raise ValueError('Export paths may not be symlinks')
        files=[target] if target.is_file() else target.rglob('*')
        for file in files:
            if file.is_symlink():raise ValueError('Export tree contains a symlink')
            if not file.is_file() or any(file==item or file.is_relative_to(item) for item in skipped):continue
            if file.is_relative_to(directory) and not file.is_relative_to(snapshot):continue
            if file.name in ['.lock','writer.json','bases.npz'] or file.name.endswith('.tmp'):continue
            if file.suffix=='.safetensors' and ('models/' in str(file.relative_to(root)) or 'weights/' in str(file.relative_to(root))):continue
            paths.add(str(file.relative_to(root)))
    for name in plan['spec']['export']['paths']:add(root/name,name)
    for stage in plan['spec']['stages']:
        if stage['id'] in REGENERATED:continue
        stage_plan=inside(root,stage['plan']);add(stage_plan,stage['id']+':plan')
        if not stage_plan.is_file():continue
        config=json.loads(stage_plan.read_text())
        if config.get('run_path'):add(root/config['run_path'],stage['id']+':run')
        if stage.get('report'):add(root/stage['report'],stage['id']+':report')
    for relative in plan['spec']['source_hashes']:add(root/relative,'source:'+relative)
    pool_stage=next(stage for stage in plan['spec']['stages'] if stage['id']=='pool')
    pool_plan=inside(root,pool_stage['plan'])
    if pool_plan.is_file():
        pool_config=json.loads(pool_plan.read_text())
        handoff=inside(root,pool_config['run_path'])/'handoff.json'
        if handoff.is_file():
            handoff_data=json.loads(handoff.read_text())
            rollout=inside(root,handoff_data['r0_path'])
            add(rollout,'pool rollout plan')
            if rollout.is_file():
                rollout_config=json.loads(rollout.read_text())
                if rollout_config.get('run_path'):add(root/rollout_config['run_path'],'pool rollout evidence')
                for field in ['audit_path','baseline_path','runtime_path']:
                    if rollout_config.get(field):
                        parent=inside(root,rollout_config[field]);add(parent,'pool rollout '+field)
                        if field=='audit_path' and parent.is_file():
                            audit=json.loads(parent.read_text())
                            if audit.get('run_path'):add(root/audit['run_path'],'pool source audit evidence')
                for parent in rollout_config.get('parents',{}):add(root/parent,'pool source:'+parent)
        elif not partial:missing.append('pool handoff')
    diagnostic=next(stage for stage in plan['spec']['stages'] if stage['id']=='diagnostics')
    setup=next(command for command in diagnostic['commands'] if command[0]=='offline-grpo-train-prepare')
    if setup.count('--checkpoint-zero')!=1:raise ValueError('Checkpoint-zero source is not explicitly bound')
    add(inside(root,setup[setup.index('--checkpoint-zero')+1]),'checkpoint-zero source')
    add(root/'plans'/plan['spec']['name']/'offline.overnight.json','overnight plan')
    add(root/plan['spec_path'],'overnight specification')
    add(snapshot,'wrapper snapshot')
    if missing and not partial:raise ValueError('Required export products missing: '+', '.join(missing))
    value=dict(schema_version=1,job_sha256=_hash(plan),files=inventory(root,sorted(paths)),
               worker_host=platform.node(),simulation=False,partial=partial,missing=missing,
               restore_requires_model_snapshot=True,wrapper_snapshot=str(snapshot.relative_to(root)),regenerated_on_resume=sorted(REGENERATED),external_sources=_external_sources(plan,root),
               scope='offline GRPO artifact snapshot; missing items are explicit; base model weights and external_sources are excluded and must be restored separately; regenerated_on_resume stages are rebuilt by a resumed run')
    value['sha256']=_hash(value)
    target=directory/'exports'/value['sha256']/'manifest.json'
    _save_frozen(target,value)
    return target


def _external_sources(plan, root):
    """Name verification inputs outside the artifact root, which an export cannot contain."""
    found=[]
    fields=dict(dataset=('sft_root','sft_path','sft_sha256'),cut_build=('reference_root','reference_path','reference_sha256'))
    for stage in plan['spec']['stages']:
        if stage['id'] not in fields:continue
        path=inside(root,stage['plan'])
        if not path.is_file():continue
        config=json.loads(path.read_text());base,relative,digest=fields[stage['id']]
        if config.get(base) and Path(config[base]).is_absolute():
            found.append(dict(stage=stage['id'],root=config[base],path=config[relative],sha256=config[digest]))
    return found


def _export(plan, root, directory, snapshot, *, partial, dependencies):
    from pilot_eval.sampling_sft_hf import upload
    manifest=_artifact_manifest(plan,root,directory,snapshot,partial=partial)
    record=upload(manifest,root,repo=plan['spec']['export']['repo'],client=dependencies.hub())
    value=json.loads(manifest.read_text())
    if record['manifest_sha256']!=value['sha256']:
        raise ValueError('HF receipt does not bind the export manifest')
    _fresh_readback(record,value,dependencies)
    return dict(record=record,manifest_path=str(manifest.relative_to(root)),partial=partial,external_sources=len(value['external_sources']))


def run(path, output_root, authorization_path, *, dependencies=None):
    import time
    import uuid
    from pilot_eval.stage0_qualification_backend import deadline_guard
    root=Path(output_root).resolve();plan,directory=_load(path,root);deps=dependencies or Dependencies()
    with _writer(directory/'execution'):
        preflight_path=directory/'meta/preflight.json'
        if not preflight_path.exists():raise ValueError('Run preflight before authorizing launch')
        prior=json.loads(preflight_path.read_text())
        if not prior['ok']:raise ValueError('Preflight has unresolved failures')
        authorization=_authorization(authorization_path,plan,prior,root)
        if deps.runtime()!=plan['spec']['stage0_runtime']:raise ValueError('Runtime changed after prelaunch approval')
        if deps.hub().info(plan['spec']['export']['repo']).private is not True:raise ValueError('Private destination changed')
        _save_frozen(directory/'meta/authorization.json',authorization)
        current=json.loads((directory/'meta/status.json').read_text())
        if current.get('state')=='completed' and current.get('export',{}).get('state')=='verified':
            verify(path,root,dependencies=deps)
            return current
        remaining=authorization.get('max_seconds',DEFAULT_MAX_SECONDS)
        reservation=dict(reservation_id='overnight-'+_hash(plan)[:16])
        attempt=directory/'attempts'/uuid.uuid4().hex
        _save_frozen(attempt/'start.json',dict(reservation_id=reservation['reservation_id'],plan_sha256=_hash(plan)))
        began=time.monotonic()
        state=dict(state='running',stages={},export=dict(state='not_started'))
        _write_state(directory,plan,state)
        active_stage=None
        def provider(child_config,execution,child_dir,child_root,runtime,approval_path):
            if Path(child_root).resolve()!=root or runtime!=plan['spec']['stage0_runtime'] or approval_path is not None:
                raise ValueError('Child runtime, root or separate approval differs from the overnight approval')
            if active_stage is None or state['stages'].get(active_stage,{}).get('state')!='running':
                raise ValueError('Child reservation is not bound to the active approved stage')
            left=remaining-(time.monotonic()-began)
            if left<=0:raise TimeoutError('Approved run time limit expired')
            return reservation,left
        def bounded(function, *args, **kwargs):
            left=remaining-(time.monotonic()-began)
            if left<=0:raise TimeoutError('Approved run time limit expired')
            with deadline_guard(left,plan['spec']['stage0_runtime']['execution_kind']=='real-cuda'):
                value=function(*args,**kwargs)
            if time.monotonic()-began>=remaining:
                raise TimeoutError('Approved run time limit expired')
            return value
        try:
            with deadline_guard(remaining,plan['spec']['stage0_runtime']['execution_kind']=='real-cuda'):
                for stage in plan['spec']['stages']:
                    identifier=stage['id']
                    if any(state['stages'].get(dep,{}).get('state')!='verified' for dep in stage['depends_on']):
                        state['stages'][identifier]=dict(state='blocked_dependency')
                        _write_state(directory,plan,state);continue
                    try:
                        try:
                            verified=bounded(_native_verify,stage,root)
                        except FileNotFoundError:
                            verified=False
                        if not verified:
                            state['stages'][identifier]=dict(state='running',log=str((directory/'logs'/f'{identifier}.log').relative_to(root)))
                            _write_state(directory,plan,state)
                            active_stage=identifier
                            for command in stage['commands']:
                                left=remaining-(time.monotonic()-began)
                                if left<=0:raise TimeoutError('Approved run time limit expired')
                                print(f"Offline overnight: {identifier}: {command[0]}",flush=True)
                                log_path=directory/'logs'/f'{identifier}.log'
                                with _tee_output(log_path), deadline_guard(left,plan['spec']['stage0_runtime']['execution_kind']=='real-cuda'):
                                    if deps.invoke(command,reservation_provider=provider)!=0:
                                        raise RuntimeError('Child command failed: '+command[0])
                            bounded(_native_verify,stage,root)
                        state['stages'][identifier]=dict(state='verified',log=str((directory/'logs'/f'{identifier}.log').relative_to(root)))
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        state['stages'][identifier]=dict(state='failed',error_type=type(exc).__name__,message=_safe_message(exc),log=str((directory/'logs'/f'{identifier}.log').relative_to(root)))
                    finally:
                        active_stage=None
                    _write_state(directory,plan,state)
                complete=all(state['stages'].get(name,{}).get('state')=='verified' for name in STAGES)
                state['state']='exporting' if complete else 'incomplete'
                _write_state(directory,plan,state)
                snapshot=bounded(_snapshot_report,plan,directory,root,state,attempt,time.monotonic()-began)
                try:
                    proof=bounded(_export,plan,root,directory,snapshot,partial=not complete,dependencies=deps)
                    record=proof['record']
                    state['export']=dict(state='verified' if complete else 'partial_verified',revision=record['revision'],
                                         manifest_sha256=record['manifest_sha256'],manifest_path=proof['manifest_path'],external_sources=proof.get('external_sources',0))
                    state['state']='completed' if complete else 'incomplete'
                except Exception as exc:
                    state['export']=dict(state='failed',error_type=type(exc).__name__,message=_safe_message(exc))
                    state['state']='incomplete'
                _write_state(directory,plan,state)
                return state
        except BaseException as exc:
            state['state']='interrupted'
            state['error_type']=type(exc).__name__
            state['message']=_safe_message(exc)
            _write_state(directory,plan,state)
            raise
        finally:
            _write_json(attempt/'timing.json',dict(seconds=time.monotonic()-began,
                reservation_id=reservation['reservation_id'],start_sha256=file_hash(attempt/'start.json'),
                final_integrity_sha256=None,usage_reconciled=False))
            _write_json(attempt/'timing.complete.json',dict(sha256=file_hash(attempt/'timing.json')))


def status(path,output_root):
    root=Path(output_root).resolve();plan,directory=_load(path,root)
    value=json.loads((directory/'meta/status.json').read_text())
    if value.get('config_sha256') not in [None,_hash(plan)]:raise ValueError('Overnight status identity changed')
    return value


def verify(path,output_root,*,dependencies=None):
    root=Path(output_root).resolve();plan,directory=_load(path,root);deps=dependencies or Dependencies()
    from pilot_eval.sampling_sft_exports import verify_manifest
    from pilot_eval.sampling_sft_hf import validate,private
    current=status(path,root)
    if current.get('state')!='completed' or current.get('export',{}).get('state')!='verified':
        raise ValueError('Completed verified export is required')
    for stage in plan['spec']['stages']:_native_verify(stage,root)
    manifest_path=inside(root,current['export']['manifest_path'])
    manifest=verify_manifest(manifest_path,root)
    receipt=json.loads((manifest_path.parent/'hf-upload.json').read_text())
    validate(receipt);private(deps.hub(),plan['spec']['export']['repo'])
    if manifest.get('partial') is not False or manifest.get('missing') or receipt['manifest_sha256']!=manifest['sha256'] or receipt['repo']!=plan['spec']['export']['repo']:
        raise ValueError('Export receipt differs from verified manifest')
    _fresh_readback(receipt,manifest,deps)
    return dict(complete=True,manifest_sha256=manifest['sha256'],revision=receipt['revision'])


def _fresh_readback(record, manifest, dependencies):
    """Freshly download an exact revision and validate every archived member."""
    import hashlib
    import tarfile
    import tempfile
    from pilot_eval.sampling_sft_exports import checked_relative
    with tempfile.TemporaryDirectory(prefix='offline-grpo-readback-') as temp:
        archive=Path(dependencies.fresh_fetch(record['repo'],record['archive'],record['revision'],Path(temp)))
        if not archive.resolve().is_relative_to(Path(temp).resolve()):
            raise ValueError('Readback must use the independent temporary cache')
        if file_hash(archive)!=record['archive_sha256']:
            raise ValueError('Fresh HF archive hash differs')
        expected=set(manifest['files']) | {record['manifest_path']}
        with tarfile.open(archive,'r:gz') as bundle:
            members=bundle.getmembers()
            if len(members)!=len(expected) or {m.name for m in members}!=expected:
                raise ValueError('Fresh HF archive membership differs from manifest')
            for member in members:
                checked_relative(member.name)
                if not member.isfile():raise ValueError('Archive contains a non-file member')
                stream=bundle.extractfile(member)
                if member.name==record['manifest_path']:
                    actual=json.load(stream)
                    if actual!=manifest:raise ValueError('Downloaded manifest differs')
                else:
                    digest=hashlib.sha256()
                    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                    entry=manifest['files'][member.name]
                    if digest.hexdigest()!=entry['sha256'] or member.size!=entry['bytes']:
                        raise ValueError('Downloaded archive member differs: '+member.name)


class _tee_output:
    """Keep native child progress visible while saving the same text to disk."""
    def __init__(self,path):
        self.path=path
    def __enter__(self):
        from contextlib import redirect_stdout,redirect_stderr
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('a',buffering=1)
        class Tee:
            def __init__(self,original,log):self.original,self.log=original,log
            def write(self,text):self.original.write(text);self.log.write(text);return len(text)
            def flush(self):self.original.flush();self.log.flush()
        self.stdout=redirect_stdout(Tee(sys.stdout,self.stream))
        self.stderr=redirect_stderr(Tee(sys.stderr,self.stream))
        self.stdout.__enter__();self.stderr.__enter__()
        return self
    def __exit__(self,*args):
        self.stderr.__exit__(*args);self.stdout.__exit__(*args);self.stream.close()


def template(inputs_path, output_root, name):
    """Create an editable full DAG from explicit local source and runtime inputs."""
    root=Path(output_root).resolve();safe_name(name)
    inputs=json.loads(inside(root,inputs_path).read_text())
    def need(key):
        value=inputs.get(key)
        if value is None or value=='':raise ValueError('Template input missing: '+key)
        return value
    def plan(label,filename):return str(root/'plans'/f'{name}-{label}'/filename)
    pool=plan('pool','offline.pool.json')
    fresh=str(root/'runs/offline-grpo'/f'{name}-pool'/'pool/fresh.plan.json')
    training=plan('training','offline.training.json')
    behaviour=plan('behaviour','offline.behaviour.json')
    dataset=plan('dataset','offline.dataset.json')
    internals=plan('internals','offline.internals.json')
    spectral=plan('spectral','stage0.spectral.json')
    cuts=plan('cuts','offline.cuts.json')
    common=['--output-root',str(root)]
    def cmd(operation,*args):return [operation,*common,*map(str,args)]
    commands={
        'pool':[cmd('offline-pool-inventory','--audit',need('audit'),'--baseline',need('historical_baseline'),'--name',f'{name}-pool'),
                cmd('offline-pool-prepare','--source',need('frozen'),'--name',f'{name}-pool'),
                cmd('offline-pool-route','--pool',pool,'--audit',need('audit'),'--baseline',need('historical_baseline'),'--weights-dir',need('weights_dir'),'--runtime',need('stage0_runtime_file')),
                cmd('offline-pool-fresh-prepare','--pool',pool,'--audit',need('audit'),'--weights-dir',need('weights_dir'),'--runtime',need('stage0_runtime_file')),
                cmd('offline-pool-fresh-run','--config',fresh),cmd('offline-pool-import','--config',pool,'--rollouts',fresh)],
        'spectral':[cmd('stage0-spectral-prepare','--audit',need('audit'),'--weights-dir',need('weights_dir'),'--name',f'{name}-spectral'),
                    cmd('stage0-spectral-run','--config',spectral)],
        'dataset':[cmd('offline-grpo-dataset-prepare','--pool',pool,'--sft',need('dataset_source'),'--comparison-root',need('dataset_root'),'--sampler',fresh,'--runtime',need('stage0_runtime_file'),'--name',f'{name}-dataset'),
                   cmd('offline-grpo-dataset-run','--config',dataset),cmd('offline-grpo-dataset-report','--config',dataset)],
        'diagnostics':[cmd('offline-grpo-train-prepare','--frozen',need('frozen'),'--pool',pool,'--checkpoint-zero',need('checkpoint_zero'),
                    '--checkpoint-zero-sha256',need('checkpoint_weights_sha256'),'--checkpoint-zero-config-sha256',need('checkpoint_config_sha256'),
                    '--runtime',need('training_runtime_file'),'--name',f'{name}-training'),
                    cmd('offline-grpo-train-check-diagnostics','--config',training,'--long-min-tokens',need('long_min_tokens'),'--prompt-count',need('prompt_count'),'--fidelity-tolerance',need('fidelity_tolerance_nats'))],
        'train':[cmd('offline-grpo-train-run','--config',training)],
        'behaviour':[cmd('offline-grpo-behaviour-prepare','--config',training,'--name',f'{name}-behaviour'),
                     cmd('offline-grpo-behaviour-run','--config',behaviour),
                     cmd('offline-grpo-behaviour-report','--config',behaviour,'--name',f'{name}-behaviour-report',*(['--online',inputs['online_behaviour']] if inputs.get('online_behaviour') else []))],
        'geometry':[cmd('offline-grpo-internals-prepare','--training',training,'--spectral',spectral,
                        '--inputs',need('fixed_inputs'),'--runtime',need('stage0_runtime_file'),'--qualification',need('qualification'),
                        '--name',f'{name}-internals',*(['--online',inputs['online_training']] if inputs.get('online_training') else [])),
                    cmd('offline-grpo-internals-geometry','--config',internals)],
        'push':[cmd('offline-grpo-internals-collect','--config',internals,'--kind','push')],
        'drift':[cmd('offline-grpo-internals-collect','--config',internals,'--kind','drift')],
        'internals_report':[cmd('offline-grpo-internals-report','--config',internals)],
        'cut_build':[cmd('offline-grpo-cuts-prepare','--internals',internals,'--behaviour',behaviour,'--reference',need('cut_reference'),
                         '--reference-root',need('cut_reference_root'),'--name',f'{name}-cuts'),
                     cmd('offline-grpo-cuts-build','--config',cuts)],
        'cut_agreement':[cmd('offline-grpo-cuts-agreement','--config',cuts)],
        'cut_evaluation':[cmd('offline-grpo-cuts-evaluate','--config',cuts)],
        'cut_report':[cmd('offline-grpo-cuts-report','--config',cuts)],
    }
    plans=dict(pool=pool,spectral=spectral,dataset=dataset,diagnostics=training,train=training,behaviour=behaviour,
               geometry=internals,push=internals,drift=internals,internals_report=internals,
               cut_build=cuts,cut_agreement=cuts,cut_evaluation=cuts,cut_report=cuts)
    stages=[]
    for identifier in STAGES:
        stage=dict(id=identifier,depends_on=DEPENDENCIES[identifier],plan=plans[identifier],commands=commands[identifier])
        if identifier=='pool':stage['additional_plans']=[fresh]
        if identifier=='behaviour':
            stage['report']=str(root/'reports'/f'{name}-behaviour-report')
            stage['additional_plans']=[training]
        stages.append(stage)
    spec=dict(protocol='offline-grpo-overnight-v1',name=name,stages=stages,
        stage0_runtime=json.loads(inside(root,need('stage0_runtime_file')).read_text()),
        training_runtime=json.loads(inside(root,need('training_runtime_file')).read_text()),
        model_identity=need('model_identity'),eos_token_id=need('eos_token_id'),
        snapshot_path=need('weights_dir'),snapshot_hashes=need('snapshot_hashes'),
        hf_cache_root=need('hf_cache_root'),minimum_cache_free_bytes=need('minimum_cache_free_bytes'),
        source_hashes=need('source_hashes'),software_versions=need('software_versions'),
        minimum_free_bytes=need('minimum_free_bytes'),export=dict(repo=need('hf_repo'),paths=need('export_paths')))
    _validate_spec(spec,root)
    output=root/'plans'/name/'offline.overnight.input.json'
    _save_frozen(output,spec)
    return output


class _offline_model_scope:
    """Scope HF offline settings despite libraries that cache environment flags."""
    def __enter__(self):
        from huggingface_hub import constants
        self.constants=constants
        self.old_flag=constants.HF_HUB_OFFLINE
        self.old_env={key:os.environ.get(key) for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']}
        self.hub=sys.modules.get('transformers.utils.hub')
        self.old_transformers=getattr(self.hub,'_is_offline_mode',None) if self.hub else None
        for key in self.old_env:os.environ[key]='1'
        constants.HF_HUB_OFFLINE=True
        if self.hub is not None:self.hub._is_offline_mode=True
        return self
    def __exit__(self,*args):
        self.constants.HF_HUB_OFFLINE=self.old_flag
        hub=sys.modules.get('transformers.utils.hub')
        if hub is not None:
            hub._is_offline_mode=self.old_transformers if self.hub is not None else self.old_flag
        for key,value in self.old_env.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def _verify_model_cache(spec,dependencies):
    cache=dependencies.cache_root().resolve()
    if cache!=Path(spec['hf_cache_root']).resolve():
        raise ValueError('HF runtime cache root differs from the approved location')
    if not cache.is_dir():raise ValueError('HF cache root is unavailable')
    free=shutil.disk_usage(cache).free
    if free<spec['minimum_cache_free_bytes']:raise ValueError('Insufficient free space in HF cache')
    mountinfo=Path('/proc/self/mountinfo')
    filesystem='unknown'
    if mountinfo.exists():
        matches=[]
        for line in mountinfo.read_text().splitlines():
            left,separator,right=line.partition(' - ')
            if not separator:continue
            fields=left.split();details=right.split()
            if len(fields)<5 or not details:continue
            mount=Path(fields[4].replace('\\040',' '))
            if cache.is_relative_to(mount):matches.append((len(str(mount)),details[0]))
        if matches:filesystem=max(matches)[1]
    if filesystem.startswith('fuse'):
        raise ValueError('HF cache must be on local storage, not a FUSE network mount')
    checked=[]
    for name,digest in spec['snapshot_hashes'].items():
        tokenizer=(name.startswith('tokenizer') or name in ['special_tokens_map.json','vocab.json','merges.txt'])
        revision=spec['model_identity']['tokenizer_revision' if tokenizer else 'model_revision']
        path=dependencies.cache_lookup(spec['model_identity']['model'],name,revision)
        if path is None or not path.is_file() or file_hash(path)!=digest:
            raise ValueError('Model loader cache file missing or differs: '+name)
        checked.append(name)
    return dict(root=str(cache),filesystem=filesystem,free_bytes=free,verified_files=checked,
                scope='same local cache lookup and revision used by the model loader; no download')
