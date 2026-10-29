"""Gold-ordered stored-ID training exports and separate readiness receipts.

No full arm is trained here. Initialization loads an exact copy of gold A0/zeroB;
seed equality is never substituted for checkpoint identity.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_generation as generation
from pilot_eval import sampling_sft_review as review
from pilot_eval.run import _write_state
from pilot_eval.sampling_sft_sources import inside,verify_parents
from pilot_eval.sft import load_sft,safe_name
from pilot_eval.stage0_sources import factors
from pilot_eval.stage0_workflow import _writer
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash


LOSS_GATE=dict(atol=1e-6,rtol=1e-6,approved='2026-10-29; user approved loss-only gate')


def implementation():
    return {name:file_hash(Path(__file__).with_name(name)) for name in
        ['sampling_sft_training_data.py','sampling_sft_readiness_backend.py','sft_backend.py']}


def prepare(review_path,output_root,name):
    root,parent,review_folder=review.load(review_path,output_root);coverage=review.verify(review_path,root)
    _,gen,_=generation.load(root/parent['generation_plan'],root)
    audit=source.verify(root/gen['source_plan'],root)['audit'];safe_name(name)
    folder=root/json.loads((root/gen['source_plan']).read_text())['run_path']/'training-data'/name
    config=dict(protocol='sampling-sft-training-data-v1',name=name,review_plan=str(inside(root,review_path).relative_to(root)),
        generation_plan=parent['generation_plan'],source_plan=gen['source_plan'],gold_config=audit['training']['config_path'],
        gold_checkpoint0=audit['training']['checkpoints']['0']['path'],loss_gate=LOSS_GATE,
        parents={str(inside(root,review_path).relative_to(root)):file_hash(inside(root,review_path)),
            str((review_folder/'complete.json').relative_to(root)):file_hash(review_folder/'complete.json'),
            str((review_folder/'results/coverage.json').relative_to(root)):file_hash(review_folder/'results/coverage.json')},
        implementation=implementation(),run_path=str(folder.relative_to(root)),arms=[0,2,5],
        training_seed=42,training_arms_default=[0,5],training_authorized=False)
    path=root/'plans'/name/'sampling-sft.training-data.json'
    source.frozen(path,config);source.frozen(folder/'meta/run_manifest.json',dict(config_sha256=_hash(config)))
    return path


def load(path,output_root):
    root=Path(output_root).resolve();path=inside(root,path);config=json.loads(path.read_text());folder=inside(root,config['run_path'])
    review.verify(root/config['review_plan'],root)
    if (config.get('protocol')!='sampling-sft-training-data-v1' or config['implementation']!=implementation()
            or config['loss_gate']!=LOSS_GATE
            or json.loads((folder/'meta/run_manifest.json').read_text())!={'config_sha256':_hash(config)}):
        raise ValueError('Training export/readiness implementation or source plan changed.')
    verify_parents(root,config['parents'])
    return root,config,folder


def supervised_row(problem,attempt,eos):
    prompt=problem['scorer_prompt_ids'];response=attempt['response_ids']
    if (attempt['end_state']!='finished' or not prompt or not response or response[-1]!=eos
            or sum(i==eos for i in response)!=1
            or any(type(i) is not int or i<0 for i in prompt+response)):
        raise ValueError('Selected target must have stored integer IDs and exactly one final gold-matching end-turn; no text repair.')
    return dict(id=problem['source_id'],problem_id=problem['problem_id'],attempt_id=attempt['attempt_id'],
        input_ids=prompt+response,labels=[-100]*len(prompt)+response,prompt_tokens=len(prompt),
        response_tokens=len(response),supervised_end_turn_count=1,
        tokenization='original stored prompt IDs plus retained generated IDs; no decode/re-encode')


def expected(root,config):
    coverage=review.verify(root/config['review_plan'],root)
    if coverage['coverage_blocked']:raise ValueError('More than5 uncovered problems blocks this training handoff.')
    _,gen,_=generation.load(root/config['generation_plan'],root)
    gold,gold_rows,_=load_sft(root/config['gold_config'],root)
    eos=gold['source_config']['decoding']['eos_token_id']
    problems={p['source_id']:p for p in gen['problems']};gold_by_id={r['id']:r for r in gold_rows}
    arms={}
    for k in config['arms']:
        arm=coverage['arms'][str(k)];attempts={a['attempt_id']:a for a in arm['attempts']};rows=[]
        for batch in arm['batches']:
            if not batch['nonmissing_count']:raise ValueError('All64 optimizer steps must remain nonempty.')
            for slot in batch['slots']:
                if slot['missing']:continue
                p=problems[slot['source_id']];old=gold_by_id[slot['source_id']]
                if p['scorer_prompt_ids']!=old['input_ids'][:old['prompt_tokens']]:
                    raise ValueError('Sample/gold stored prompt or mask boundary mismatch.')
                row=supervised_row(p,attempts[slot['selected_attempt']],eos)
                row.update(step=batch['step'],slot=next(i for i,s in enumerate(batch['slots']) if s['source_id']==row['id']))
                rows.append(row)
        arms[str(k)]=dict(rows=rows,batches=arm['batches'],uncovered_ids=arm['uncovered_ids'],
            coverage_decision=arm['coverage_decision'])
    return gold,arms


def t15(root,config,folder):
    original=inside(root,config['gold_checkpoint0']);copy=folder/'initialization/checkpoint-0'
    for name in ['adapter_config.json','adapter_model.safetensors']:
        if file_hash(original/name)!=file_hash(copy/name):raise ValueError('T15 exported checkpoint0 differs from gold bytes.')
    left,right=factors(original),factors(copy)
    if set(left)!=set(right) or len(left)!=196:raise ValueError('T15 must cover exactly196 rank-one modules.')
    for module,(a,b) in left.items():
        c,d=right[module]
        if (a.shape!=c.shape or b.shape!=d.shape or a.dtype!=c.dtype or b.dtype!=d.dtype
                or a.tobytes()!=c.tobytes() or b.tobytes()!=d.tobytes()):
            raise ValueError('T15 A0/B factor bytes changed: '+module)
        if (b!=0).any():raise ValueError('T15 requires exact zero B.')
    return dict(passed=True,modules=196,a_bitwise_equal=True,b_bitwise_equal=True,b_exact_zero=True,
        source_checkpoint=str(original.relative_to(root)),
        payload_sha256={name:file_hash(copy/name) for name in ['adapter_config.json','adapter_model.safetensors']},
        initialization_strategy='load exported gold checkpoint0; never regenerate A0 from seed alone',
        scope='exported initialization payload identity; real model load/save identity is rechecked at training start',
        optimizer_resume_state=False)


def export(path,output_root):
    root,config,folder=load(path,output_root)
    if (folder/'export.complete.json').exists():return verify(path,root)
    with _writer(folder):
        gold,arms=expected(root,config)
        original=inside(root,config['gold_checkpoint0']);copy=folder/'initialization/checkpoint-0';copy.mkdir(parents=True,exist_ok=True)
        for name in ['adapter_config.json','adapter_model.safetensors']:
            if (copy/name).exists() and file_hash(copy/name)!=file_hash(original/name):
                raise ValueError('Existing exported initialization changed; do not overwrite it.')
            if not (copy/name).exists():shutil.copy2(original/name,copy/name)
        identity=t15(root,config,folder)
        for k,arm in arms.items():
            target=folder/f'arms/k{k}';target.mkdir(parents=True,exist_ok=True)
            generation.jsonl(target/'training.jsonl',arm['rows']);source.frozen(target/'batches.json',arm['batches'])
            source.frozen(target/'coverage.json',{key:arm[key] for key in ['uncovered_ids','coverage_decision']})
        result=dict(protocol='sampling-sft-training-export-v1',t15=identity,training_seed=42,hardware=gold['hardware'],
            dtype=gold['dtype'],tf32=False,optimizer=gold['optimizer'],adapter=gold['adapter'],
            checkpoint_steps=gold['checkpoint_steps'],optimizer_steps=64,microbatch_size=1,
            loss_reduction='sum shifted supervised token CE / all supervised tokens in this original step',
            t16_required_sizes=[3,4,5,6,7],training_authorized=False,
            arms={k:dict(examples=len(a['rows']),missing=len(a['uncovered_ids']),
                step_sizes=[b['nonmissing_count'] for b in a['batches']],
                max_length=max(len(r['input_ids']) for r in a['rows'])) for k,a in arms.items()},
            required_training_rechecks=['exact initialized model adapter vs exported checkpoint0','real L4 FP32 and gold-matching trainer config'],
            scope='pretokenized data and initialization only; no optimizer recovery state or trained arm')
        source.frozen(folder/'results/export.json',result)
        files={str(p.relative_to(folder)):file_hash(p) for d in ['arms','initialization','results'] for p in (folder/d).rglob('*')
               if p.is_file() and p.name!='readiness.json'}
        source.frozen(folder/'export.complete.json',dict(config_sha256=_hash(config),files=files))
        _write_state(folder,'exported-readiness-pending',0,1)
    return result


def verify(path,output_root):
    root,config,folder=load(path,output_root)
    seal=json.loads((folder/'export.complete.json').read_text())
    if seal['config_sha256']!=_hash(config):raise ValueError('Training export config identity changed.')
    for name,digest in seal['files'].items():
        if file_hash(inside(folder,name))!=digest:raise ValueError('Exported training/initialization payload changed.')
    gold,arms=expected(root,config);identity=t15(root,config,folder)
    for k,arm in arms.items():
        if (generation.read_lines(folder/f'arms/k{k}/training.jsonl')!=arm['rows']
                or json.loads((folder/f'arms/k{k}/batches.json').read_text())!=arm['batches']):
            raise ValueError('Training IDs/masks/step slots no longer reconcile.')
    result=json.loads((folder/'results/export.json').read_text())
    if result['t15']!=identity or result['optimizer']!=gold['optimizer'] or result['checkpoint_steps']!=[0,8,16,32,64]:
        raise ValueError('Training protocol/T15 export receipt changed.')
    return result


def validate_check(value,request,code):
    """Recompute acceptance from bound measurements, never trust a pass flag alone."""
    if (value.get('request_sha256')!=_hash(request) or value.get('mode')!=request['mode']
            or value.get('loss_gate')!=request['loss_gate']):
        raise ValueError('CPU readiness request identity changed.')
    if not value.get('kernel_passed') and not value.get('t16_passed'):
        return False,False
    sizes=value.get('sizes',[])
    if code!=0 or [r.get('batch_size') for r in sizes]!=[3,4,5,6,7]:
        raise ValueError('CPU readiness requires all five short-batch measurements and successful exit.')
    gate=request['loss_gate']
    for row in sizes:
        actual,reference=row['actual_loss'],row['oracle_loss']
        if (not all(math.isfinite(x) for x in [actual,reference,row['absolute_error']])
                or row['absolute_error']!=abs(actual-reference)
                or abs(actual-reference)>gate['atol']+gate['rtol']*abs(reference)
                or row.get('finite_gradients') is not True or row.get('parameters_changed') is not True
                or row.get('optimizer_steps')!=1 or row.get('scheduler_steps')!=1 or row.get('passed') is not True):
            raise ValueError('CPU readiness loss/gradient/step measurements failed.')
        if request['mode']=='full':
            count=row['supervised_tokens']
            if (row.get('trainer_global_step')!=1 or row.get('base_unchanged') is not True
                    or row.get('configured_accumulation')!=8
                    or row.get('observed_denominators')!=[count]*row['batch_size']
                    or row.get('observed_ids')!=[f'tiny:{i}' for i in range(row['batch_size'])]
                    or not math.isfinite(row['component_loss_sum'])
                    or abs(row['component_loss_sum']-reference)>gate['atol']+gate['rtol']*abs(reference)):
                raise ValueError('Full T16 trainer measurements failed.')
    full=request['mode']=='full' and value.get('t16_passed') is True
    if full:
        from pilot_eval.sft_backend import PINS
        if any((value.get('versions',{}).get(name) or '').split('+')[0]!=pin for name,pin in PINS.items()):
            raise ValueError('Full T16 requires the pinned trainer versions.')
    return full,value.get('kernel_passed') is True


def check(path,output_root,python=None,mode='full'):
    root,config,folder=load(path,output_root);exported=verify(path,root)
    if mode not in ['full','kernel']:raise ValueError('Choose full trainer or isolated kernel check.')
    interpreter=Path(python or sys.executable).resolve()
    if not interpreter.is_file():raise ValueError('Select an existing Python interpreter; no environment install is automatic.')
    with run_lock(folder):
        attempt=folder/'checks'/uuid.uuid4().hex;attempt.mkdir(parents=True)
        request=dict(mode=mode,loss_gate=config['loss_gate'],gold_config=json.loads((root/config['gold_config']).read_text()),
            config_sha256=_hash(config),export_sha256=file_hash(folder/'export.complete.json'))
        source.save(attempt/'request.json',request)
        command=[str(interpreter),'-m','pilot_eval.sampling_sft_readiness_backend',
            '--request',str(attempt/'request.json'),'--result',str(attempt/'result.json')]
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
             'WANDB_DISABLED':'true','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        with (attempt/'console.log').open('w') as log:
            try:
                child=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,
                    timeout=180,check=False)
                code=child.returncode
            except subprocess.TimeoutExpired:code=None
        if (attempt/'result.json').exists():value=json.loads((attempt/'result.json').read_text())
        else:value=dict(state='missing-check' if code!=0 else 'failed-check',t16_passed=False,kernel_passed=False,
            error='CPU readiness child did not produce a verified receipt.',returncode=code)
        if not (attempt/'result.json').exists():
            value.update(request_sha256=_hash(request),mode=mode,loss_gate=request['loss_gate'])
            source.save(attempt/'result.json',value)
        try:full,kernel=validate_check(value,request,code)
        except (ValueError,KeyError,TypeError) as exc:
            value=dict(state='failed-check',kernel_passed=False,t16_passed=False,error=str(exc),raw_check=value)
            full=kernel=False
        result=dict(state='ready' if full else 'kernel-checked-full-T16-pending' if kernel else value['state'],
            t15_passed=exported['t15']['passed'],t16_passed=full,kernel_passed=kernel,
            training_ready=full and exported['t15']['passed'],training_authorized=False,check=value,
            config_sha256=_hash(config),export_sha256=file_hash(folder/'export.complete.json'),
            check_request=str((attempt/'request.json').relative_to(root)),check_result=str((attempt/'result.json').relative_to(root)),
            returncode=code,
            scope='CPU trainer/data readiness only; gate approval and paid training remain separate')
        source.save(attempt/'readiness.json',result)
        source.save(folder/'results/readiness.json',result)
        _write_state(folder,result['state'],int(full),1)
        source.save(attempt/'complete.json',dict(config_sha256=_hash(config),
            files={p.name:file_hash(p) for p in attempt.iterdir() if p.is_file()}))
    return result


def status(path,output_root):
    root,config,folder=load(path,output_root)
    exported=(folder/'export.complete.json').exists()
    if exported:verify(path,root)
    result=None
    if (folder/'results/readiness.json').exists():
        result=json.loads((folder/'results/readiness.json').read_text())
        attempt=inside(root,result['check_request']).parent
        seal=json.loads((attempt/'complete.json').read_text())
        if (seal['config_sha256']!=_hash(config) or result['config_sha256']!=_hash(config)
                or result['export_sha256']!=file_hash(folder/'export.complete.json')):
            raise ValueError('Readiness receipt source identity changed.')
        for name,digest in seal['files'].items():
            if file_hash(inside(attempt,name))!=digest:raise ValueError('CPU readiness evidence changed.')
        if json.loads((attempt/'readiness.json').read_text())!=result:
            raise ValueError('Published readiness receipt differs from sealed check.')
        raw=json.loads((attempt/'result.json').read_text())
        request=json.loads((attempt/'request.json').read_text())
        if result['kernel_passed'] or result['t16_passed']:
            full,kernel=validate_check(raw,request,result['returncode'])
            if (result['t16_passed']!=full or result['kernel_passed']!=kernel
                    or result['training_ready']!=(full and result['t15_passed'])):
                raise ValueError('Published readiness flags disagree with measurements.')
    return dict(exported=exported,readiness=result,training_authorized=False)


def console(value):
    if 'arms' in value:return {key:value[key] for key in ['t15','hardware','optimizer_steps','checkpoint_steps','arms','training_authorized']}
    return value
