"""Explicit original-step training with the gold loss/optimizer primitives.

Short steps never borrow microbatches from the next step. This loop uses the
same token-summed CE, clipping, constant schedule and AdamW settings as gold;
full T16 remains a separate prerequisite, not inferred from this implementation.
"""
import csv
import json
import math
import random
import shutil
from pathlib import Path


def open_engine(gold,sequence_length):
    import torch
    import subprocess
    from pilot_eval.sft_backend import HFTrainingEngine
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    runtime=HFTrainingEngine.runtime(gold) # Before loading any model weights.
    if gold['hardware']!='L4' or 'L4' not in runtime['device']:
        raise ValueError('Endpoint training requires the audited gold L4 hardware.')
    runtime.update(execution_kind='real-cuda',dtype='float32',tf32=False,autocast=False,
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip(),
        optimizer='torch.optim.AdamW; foreach=False',step_policy='explicit original slots; microbatch1')
    from transformers import AutoConfig
    metadata=AutoConfig.from_pretrained(gold['model'],revision=gold['model_revision'],trust_remote_code=False)
    limit=metadata.max_position_embeddings
    if type(limit) is not int or sequence_length>limit:
        raise ValueError('Retained target exceeds pinned model context; no truncation allowed.')
    runtime.update(model_context_limit=limit,arm_max_length=sequence_length)
    engine=HFTrainingEngine(gold,[])
    return engine,runtime


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        if not torch.cuda.is_available():raise ValueError('CUDA recovery cannot be loaded on CPU.')
        torch.cuda.set_rng_state_all(value['cuda'])


def save_adapter(engine,folder):
    if hasattr(engine,'save_adapter'):engine.save_adapter(folder)
    else:engine.model.save_pretrained(str(folder),safe_serialization=True)


def load_adapter(engine,folder):
    if hasattr(engine,'load_adapter'):engine.load_adapter(folder)
    else:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(engine.model,load_file(str(folder/'adapter_model.safetensors')),adapter_name='default')


def assert_initialization(engine,initial,folder):
    from pilot_eval.stage0_sources import factors
    from pilot_eval import sampling_sft as source
    load_adapter(engine,initial)
    trial=folder/'initialization-loaded';trial.mkdir()
    save_adapter(engine,trial)
    expected,actual=factors(initial),factors(trial)
    if set(actual)!=set(expected) or len(actual)!=196:raise ValueError('Real loaded checkpoint0 module identity changed.')
    for name,(a,b) in expected.items():
        c,d=actual[name]
        if (a.dtype!=c.dtype or b.dtype!=d.dtype or a.shape!=c.shape or b.shape!=d.shape
                or a.tobytes()!=c.tobytes() or b.tobytes()!=d.tobytes() or (b!=0).any()):
            raise ValueError('Real initialized model does not equal exported gold A0/zeroB: '+name)
    source.frozen(folder/'results/initialization.json',dict(passed=True,modules=196,a_bitwise_equal=True,b_exact_zero=True))


def train(engine,rows,batches,gold,config,folder,root,initial,runtime,after_step=None):
    import torch
    from pilot_eval.sft_backend import completion_loss,CompletionCollator,frozen_weight_hash
    from pilot_eval import sampling_sft as source
    from pilot_eval import sampling_sft_generation as generation
    from pilot_eval.sampling_sft_training import checkpoint,seal,event,checkpoints,load
    from pilot_eval.run import _write_state
    from pilot_eval.workflow import _hash
    model=engine.model;device=next(model.parameters()).device
    if max(len(r['input_ids']) for r in rows)>runtime['model_context_limit']:
        raise ValueError('Retained target exceeds model context; never truncate or skip it.')
    opt=gold['optimizer'];plan_sha=_hash(config);runtime_sha=_hash(runtime)
    if not all(p.dtype==torch.float32 for p in model.parameters()):raise ValueError('Every model/adapter weight must be FP32.')
    if hasattr(model,'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if hasattr(engine,'assert_initialization'):engine.assert_initialization(initial,folder)
    else:assert_initialization(engine,initial,folder)
    base=frozen_weight_hash(model)
    if runtime['execution_kind']=='real-cuda' and base!=config['expected_base_sha256']:
        raise ValueError('Loaded frozen base differs from the audited gold source.')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters:raise ValueError('No trainable adapter parameters.')
    optimizer=torch.optim.AdamW(parameters,lr=opt['learning_rate'],betas=(opt['beta1'],opt['beta2']),
        eps=opt['epsilon'],weight_decay=0,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    records=[];start=0
    continuation=config['continuation']
    if continuation and continuation['kind']=='resume':
        _,old,oldfolder=load(root/continuation['plan'],root)
        old_runtime=json.loads((oldfolder/'meta/runtime.json').read_text())
        # Git metadata can change with documentation-only commits; numerical
        # runtime, package versions, hardware and implementation may not.
        if {k:v for k,v in old_runtime.items() if k!='git_commit'}!={k:v for k,v in runtime.items() if k!='git_commit'}:
            raise ValueError('Training recovery runtime changed; use an approved checkpoint0 restart.')
        cp=root/continuation['checkpoint'];checkpoint(cp,_hash(old),_hash(old_runtime),require_recovery=True)
        state=json.loads((cp/'trainer_state.json').read_text());start=state['global_step']
        if state['base_sha256']!=base or start!=continuation['step']:
            raise ValueError('Recovery base/progress differs from the frozen continuation.')
        # Loading pickle state is restricted to hash-verified locally produced
        # recovery files, never arbitrary external optimizer files.
        load_adapter(engine,cp)
        optimizer.load_state_dict(torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False))
        scheduler.load_state_dict(torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False))
        restore_rng(torch.load(cp/'rng_state.pth',map_location='cpu',weights_only=False))
        records=state['records']
        if [r['step'] for r in records]!=list(range(1,start+1)) or scheduler.last_epoch!=start:
            raise ValueError('Recovery optimizer/scheduler/log progress does not reconcile.')
        for step in sorted(checkpoints(oldfolder,old)):
            if step>start:continue
            src=oldfolder/'checkpoints'/f'checkpoint-{step}';target=folder/'checkpoints'/src.name
            checkpoint(src,_hash(old),_hash(old_runtime));target.mkdir(parents=True)
            for name in json.loads((src/'complete.json').read_text())['files']:shutil.copy2(src/name,target/name)
            value=json.loads((src/'complete.json').read_text())
            seal(target,step,value['recoverable'],plan_sha,runtime_sha)
        event(folder,'optimizer-recovered',step=start,source=continuation['checkpoint'])
    by_id={r['id']:r for r in rows}
    collator=CompletionCollator(engine.tokenizer.pad_token_id if getattr(engine,'tokenizer',None) else 0)
    model.train()

    def save_step(step):
        if step not in config['checkpoint_steps']:return
        target=folder/'checkpoints'/f'checkpoint-{step}';target.mkdir(parents=True)
        if step==0:
            for n in ['adapter_config.json','adapter_model.safetensors']:shutil.copy2(initial/n,target/n)
        else:save_adapter(engine,target)
        if config['capture_recovery']:
            torch.save(optimizer.state_dict(),target/'optimizer.pt')
            torch.save(scheduler.state_dict(),target/'scheduler.pt')
            torch.save(rng_state(),target/'rng_state.pth')
            source.save(target/'trainer_state.json',dict(global_step=step,records=records,base_sha256=base))
        if frozen_weight_hash(model)!=base:raise ValueError('Training modified a frozen base parameter.')
        seal(target,step,config['capture_recovery'],plan_sha,runtime_sha)

    if start==0 and not continuation:save_step(0)
    elif start==0 and continuation['kind']=='restart':save_step(0)
    log=folder/'logs/steps.jsonl';generation.jsonl(log,records)
    for batch in batches[start:]:
        step=batch['step'];selected=[by_id[s['source_id']] for s in batch['slots'] if not s['missing']]
        if not 3<=len(selected)<=8:raise ValueError('Training step size outside the approved 3–8 range.')
        denominator=sum(sum(v!=-100 for v in r['labels'][1:]) for r in selected)
        if denominator<=0:raise ValueError('Training step has no supervised tokens.')
        optimizer.zero_grad(set_to_none=True);losses=[]
        for row in selected:
            values=collator([row]);values.pop('example_ids');labels=values.pop('labels').to(device)
            output=model(**{k:v.to(device) for k,v in values.items()},use_cache=False)
            loss=completion_loss(output,labels,denominator);losses.append(float(loss.detach()));loss.backward()
            del output,loss
        gradients=[p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise ValueError('Missing/nonfinite training gradients.')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True))
        learning_rate=float(optimizer.param_groups[0]['lr'])
        optimizer.step();scheduler.step()
        if scheduler.last_epoch!=step or not all(torch.isfinite(p).all().item() for p in parameters):
            raise ValueError('Optimizer/scheduler progress or adapter finiteness failed.')
        record=dict(step=step,loss=math.fsum(losses),learning_rate=learning_rate,grad_norm=norm,
            example_ids=[r['id'] for r in selected],missing_slots=[s for s in batch['slots'] if s['missing']],
            supervised_tokens=denominator,optimizer_steps=step,scheduler_steps=scheduler.last_epoch)
        records.append(record);save_step(step);generation.jsonl(log,records)
        _write_state(folder,'running',step,64)
        print(f'Training k{config["k"]}: {step}/64 optimizer steps; {len(selected)} problems; loss={record["loss"]:.6g}',flush=True)
        if after_step:after_step(step)
    if len(records)!=64:raise ValueError('Training did not produce 64 successful step records.')
    with (folder/'logs/steps.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return dict(successful_steps=64,example_exposures=sum(len(r['example_ids']) for r in records),
        base_sha256_before=base,base_sha256_after=frozen_weight_hash(model),target_count=196,
        checkpoint_steps=config['checkpoint_steps'],initialization_verified=True,capture_recovery=config['capture_recovery'])
