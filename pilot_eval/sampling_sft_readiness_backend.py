"""Bounded CPU checks using real autograd, plus the pinned SFTTrainer when present.

No download, GPU model, or full training arm exists in this worker. The kernel
diagnostic is deliberately weaker than T16 and can never certify readiness.
"""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


def rows(size):
    result=[]
    for i in range(size):
        prompt=[1]*(1+i%2);response=[2,4,5,6][:1+i%3]+[3]
        result.append(dict(id=f'tiny:{i}',input_ids=prompt+response,labels=[-100]*len(prompt)+response,prompt_tokens=len(prompt)))
    return result


def oracle_loss(model,examples):
    """Independent Python FP64 log-sum-exp over individual scored positions."""
    import torch
    terms=[]
    model.eval()
    with torch.inference_mode():
        for row in examples:
            output=model(input_ids=torch.tensor([row['input_ids']],dtype=torch.long),use_cache=False)
            for position,label in enumerate(row['labels']):
                if position==0 or label==-100:continue
                values=output.logits[0,position-1].detach().double().tolist()
                maximum=max(values)
                terms.append(maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))-values[label])
    return math.fsum(terms)/len(terms),len(terms)


def agree(actual,reference,gate):
    return math.isfinite(actual) and abs(actual-reference)<=gate['atol']+gate['rtol']*abs(reference)


def kernel(size,gate,opt):
    import torch
    from pilot_eval.sft_backend import CompletionCollator,completion_loss
    torch.manual_seed(42)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.table=torch.nn.Embedding(8,8)
            with torch.no_grad():self.table.weight.copy_(torch.arange(64,dtype=torch.float32).reshape(8,8)/17)
        def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.table(input_ids))
    model=Tiny();examples=rows(size);expected,count=oracle_loss(model,examples)
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],betas=(opt['beta1'],opt['beta2']),
        eps=opt['epsilon'],weight_decay=0,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    first=scheduler.last_epoch;before=[p.detach().clone() for p in model.parameters()]
    model.train();losses=[]
    for row in examples:
        batch=CompletionCollator(0)([row]);batch.pop('example_ids');labels=batch.pop('labels')
        value=completion_loss(model(**batch),labels,count);losses.append(float(value.detach()));value.backward()
    finite=all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    changed=any(not torch.equal(p,old) for p,old in zip(model.parameters(),before))
    actual=math.fsum(losses)
    return dict(batch_size=size,oracle_loss=expected,actual_loss=actual,absolute_error=abs(actual-expected),
        supervised_tokens=count,finite_gradients=finite,optimizer_steps=1,scheduler_steps=scheduler.last_epoch-first,
        parameters_changed=changed,passed=agree(actual,expected,gate) and finite and changed and scheduler.last_epoch-first==1,
        scope='production completion_loss/collator plus real CPU optimizer; not the full trainer')


def full(size,gate,config,directory):
    import torch
    from datasets import Dataset
    from torch.utils.data import SequentialSampler
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast,Trainer,TrainerCallback
    from trl import SFTConfig,SFTTrainer
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from pilot_eval.sft_backend import HFTrainingEngine,CompletionCollator,completion_loss,frozen_weight_hash
    torch.manual_seed(42);examples=rows(size)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=8,hidden_size=4,intermediate_size=8,num_hidden_layers=28,
        num_attention_heads=1,num_key_value_heads=1,max_position_embeddings=32,eos_token_id=3,pad_token_id=0,
        attention_dropout=0.0,attn_implementation='eager'))
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f't{i}':i for i in range(8)},unk_token='t7')),eos_token='t3',pad_token='t0',unk_token='t7')
    engine=HFTrainingEngine(config,examples,model=model,tokenizer=tokenizer)
    model=engine.model
    expected,count=oracle_loss(model,examples);base_before=frozen_weight_hash(model)
    components=[];denominators=[];seen=[];checks=dict(finite_gradients=False,optimizer_steps=0)
    originals=[p.detach().clone() for p in model.parameters() if p.requires_grad]

    def objective(outputs,labels,num_items_in_batch=None):
        value=completion_loss(outputs,labels,num_items_in_batch)
        components.append(float(value.detach()));denominators.append(int(num_items_in_batch) if num_items_in_batch is not None else None)
        return value

    class Ordered(SFTTrainer):
        def _get_train_sampler(self,train_dataset=None):return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            seen.extend(inputs.pop('example_ids'))
            return Trainer.compute_loss(self,model,inputs,return_outputs,num_items_in_batch)

    class Audit(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            checks['finite_gradients']=bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
        def on_optimizer_step(self,args,state,control,optimizer=None,**kwargs):
            if getattr(optimizer,'step_was_skipped',False):raise ValueError('Short-batch optimizer step was skipped.')
            checks['optimizer_steps']+=1

    opt=config['optimizer']
    args=SFTConfig(output_dir=str(directory),max_steps=1,per_device_train_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=opt['learning_rate'],lr_scheduler_type='constant',warmup_steps=0,optim=opt['name'],optim_args='foreach=False',
        adam_beta1=opt['beta1'],adam_beta2=opt['beta2'],adam_epsilon=opt['epsilon'],weight_decay=0.0,max_grad_norm=1.0,
        bf16=False,fp16=False,tf32=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        full_determinism=True,seed=42,data_seed=42,dataloader_num_workers=0,dataloader_pin_memory=False,
        dataloader_drop_last=False,group_by_length=False,remove_unused_columns=False,ignore_data_skip=False,
        logging_steps=1,logging_nan_inf_filter=False,save_strategy='no',report_to='none',packing=False,
        max_length=max(len(r['input_ids']) for r in examples),completion_only_loss=True,
        dataset_kwargs={'skip_prepare_dataset':True},shuffle_dataset=False,use_cpu=True)
    trainer=Ordered(model=model,args=args,train_dataset=Dataset.from_list(examples),processing_class=tokenizer,
        data_collator=CompletionCollator(0),compute_loss_func=objective,callbacks=[Audit()])
    trainer.create_optimizer_and_scheduler(1);first=trainer.lr_scheduler.last_epoch
    outcome=trainer.train();actual=float(outcome.training_loss)
    changed=any(not torch.equal(p,old) for p,old in zip((p for p in model.parameters() if p.requires_grad),originals))
    scheduler_steps=trainer.lr_scheduler.last_epoch-first
    equal=agree(actual,expected,gate) and agree(math.fsum(components),expected,gate)
    base_unchanged=frozen_weight_hash(model)==base_before
    result=dict(batch_size=size,oracle_loss=expected,actual_loss=actual,absolute_error=abs(actual-expected),
        component_loss_sum=math.fsum(components),component_error=abs(math.fsum(components)-expected),
        supervised_tokens=count,observed_denominators=denominators,finite_gradients=checks['finite_gradients'],
        optimizer_steps=checks['optimizer_steps'],scheduler_steps=scheduler_steps,trainer_global_step=trainer.state.global_step,
        observed_ids=seen,expected_ids=[r['id'] for r in examples],configured_accumulation=8,
        base_unchanged=base_unchanged,parameters_changed=changed,
        passed=equal and checks['finite_gradients'] and checks['optimizer_steps']==scheduler_steps==trainer.state.global_step==1
            and changed and base_unchanged and seen==[r['id'] for r in examples]
            and denominators==[count]*size,
        scope='real pinned SFTTrainer, tiny28-layer Qwen+rank1 adapter; one final partial step, no model download')
    engine.close()
    return result


def run(request,directory):
    from pilot_eval.sft_backend import PINS
    versions={}
    for name in PINS:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    info=dict(python=sys.executable,python_version=sys.version,platform=platform.platform(),versions=versions,
              mode=request['mode'],loss_gate=request['loss_gate'],request_sha256=None)
    try:import torch
    except (ImportError,OSError) as exc:
        return dict(**info,state='missing-check',error=f'Torch import failed: {exc}',kernel_passed=False,t16_passed=False,sizes=[])
    torch.set_num_threads(1)
    if request['mode']=='full' and any(versions[name] is None or versions[name].split('+')[0]!=pin for name,pin in PINS.items()):
        return dict(**info,state='missing-check',error='Full T16 requires the pinned trainer environment.',required_versions=PINS,
            kernel_passed=False,t16_passed=False,sizes=[])
    values=[]
    try:
        for size in [3,4,5,6,7]:
            value=(full(size,request['loss_gate'],request['gold_config'],directory/f'batch-{size}') if request['mode']=='full'
                else kernel(size,request['loss_gate'],request['gold_config']['optimizer']))
            values.append(value)
            print(f'CPU {request["mode"]} batch{size}: error={value["absolute_error"]:.9g}; passed={value["passed"]}',flush=True)
            if not value['passed']:
                return dict(**info,state='failed-check',sizes=values,kernel_passed=False,t16_passed=False,
                    error='Approved loss/gradient/scheduler gate failed; no tolerance change or duplication fallback.')
    except Exception as exc:
        return dict(**info,state='failed-check',sizes=values,kernel_passed=False,t16_passed=False,error=str(exc),
                    error_type=type(exc).__name__,traceback=traceback.format_exc())
    return dict(**info,state='passed' if request['mode']=='full' else 'kernel-passed-full-T16-pending',sizes=values,
        kernel_passed=True,t16_passed=request['mode']=='full')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    request=json.loads(args.request.read_text())
    value=run(request,args.result.parent)
    from pilot_eval.workflow import _hash
    from pilot_eval.sampling_sft import save
    value['request_sha256']=_hash(request)
    save(args.result,value)
    return 0 if value.get('kernel_passed') else 1


if __name__=='__main__':raise SystemExit(main())
