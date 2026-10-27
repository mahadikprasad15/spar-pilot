"""Lazy FP32 initial-gradient collector; no optimizer is constructed or stepped."""
import numpy as np

from pilot_eval.stage0_evaluation_backend import EvaluationDependencies, EvaluationEngine
from pilot_eval.stage0_push_math import a0_hash
from pilot_eval.stage0_sources import factors, inside


class PushDependencies(EvaluationDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic fixtures cannot launch real gradient collection.')
        return PushEngine(config,root)


def collect(model,body,head,module_factors,unit,*,context_chunk,device):
    """One sequence per microbatch, one global denominator, bounded head graphs.

    At B=0 the initial DAPO ratio has derivative grad(log p). Its old-policy
    value and the group advantages are detached. CE uses the same token sum.
    Output-gradient hooks reconstruct G@A0 independently in FP64 without saving
    the enormous dense G matrices. Head checkpointing bounds vocabulary memory.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    model.zero_grad(set_to_none=True);handles=[];reconstructed={};seen={}
    for name,(module,a,b) in module_factors.items():
        if bool(torch.any(b.detach()!=0)):raise ValueError('Initial B must be exactly zero.')
        reconstructed[name]=np.zeros(b.numel(),dtype=np.float64);seen[name]=0
        def forward_hook(layer,args,output,*,key=name,a0=a.detach()):
            # A0 is rank one, alpha/r=1. The scientific source audit enforces it.
            coefficient=(args[0].detach().double()*a0.double()).sum(-1)
            def backward_hook(delta):
                value=(delta.detach().double()*coefficient[...,None]).reshape(-1,delta.shape[-1]).sum(0)
                reconstructed[key][:]+=value.cpu().numpy();seen[key]+=1
            if output.requires_grad:output.register_hook(backward_hook)
        handles.append(module.register_forward_hook(forward_hook))
    active=0
    try:
        for sequence in unit['sequences']:
            advantage=sequence['advantage']
            if advantage==0:continue  # denominator still includes these tokens
            active+=1;ids=torch.tensor([sequence['input_ids']],device=device,dtype=torch.long)
            hidden=body(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
            total=None
            for start in range(0,len(sequence['positions']),context_chunk):
                positions=sequence['positions'][start:start+context_chunk]
                target=torch.tensor(sequence['target_ids'][start:start+len(positions)],device=device,dtype=torch.long)
                def token_loss(value,targets):
                    logits=head(value)
                    if logits.dtype!=torch.float32 or not bool(torch.isfinite(logits).all()):raise ValueError('Finite FP32 full-vocabulary logits required.')
                    logp=torch.log_softmax(logits,-1).gather(-1,targets[:,None]).squeeze(-1)
                    terms=(logp-logp.detach()).exp() if unit['objective']=='grpo' else logp
                    return -terms.sum()*advantage/unit['token_count']
                loss=checkpoint(token_loss,hidden[0,positions],target,use_reentrant=False)
                total=loss if total is None else total+loss
            if total is None or not bool(torch.isfinite(total)):raise ValueError('Invalid initial loss.')
            total.backward()
    finally:
        for handle in handles:handle.remove()
    arrays={};errors={}
    for index,(name,(_module,a,b)) in enumerate(module_factors.items()):
        if active and (a.grad is None or b.grad is None or seen[name]!=active):raise ValueError('Missing gradient or reconstruction hook: '+name)
        ga=np.zeros(tuple(a.shape),dtype=np.float32) if a.grad is None else a.grad.detach().cpu().numpy()
        gb=np.zeros(tuple(b.shape),dtype=np.float32) if b.grad is None else b.grad.detach().cpu().numpy()
        oracle=reconstructed[name].reshape(gb.shape);den=float(np.linalg.norm(oracle))
        numerator=float(np.linalg.norm(gb.astype(np.float64)-oracle))
        error=(0. if numerator==0 else None) if den==0 else numerator/den
        if not np.isfinite(ga).all() or not np.isfinite(gb).all() or np.any(ga):raise ValueError('A gradient must be finite and exactly zero at B=0: '+name)
        if error is None or not error<1e-5:raise ValueError('Initial G@A0 reconstruction gate failed: '+name+': '+str(error))
        arrays.update({f'gradient_b_{index}':gb,f'gradient_a_{index}':ga,f'push_{index}':-gb,f'reconstruction_{index}':oracle})
        errors[name]=error
    if any(p.grad is not None for n,p in model.named_parameters() if 'lora_' not in n and p.requires_grad is False):
        raise ValueError('A frozen base parameter received a gradient.')
    return arrays,dict(a_gradient_exact_zero=True,reconstruction_relative_errors=errors,
        reconstruction_gate=1e-5,active_sequences=active,zero_advantage_sequences=len(unit['sequences'])-active,
        global_token_denominator=unit['token_count'],optimizer_steps=0,advantages_detached=True)


class PushEngine(EvaluationEngine):
    def __init__(self,config,root):
        super().__init__(config,root)
        self.model.config.use_cache=False
        self.body=self.model.get_base_model().model;self.head=self.model.get_output_embeddings()
        self.source={arm:factors(inside(root,entry['adapter_path'])) for arm,entry in config['variants'].items()}
        for case in config['cases'].values():
            if a0_hash(self.source[case['initialization_arm']])!=case['a0_sha256']:raise ValueError('Loaded initialization hash differs.')

    def measure(self,unit):
        case=self.config['cases'][unit['case']];arm=case['initialization_arm'];self.model.set_adapter(arm);self.model.eval()
        for p in self.model.parameters():p.requires_grad_(False)
        selected={}
        for name in self.config['modules']:
            module=self.model.get_submodule(name);a=module.lora_A[arm].weight;b=module.lora_B[arm].weight
            a.requires_grad_(True);b.requires_grad_(True);selected[name]=(module,a,b)
        # Real inventories include every module; fixture engines use a separate backend.
        device=next(self.model.parameters()).device
        if any(len(s['input_ids'])>self.model.config.max_position_embeddings for s in unit['sequences']):raise ValueError('Context overflow; no gradient sequence truncation allowed.')
        with self.torch.autocast('cuda',enabled=False):
            result=collect(self.model,self.body,self.head,selected,unit,context_chunk=self.config['context_chunk'],device=device)
        self.torch.cuda.synchronize();return result
