"""Standalone CPU torch checks; no TRL or model download."""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from pilot_eval.offline_grpo_diagnostics import score_window, independent_push

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embed=torch.nn.Embedding(5,3); self.head=torch.nn.Linear(3,5,bias=False)
        for p in self.parameters(): p.requires_grad_(False)
        self.lora = torch.nn.Parameter(torch.zeros(3,5)); self.enabled=True
        self.config=SimpleNamespace(pad_token_id=None)
    @contextmanager
    def disable_adapter(self):
        old=self.enabled; self.enabled=False
        try: yield
        finally: self.enabled=old
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        x=self.embed(input_ids); y=self.head(x)
        if self.enabled: y=y+x@self.lora
        return SimpleNamespace(logits=y)

def run():
    torch.manual_seed(4); model=Model(); model.train(); model.head.eval()
    rows=[dict(id=f'q{i}',input_ids=[1,2],draws=[dict(draw_id=f'q{i}:draw:{j}',
        generated_token_ids=[3,4] if j%2 else [2],completion_mask=[1,1] if j%2 else [1]) for j in range(8)]) for i in range(8)]
    before=torch.get_rng_state().clone(); model.lora.grad=torch.ones_like(model.lora)
    result=score_window(model,rows,pad_token_id=0,batch_size=8)
    assert torch.equal(before,torch.get_rng_state()) and model.training and not model.head.training
    assert torch.equal(model.lora.grad,torch.ones_like(model.lora))
    assert result['summary']['sequence_mean']==0 and result['summary']['token_rate']==0
    assert result['generated_token_ids'][1]==[3,4] and len(result['log_mu'][1])==2
    original = model.forward
    def fail(*args, **kwargs):
        torch.rand(3)
        raise RuntimeError('fixture forward failure')
    model.forward = fail
    before = torch.get_rng_state().clone()
    try:
        score_window(model, rows, pad_token_id=0, batch_size=8)
    except RuntimeError:
        pass
    else:
        raise AssertionError('expected diagnostic failure')
    assert torch.equal(before,torch.get_rng_state()) and model.training and not model.head.training and model.enabled
    assert torch.equal(model.lora.grad,torch.ones_like(model.lora))
    model.forward = original
    prompts=[r['input_ids'] for r in rows]; draws=[d for r in rows for d in r['draws']]
    tokens=[d['generated_token_ids'] for d in draws]; advantages=[(-1.)**j for _ in rows for j in range(8)]
    push=independent_push(model,prompts,tokens,advantages,['lora'])
    assert torch.equal(model.lora.grad,torch.ones_like(model.lora))
    model.zero_grad(); losses=[]
    for i,(completion,advantage) in enumerate(zip(tokens,advantages)):
        full=torch.tensor([prompts[i//8]+completion]); logits=model(full).logits
        targets=torch.tensor(completion)
        logp=logits[0,len(prompts[i//8])-1:len(prompts[i//8])-1+len(completion)].log_softmax(-1)
        losses.append(-advantage*logp.gather(-1,targets[:,None]).sum())
    (sum(losses)/sum(map(len,tokens))).backward()
    assert torch.allclose(-push, model.lora.grad.flatten().double(),atol=1e-6,rtol=1e-4)

def gate_failure():
    import pilot_eval.offline_grpo_diagnostics as diagnostics
    from pilot_eval.offline_grpo_engine import OfflineGRPOTrainingEngine
    from pilot_eval.grpo_train_engine import GRPOTrainingEngine
    e=OfflineGRPOTrainingEngine.__new__(OfflineGRPOTrainingEngine)
    e.torch=torch; e.model=Model(); e.settings={'backward_groups':1}; e.diagnostic_sink=lambda *args: {'fixture':True}
    e.window=SimpleNamespace(backward=lambda *args,**kwargs:0., flat_gradient=lambda:torch.tensor([1.],dtype=torch.float64),
                             advantages=[0.]*64, parameter_names=['lora'], pad_id=0)
    rows=[dict(id=f'q{i}',prompt='p',gold='2',input_ids=[1],draws=[dict(group_id=f'q{i}',
        draw_id=f'q{i}:draw:{j}',draw_index=j,text='#### 2',token_ids=[1],generated_token_ids=[1],
        completion_mask=[1],stop_reason='eos',capped=False,reward_v3=1) for j in range(8)]) for i in range(8)]
    calls=[]; old_train=GRPOTrainingEngine.train_window; old_score=diagnostics.score_window; old_push=diagnostics.independent_push
    def parent(self,rows,step):
        self.window.backward([[1]]*8,[[1]]*64,[1]*64,capped=[False]*64)
        calls.append('optimizer-step')
        return [],{}
    try:
        GRPOTrainingEngine.train_window=parent
        diagnostics.score_window=lambda *args,**kwargs: {}
        diagnostics.independent_push=lambda *args,**kwargs: torch.tensor([0.],dtype=torch.float64)
        try: e.train_window(rows,1)
        except AssertionError: pass
        else: raise AssertionError('gradient mismatch did not fail')
        assert not calls
    finally:
        GRPOTrainingEngine.train_window=old_train; diagnostics.score_window=old_score; diagnostics.independent_push=old_push

if __name__=='__main__':
    run(); gate_failure()
