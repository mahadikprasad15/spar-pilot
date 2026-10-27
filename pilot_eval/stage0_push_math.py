"""Initial-gradient identities, loss normalization and gauge-aware push statistics."""
import hashlib
import numpy as np

from pilot_eval.grpo_algorithm import group_advantages


def a0_hash(factors):
    h=hashlib.sha256()
    for name,(a,b) in sorted(factors.items()):
        if np.any(b):raise ValueError('Initialization B must be exactly zero.')
        h.update(name.encode()+b'\0');h.update(str(a.shape).encode());h.update(a.astype('<f4').tobytes())
    return h.hexdigest()


def cosine(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float);den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(np.clip(a@b/den,-1,1))


def orient_endpoint(a,b,a0):
    dot=float(np.asarray(a,dtype=float).reshape(-1)@np.asarray(a0,dtype=float).reshape(-1))
    return (None if dot==0 else np.asarray(b,dtype=float).reshape(-1)*(1 if dot>0 else -1)),dict(dot_a64_a0=dot,flipped=dot<0,defined=dot!=0,a64_a0_cosine=cosine(np.asarray(a).reshape(-1),np.asarray(a0).reshape(-1)))


def summarize_vectors(gradients,tokens,counts):
    gradients=np.asarray(gradients,dtype=float);tokens=np.asarray(tokens,dtype=float);counts=np.asarray(counts)
    if gradients.ndim!=2 or len(gradients)!=len(tokens) or len(tokens)!=len(counts) or not np.isfinite(gradients).all() or np.any(tokens<=0):raise ValueError('Invalid batch gradient/denominator mapping.')
    pushes=-gradients;full=counts==8;mean=pushes.mean(0);pooled=np.average(pushes,axis=0,weights=tokens)
    full_mean=pushes[full].mean(0) if full.any() else None
    den=np.linalg.norm(pushes[full],axis=1).mean() if full.any() else 0.
    return dict(mean_push=mean.tolist(),mean_push_full_batches=None if full_mean is None else full_mean.tolist(),
        token_weighted_push=pooled.tolist(),consistency=None if den==0 else float(np.linalg.norm(full_mean)/den),
        full_batches=int(full.sum()),partial_batches=int((~full).sum()),batch_norms=np.linalg.norm(pushes,axis=1).tolist(),
        raw_norm_comparison_scope='descriptive within this condition only; cross-objective raw-norm comparisons forbidden')


def instrument_check():
    """Independent tiny CPU dense-softmax and directional finite-difference oracle."""
    import torch
    x=torch.tensor([[.3,-.7,1.],[1.,.2,-.5],[.4,.9,.3]],dtype=torch.float64)
    w=torch.tensor([[.2,.1,-.4],[-.3,.5,.6],[.7,-.2,.1],[.1,.2,.3]],dtype=torch.float64)
    a=torch.tensor([[.2,-.4,.6]],dtype=torch.float64,requires_grad=True)
    b=torch.zeros((4,1),dtype=torch.float64,requires_grad=True);target=torch.tensor([0,2,1]);weights=torch.tensor([1.,-1.,.5],dtype=torch.float64)
    def loss(value,selection=slice(None)):
        logits=x[selection]@(w+value@a).T
        return -(torch.log_softmax(logits,-1)[torch.arange(len(logits)),target[selection]]*weights[selection]).sum()/3
    value=loss(b);value.backward();actual=b.grad.detach().numpy().reshape(-1)
    z=x.numpy()@w.numpy().T;p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    delta=p.copy();delta[np.arange(3),target.numpy()]-=1;delta*=weights.numpy()[:,None]/3
    oracle=(delta.T@x.numpy()@a.detach().numpy().T).reshape(-1)
    relative=float(np.linalg.norm(actual-oracle)/np.linalg.norm(oracle));a_zero=bool(torch.equal(a.grad,torch.zeros_like(a.grad)))
    direction=torch.tensor([[.1],[-.3],[.4],[.2]],dtype=torch.float64);epsilon=1e-4
    numeric=float((loss(b.detach()+epsilon*direction)-loss(b.detach()-epsilon*direction))/(2*epsilon));analytic=float(actual@direction.numpy().reshape(-1))
    fd=abs(numeric-analytic)/abs(analytic)
    b.grad=None;a.grad=None
    for i in range(3):loss(b,slice(i,i+1)).backward()
    micro=bool(np.allclose(b.grad.detach().numpy().reshape(-1),oracle,atol=1e-12,rtol=1e-12))
    dead=not np.any(group_advantages([0]*8+[1]*8))
    passed=relative<1e-5 and fd<1e-3 and a_zero and micro and dead
    return dict(passed=passed,gradient_relative_error=relative,directional_relative_error=fd,epsilon=epsilon,
        a_gradient_exact_zero=a_zero,microbatch_global_denominator_passed=micro,dead_group_zero_passed=dead,
        scope='tiny independent CPU FP64 software oracle; not real-model/GPU qualification',reconstruction_gate=1e-5,directional_gate=1e-3)
