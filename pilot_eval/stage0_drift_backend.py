"""Lazy FP32 fixed-context engine; bounded output head, actual LoRA-branch hooks."""
from contextlib import ExitStack, nullcontext

import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.activation_profile import AGREEMENT
from pilot_eval.stage0_drift_math import SUM_FIELDS, positions
from pilot_eval.stage0_evaluation_backend import EvaluationDependencies, EvaluationEngine
from pilot_eval.workflow import _hash


class DriftDependencies(EvaluationDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('CPU fixtures cannot launch real drift inference.')
        return DriftEngine(config,root)


class DriftEngine(EvaluationEngine):
    """Same checked local-only loader as greedy collection; no full [B,T,V] output."""
    def __init__(self,config,root):
        # Verify the largest allocation before paying for a model load.
        self.cache={};self.cache_group=None
        super().__init__(config,root)
        self.body=self.model.get_base_model().model
        self.model.config.use_cache=False
        modules=dict(self.model.named_modules())
        if any(m not in modules or not hasattr(modules[m],'base_layer') for m in config['modules']):
            raise ValueError('Actual adapted module inventory differs from the frozen mapping.')
        self.modules=[modules[m] for m in config['modules']]
        vocabulary=self.model.get_output_embeddings().weight.shape[0]
        width=self.model.get_output_embeddings().weight.shape[1]
        if config['context_chunk']*(96*vocabulary+8*width)>config['workspace_bytes']:
            raise ValueError('Declared output-head workspace exceeded before projection allocation.')

    def _finite(self,*values):
        if any(not bool(self.torch.isfinite(v).all()) for v in values):raise ValueError('Nonfinite drift tensor.')

    def _hidden(self,inputs,*,disabled=False):
        torch=self.torch
        with torch.inference_mode(),torch.autocast('cuda',enabled=False), (self.model.disable_adapter() if disabled else nullcontext()):
            result=self.body(**inputs,use_cache=False,return_dict=True).last_hidden_state
        self._finite(result)
        return result

    def _inputs(self,row):
        torch=self.torch
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0')
        mask=torch.ones_like(ids)
        return dict(input_ids=ids,attention_mask=mask,position_ids=mask.cumsum(-1)-1)

    def _kl(self,adapted,baseline,mapping,chunk):
        torch=self.torch;head=self.model.get_output_embeddings()
        vocabulary=head.weight.shape[0];width=head.weight.shape[1]
        if chunk*(96*vocabulary+8*width)>self.config['workspace_bytes']:
            raise ValueError('Output-head chunk exceeds declared workspace.')
        values=[];max_sum=0.
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            for start in range(0,len(mapping),chunk):
                targets=[p['context'] for p in mapping[start:start+chunk]]
                p,q=head(adapted[0,targets]),head(baseline[0,targets])
                if p.dtype!=torch.float32 or p.shape!=(len(targets),vocabulary) or q.shape!=p.shape:
                    raise ValueError('Full-vocabulary logit precision/shape mismatch.')
                self._finite(p,q)
                lp,lq=torch.log_softmax(p.double(),-1),torch.log_softmax(q.double(),-1)
                max_sum=max(max_sum,float((lp.abs()+lq.abs()).max()))
                kl=(lp.exp()*(lp-lq)).sum(-1);self._finite(kl)
                bound=64*torch.finfo(torch.float64).eps*(1+max_sum)
                if bool((kl < -bound).any()):raise ValueError('KL negative beyond FP64 arithmetic bound.')
                values.extend(kl.cpu().tolist())
        return np.asarray(values,dtype=np.float64),max_sum

    def _one(self,row,variant,*,capture_coefficients=False):
        torch=self.torch;inputs=self._inputs(row);key=_hash(row)
        if key not in self.cache:self.cache[key]=self._hidden(inputs,disabled=True)
        baseline=self.cache[key]
        if variant!='untuned':
            self.model.set_adapter(variant)
            if self.model.active_adapters!=[variant]:raise ValueError('Wrong active drift adapter.')
        self.model.eval()
        if not torch.equal(baseline,self._hidden(inputs,disabled=True)):
            raise ValueError('Disabled-adapter reference changed after switching.')
        mapping=positions([row]);sums=np.zeros((1,len(VIEWS),len(self.modules),3),dtype=np.float64)
        coefficients=[];seen=set();fraction=0.
        with ExitStack() as stack:
            for mi,module in enumerate(self.modules):
                captured={}
                def capture_base(m,args,output,c=captured):c['base']=output
                def capture_a(m,args,output,c=captured):c['coefficient']=output
                def capture_b(m,args,output,c=captured):c['branch']=output
                def capture_module(m,args,output,mi=mi,c=captured):
                    nonlocal fraction
                    if mi in seen:raise ValueError('Duplicate fixed-context module invocation.')
                    seen.add(mi);x=args[0];base=c.pop('base')
                    if x.shape[:2]!=inputs['input_ids'].shape or output.shape!=base.shape:
                        raise ValueError('Actual module/input token mapping differs.')
                    self._finite(x,base,output)
                    if variant=='untuned':
                        branch=torch.zeros_like(base)
                    else:
                        scalar=c.pop('coefficient');branch=c.pop('branch')*m.scaling[variant]
                        a=m.lora_A[variant].weight;b=m.lora_B[variant].weight
                        if a.shape[0]!=1 or b.shape[1]!=1 or m.scaling[variant]!=1:
                            raise ValueError('Drift hook requires audited rank-one alpha-one factors.')
                        self._finite(scalar,branch)
                        selected=[p['position'] for p in mapping['writes']]
                        if selected:
                            dot=(x[0,selected].double()*a[0].double()).sum(-1)
                            actual=scalar[0,selected,0].double()
                            frac=float(((dot-actual).abs()/(1e-6+1e-5*dot.abs())).max())
                            expected=scalar[0,selected].double()*b[:,0].double()*m.scaling[variant]
                            frac=max(frac,float(((branch[0,selected].double()-expected).abs()/(1e-6+1e-5*expected.abs())).max()))
                            fraction=max(fraction,frac)
                            if frac>1:raise ValueError('Signed coefficient/direct branch reconstruction failed.')
                            if capture_coefficients:coefficients.append(actual.cpu().numpy())
                    if not torch.equal(output,base+branch):
                        raise ValueError('Module output does not reconstruct from actual base and adapter branches.')
                    for vi,view in enumerate(VIEWS):
                        selected=[p['position'] for p in mapping['writes'] if p['view']==view]
                        if not selected:continue
                        bn=base[0,selected].double().norm(dim=-1);un=branch[0,selected].double().norm(dim=-1)
                        if bool((bn==0).any()):raise ValueError('Effective-write denominator is zero; ratio undefined.')
                        sums[0,vi,mi]=[float((un**2).sum()),float((bn**2).sum()),float((un/bn).sum())]
                for object_,hook in [(module.base_layer,capture_base),(module,capture_module)]:
                    handle=object_.register_forward_hook(hook);stack.callback(handle.remove)
                if variant!='untuned':
                    for object_,hook in [(module.lora_A[variant],capture_a),(module.lora_B[variant],capture_b)]:
                        handle=object_.register_forward_hook(hook);stack.callback(handle.remove)
            adapted=self._hidden(inputs,disabled=variant=='untuned')
        if len(seen)!=len(self.modules):raise ValueError('Missing fixed-context module hooks.')
        kl,max_sum=self._kl(adapted,baseline,mapping['predictions'],self.config['context_chunk'])
        if variant in self.config['controls']:
            if not torch.equal(adapted,baseline) or np.any(kl) or np.any(sums[...,[0,2]]):
                raise ValueError('Zero/no-operator control changes hidden states, KL or direct writes.')
        return dict(kl=kl,write_sums=sums,coefficients=coefficients,hidden=adapted,baseline=baseline,
                    mapping=mapping,reconstruction_max_fraction=fraction,max_abs_log_sum=max_sum)

    def measure(self,rows,variant,*,diagnostic):
        # Current real qualification is singleton. Save groups do not alter inference batching.
        if self.config['batch_size']!=1:raise ValueError('Real drift engine requires qualified singleton batches.')
        group=tuple(r['id'] for r in rows)
        if group!=self.cache_group:
            self.cache.clear();self.cache_group=group
        outputs=[];fraction=0.;max_sum=0.
        for i,row in enumerate(rows):
            value=self._one(row,variant,capture_coefficients=diagnostic and i==0)
            fraction=max(fraction,value['reconstruction_max_fraction']);max_sum=max(max_sum,value['max_abs_log_sum'])
            if diagnostic and i==0:
                repeat=self._one(row,variant,capture_coefficients=True)
                for key in ['kl','write_sums']:
                    if not np.allclose(value[key],repeat[key],**AGREEMENT):raise ValueError('Strict repeated drift measurement failed.')
                if len(value['coefficients'])!=len(repeat['coefficients']) or any(
                        not np.allclose(a,b,**AGREEMENT) for a,b in zip(value['coefficients'],repeat['coefficients'])):
                    raise ValueError('Strict repeated coefficient measurement failed.')
                chunk=max(1,self.config['context_chunk']//2)
                if chunk==self.config['context_chunk']:chunk+=1
                alt,alt_sum=self._kl(value['hidden'],value['baseline'],value['mapping']['predictions'],chunk)
                if not np.allclose(value['kl'],alt,**AGREEMENT):raise ValueError('Strict KL chunk agreement failed.')
                max_sum=max(max_sum,alt_sum,repeat['max_abs_log_sum'])
                fraction=max(fraction,repeat['reconstruction_max_fraction'])
            outputs.append(dict(kl=value['kl'],write_sums=value['write_sums']))
        # Check restoration again after all adapted measurements, including diagnostic repeats.
        for row in rows:
            if not self.torch.equal(self.cache[_hash(row)],self._hidden(self._inputs(row),disabled=True)):
                raise ValueError('Final disabled-adapter restoration failed.')
        mapping=positions(rows)
        return dict(arrays=dict(kl=np.concatenate([v['kl'] for v in outputs]),
                               write_sums=np.concatenate([v['write_sums'] for v in outputs])),
            metadata=dict(example_ids=[r['id'] for r in rows],**mapping,modules=self.config['modules'],variant=variant,
                kl_direction='variant||untuned',write_inputs='adapted-module-input',write_fields=SUM_FIELDS),
            validation=dict(reference_invariant=True,rank1_passed=True,finite=True,mapping_exact=True,
                reconstruction_max_fraction=fraction,diagnostic=diagnostic,repeat_passed=True if diagnostic else None,
                chunk_passed=True if diagnostic else None,context_chunk=self.config['context_chunk'],
                max_logit_contexts=min(self.config['context_chunk'],max(len(positions([r])['predictions']) for r in rows)),
                workspace_bytes=self.config['workspace_bytes'],max_abs_log_sum=max_sum,
                kl_roundoff_bound=64*np.finfo(float).eps*(1+max_sum)))

    def close(self):
        self.cache.clear();self.modules=[];self.body=None
        super().close()
