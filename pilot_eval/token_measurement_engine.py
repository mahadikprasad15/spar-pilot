"""Additive rank-1 token capture and bounded, full-vocabulary paired output KL."""
from contextlib import nullcontext
import hashlib
from pathlib import Path
from pilot_eval.training import file_hash
import numpy as np
from pilot_eval.activation_prepare import VIEWS
from pilot_eval.token_measurement_math import prediction_positions


class TokenMeasurementEngine:
    """Reuse a validated instrument; do not retain full-sequence vocabulary logits."""
    def __init__(self,instrument,*,context_chunk,workspace_bytes,checkpoint_hashes=None):
        if (not isinstance(context_chunk,int) or isinstance(context_chunk,bool) or context_chunk<1
                or not isinstance(workspace_bytes,int) or workspace_bytes<1):
            raise ValueError('invalid output-head chunk/workspace budget')
        self.instrument=instrument
        self.torch=instrument.torch
        self.context_chunk=context_chunk
        self.workspace_bytes=workspace_bytes
        self.checkpoint_hashes=checkpoint_hashes

    def integrity_scope(self):return self.instrument.integrity_scope()
    def close(self):self.instrument.close()

    def _hidden(self,inputs,*,disabled=False,registrations=()):
        torch=self.torch;engine=self.instrument
        with torch.no_grad(),torch.autocast(next(engine.model.parameters()).device.type,enabled=False), \
                engine._hooks(registrations), (engine.model.disable_adapter() if disabled else nullcontext()):
            result=engine.body(**inputs,use_cache=False,return_dict=True).last_hidden_state
        engine._check_finite(result)
        if result.shape[:2]!=inputs['input_ids'].shape:raise ValueError('final hidden/input shape mismatch')
        return result

    def _source_hash(self,checkpoint):
        if self.checkpoint_hashes is None:return
        key=str(Path(checkpoint).resolve())
        if key not in self.checkpoint_hashes or file_hash(Path(checkpoint)/'adapter_model.safetensors')!=self.checkpoint_hashes[key]:
            raise ValueError('supplement checkpoint source hash differs from frozen evidence')

    def reference(self,rows):
        inputs,_=self.instrument._batch(rows)
        return dict(inputs=inputs,hidden=self._hidden(inputs,disabled=True))

    def release_reference(self,reference):
        reference.clear()

    def measure(self,checkpoint,rows,*,step,context_chunk=None,reference=None):
        engine=self.instrument;torch=self.torch
        self._source_hash(checkpoint)
        head=engine.model.get_output_embeddings()
        vocabulary=head.weight.shape[0]
        # Conservative live-output workspace: paired FP32 logits, FP64 logs,
        # exponent/difference/reduction temporaries and selected FP32 hidden rows.
        chunk=self.context_chunk if context_chunk is None else context_chunk
        if type(chunk)!=int or chunk<1:raise ValueError("invalid comparison context chunk")
        workspace=chunk*(96*vocabulary+8*head.weight.shape[1])
        if workspace>self.workspace_bytes:raise ValueError('declared output-head workspace budget exceeded before allocation')
        inputs,masks=engine._batch(rows)
        coefficients_positions=[]
        for example,row in enumerate(rows):
            for position in range(len(row['input_ids'])):
                views=[v for v in VIEWS if row['masks'][v][position]]
                if len(views)>1:raise ValueError('overlapping coefficient views')
                if views:coefficients_positions.append(dict(example=example,position=position,view=views[0],
                                                             token_id=row['input_ids'][position]))
        predictions=prediction_positions(rows)
        if reference is None:
            baseline=self._hidden(inputs,disabled=True)
        else:
            if set(reference['inputs'])!=set(inputs) or any(not torch.equal(reference['inputs'][k],inputs[k]) for k in inputs):
                raise ValueError('supplement cached reference/input alignment mismatch')
            baseline=reference['hidden']
        engine._switch(checkpoint)
        repeated=self._hidden(inputs,disabled=True)
        if not torch.equal(baseline,repeated):raise ValueError('disabled-adapter final reference changed after switch')
        del repeated
        engine._verify_base('during supplemental measurement')
        coefficients=np.empty((len(coefficients_positions),len(engine.modules)),dtype=np.float32)
        bi=[p['example'] for p in coefficients_positions];ti=[p['position'] for p in coefficients_positions]
        modules=[];registrations=[];seen=set();zero_contribution=True
        max_fraction=0.
        def digest(tensor):
            array=tensor.detach().cpu().numpy()
            return hashlib.sha256(str((array.shape,str(array.dtype))).encode()+array.tobytes()).hexdigest()
        for mi,(layer,projection,module) in enumerate(engine.modules):
            A,B=module.lora_A['default'].weight,module.lora_B['default'].weight
            modules.append(dict(layer=layer,projection=projection,A_sha256=digest(A),B_sha256=digest(B),
                scale=float(module.scaling['default']),A_norm=float(A.detach().double().norm()),B_norm=float(B.detach().double().norm())))
            captured={}
            def input_hook(m,args,output,captured=captured):captured.update(x=args[0],c=output)
            def branch_hook(m,args,output,captured=captured):captured['branch']=output
            def module_hook(m,args,output,mi=mi,captured=captured):
                nonlocal zero_contribution,max_fraction
                if mi in seen:raise ValueError('duplicate supplemental projection hook')
                seen.add(mi)
                x,c,branch=captured.pop('x'),captured.pop('c'),captured.pop('branch')
                if x.shape[:2]!=inputs['input_ids'].shape or c.shape!=(*x.shape[:2],1):
                    raise ValueError('coefficient token shape mismatch')
                engine._check_finite(c,branch,output)
                selected=c[bi,ti,0]
                coefficients[:,mi]=selected.detach().cpu().numpy()
                # Validate scalar inputs and branch reconstruction independently on
                # the instrument's bounded round-robin diagnostic positions.
                for view in VIEWS:
                    positions=engine._positions(masks[view])
                    if not positions:continue
                    b,t=zip(*positions);b=list(b);t=list(t)
                    dot=(x[b,t].double()*m.lora_A['default'].weight[0].double()).sum(-1)
                    scalar=c[b,t,0].double()
                    scalar_limit=1e-6+1e-5*dot.abs()
                    if ((dot-scalar).abs()>scalar_limit).any():raise ValueError('signed input coefficient reconstruction failed')
                    expected=c[b,t]*m.lora_B['default'].weight[:,0]*m.scaling['default']
                    actual=branch[b,t]*m.scaling['default']
                    limit=1e-6+1e-5*expected.abs()
                    fraction=float(((actual-expected).abs()/limit).max())
                    max_fraction=max(max_fraction,fraction)
                    if fraction>1:raise ValueError('rank-1 coefficient branch reconstruction failed')
                if step==0 and torch.count_nonzero(branch).item():
                    zero_contribution=False
                    raise ValueError('zero checkpoint has nonzero direct branch')
            registrations.extend([(module.lora_A['default'],'register_forward_hook',input_hook),
                                  (module.lora_B['default'],'register_forward_hook',branch_hook),
                                  (module,'register_forward_hook',module_hook)])
        adapted=self._hidden(inputs,registrations=registrations)
        if len(seen)!=len(engine.modules):raise ValueError('missing supplemental projection hooks')
        if step==0 and not torch.equal(baseline,adapted):raise ValueError('zero checkpoint changes final hidden states')
        values=np.empty(len(predictions),dtype=np.float64);max_logit_contexts=0
        with torch.no_grad(),torch.autocast(head.weight.device.type,enabled=False):
            for start in range(0,len(predictions),chunk):
                positions=predictions[start:start+chunk]
                b=[p['example'] for p in positions];t=[p['context'] for p in positions]
                # Output head sees only this bounded context chunk, never [B,T,V].
                p,q=head(adapted[b,t]),head(baseline[b,t])
                if p.shape!=(len(positions),vocabulary) or q.shape!=p.shape or p.dtype!=torch.float32:
                    raise ValueError('paired output-head vocabulary/precision mismatch')
                engine._check_finite(p,q)
                lp,lq=torch.log_softmax(p.double(),-1),torch.log_softmax(q.double(),-1)
                kl=(lp.exp()*(lp-lq)).sum(-1)
                # Analytic FP64 rounding bound; preserve tiny signed residuals,
                # rather than secretly clamping them or declaring damage.
                bound=64*torch.finfo(torch.float64).eps*(1+(lp.abs()+lq.abs()).max())
                if (kl < -bound).any():raise ValueError('KL negative beyond FP64 arithmetic bound')
                engine._check_finite(kl)
                values[start:start+len(positions)]=kl.cpu().numpy()
                max_logit_contexts=max(max_logit_contexts,len(positions))
                del p,q,lp,lq,kl
        if step==0 and np.count_nonzero(values):raise ValueError('identical checkpoint distributions have nonzero KL')
        engine._verify_base('after supplemental measurement')
        self._source_hash(checkpoint)
        return dict(arrays=dict(coefficients=coefficients,kl=values),metadata=dict(
            example_ids=[r['id'] for r in rows],coefficient_positions=coefficients_positions,
            predictions=predictions,modules=modules,step=step,
            coefficient_convention='raw-Ax-adapted-input',kl_direction='tuned||untuned',
            kl_units='nats',logit_precision='float32',reduction_precision='float64',
            zero_coefficient_note='A can be nonzero at zero initialization; B and branch must be zero.'),
            validation=dict(rank1_passed=True,reference_invariant=True,
                zero_contribution=zero_contribution if step==0 else None,
                module_count=len(seen),rank1_max_fraction=max_fraction,
                context_chunk=chunk,max_logit_contexts=max_logit_contexts,
                declared_workspace_bytes=self.workspace_bytes,bounded_workspace_estimate=workspace,
                rank1_thresholds=dict(atol=1e-6,rtol=1e-5),
                kl_negative_policy='preserve residual; fail below analytic FP64 bound'))
