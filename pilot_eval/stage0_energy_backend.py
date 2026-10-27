"""All factors contract the same untuned inputs; one body forward per sequence."""
from contextlib import ExitStack

import numpy as np

from pilot_eval.activation_prepare import VIEWS
from pilot_eval.stage0_evaluation_backend import EvaluationDependencies,EvaluationEngine
from pilot_eval.stage0_sources import inside


class EnergyDependencies(EvaluationDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic fixtures cannot launch real activation collection.')
        return EnergyEngine(config,root)


def torch_view_sums(x,a,parallel,row):
    """Device-side FP64 contractions and reductions; total is independently dotted."""
    import torch
    if x.shape!=(len(row['input_ids']),a.shape[1]):raise ValueError('Actual module-input mapping differs.')
    result=torch.zeros((len(a),parallel.shape[1],len(VIEWS),4),device=x.device,dtype=torch.float64)
    flat=parallel.reshape(-1,a.shape[1]);rest=(a[:,None,:]-parallel).reshape_as(flat)
    for vi,view in enumerate(VIEWS):
        indices=[i for i,counted in enumerate(row['masks'][view]) if counted]
        if not indices:continue
        selected=x[indices].double();p=(selected@flat.T).reshape(len(indices),len(a),parallel.shape[1])
        r=(selected@rest.T).reshape_as(p);total=selected@a.T
        result[:,:,vi,0]=(p*p).sum(0);result[:,:,vi,1]=(r*r).sum(0)
        result[:,:,vi,2]=2*(p*r).sum(0);result[:,:,vi,3]=(total*total).sum(0)[:,None]
    if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite input-energy contraction.')
    return result.cpu().numpy()


def capture_body(model,body,modules,factors,rows,*,device,context_limit):
    """Shared real/tiny-model capture seam; counts only the requested body calls."""
    import torch
    outputs=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=False),model.disable_adapter():
        for row in rows:
            ids=torch.tensor([row['input_ids']],device=device,dtype=torch.long);mask=torch.ones_like(ids)
            if ids.shape[1]>context_limit:raise ValueError('Input-energy context overflow; no truncation allowed.')
            captured={}
            with ExitStack() as stack:
                for mi,module in enumerate(modules):
                    def capture(_module,args,mi=mi):
                        if mi in captured:raise ValueError('Duplicate energy module invocation.')
                        x=args[0]
                        if x.dtype!=torch.float32 or x.shape[:2]!=ids.shape:raise ValueError('Actual untuned input precision/token mapping differs.')
                        captured[mi]=torch_view_sums(x[0],*factors[mi],row)
                    handle=module.register_forward_pre_hook(capture);stack.callback(handle.remove)
                body(input_ids=ids,attention_mask=mask,position_ids=mask.cumsum(-1)-1,use_cache=False,return_dict=True)
            if set(captured)!=set(range(len(modules))):raise ValueError('Missing base-input hooks.')
            outputs.append(np.stack([captured[i] for i in range(len(modules))],axis=-2))
    return dict(sums=np.stack(outputs)),dict(base_input_scope=True,sequence_forwards=len(rows),all_factors_shared_inputs=True,hook_calls_per_module=len(rows))


class EnergyEngine(EvaluationEngine):
    def __init__(self,config,root):
        super().__init__(config,root);self.body=self.model.get_base_model().model
        self.model.config.use_cache=False;self.modules=[];self.factors=[]
        for index,name in enumerate(config['modules']):
            module=self.model.get_submodule(name)
            if not hasattr(module,'base_layer'):raise ValueError('Energy capture requires the actual audited adapted module.')
            self.modules.append(module)
            with np.load(inside(root,config['run_path'])/config['factor_files'][str(index)]['path'],allow_pickle=False) as arrays:
                self.factors.append(tuple(self.torch.tensor(arrays[k],device='cuda:0',dtype=self.torch.float64) for k in ['a','parallel']))

    def measure(self,rows):
        result=capture_body(self.model,self.body,self.modules,self.factors,rows,device='cuda:0',context_limit=self.model.config.max_position_embeddings)
        self.torch.cuda.synchronize();return result
