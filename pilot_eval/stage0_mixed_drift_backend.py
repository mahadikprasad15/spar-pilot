"""Fixed-context KL and effective writes for full mixed-operator edits."""
from contextlib import contextmanager,nullcontext

from pilot_eval.stage0_drift_backend import DriftEngine,DriftDependencies
from pilot_eval.stage0_merged_backend import MergedEvaluationEngine,merged_scope


class MixedDriftDependencies(DriftDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic inputs cannot launch real mixed drift.')
        return MixedDriftEngine(config,root)


class MixedDriftEngine(DriftEngine,MergedEvaluationEngine):
    @contextmanager
    def variant_scope(self,variant):
        definition=self.config['variants'].get(variant,{})
        self.original_weights={}
        if definition.get('representation')!='mixed-full-matrices':
            yield;return
        self.original_weights={self.model.get_submodule(name):self.model.get_submodule(name).base_layer.weight
                               for name in definition['adapter_disable_modules']}
        try:
            with merged_scope(self.model,definition,self.root,self.torch,adapter_name=variant,matrix_cache=self.matrix_cache(variant)):yield
        finally:self.original_weights={}

    def _module_write(self,m,x,output,base,c,variant,mapping,capture_coefficients):
        if m not in self.original_weights:
            return super()._module_write(m,x,output,base,c,variant,mapping,capture_coefficients)
        torch=self.torch
        if not m._disable_adapters or not torch.equal(output,base):
            raise ValueError('Merged module must use the saved full matrix with its adapter disabled.')
        original=self.original_weights[m]
        # Exact saved FP32 operators promoted to FP64; never infer an edit by
        # subtracting two rounded module outputs or pretending its rank is one.
        linear=torch.nn.functional.linear
        old=linear(x,original,m.base_layer.bias)
        delta=m.base_layer.weight.double()-original.double()
        branch=linear(x.double(),delta,None)
        self._finite(old,branch,output)
        if not torch.equal(base,linear(x,m.base_layer.weight,m.base_layer.bias)):
            raise ValueError('Actual mixed base-layer output differs from its full saved operator.')
        return old,branch,None,0.

    def measure(self,rows,variant,*,diagnostic):
        result=super().measure(rows,variant,diagnostic=diagnostic)
        result['validation']['operator_passed']=result['validation'].pop('rank1_passed')
        return result
