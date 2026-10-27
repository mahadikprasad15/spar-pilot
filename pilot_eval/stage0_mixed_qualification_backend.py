"""Bounded qualification of the exact saved conditional mixed-matrix loader."""
from contextlib import contextmanager

from pilot_eval.stage0_qualification_backend import QualificationDependencies,QualificationEngine
from pilot_eval.stage0_merged_backend import MergedEvaluationEngine,merged_scope


class MixedQualificationDependencies(QualificationDependencies):
    def engine(self,config,root):
        if config['fixture']:raise ValueError('Synthetic catalogue cannot authorize CUDA qualification.')
        return MixedQualificationEngine(config,root)


class MixedQualificationEngine(MergedEvaluationEngine,QualificationEngine):
    def __init__(self,config,root):
        actual=dict(config,variant_order=list(config['variants']),execution_kind='real-cuda',
                    decoding=dict(config['decoding'],use_cache=False))
        MergedEvaluationEngine.__init__(self,actual,root)

    def prepare_inputs(self,rows):return QualificationEngine.prepare_inputs(self,rows)

    @contextmanager
    def _mixed(self,reference,negative=False):
        definition=self.config['diagnostic_variants'][self._case]
        if reference!=definition['source_reference']:raise ValueError('Wrong original adapter for this diagnostic mask.')
        with merged_scope(self.model,definition,self.root,self.torch,
                          adapter_name=reference,negative_double_adapter=negative,matrix_cache=self.matrix_cache(reference)):yield

    def _teacher(self,row,arm,mode):
        return QualificationEngine._teacher(self,row,self.config['diagnostic_variants'][arm]['source_reference'],mode)

    def _generate(self,row,arm,mode):
        return QualificationEngine._generate(self,row,self.config['diagnostic_variants'][arm]['source_reference'],mode)

    def _next_logits(self,row,arm,mode,prefix):
        return QualificationEngine._next_logits(self,row,self.config['diagnostic_variants'][arm]['source_reference'],mode,prefix)

    def diagnose(self,row,arm,check_deadline):
        self._case=arm
        data=QualificationEngine.diagnose(self,row,arm,check_deadline)
        proof=self.check_integrity()
        data['integrity']=proof['base_unchanged'] and proof['adapters_unchanged']
        return data
