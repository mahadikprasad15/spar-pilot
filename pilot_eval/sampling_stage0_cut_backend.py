"""One FP32 base model; switch and verify one PEFT cut adapter at a time."""
from pilot_eval.sampling_stage0_cuts import IntegrityFailure
from pilot_eval.stage0_batching_backend import BatchEngine
from pilot_eval.stage0_evaluation_backend import EvaluationEngine, verify_loaded_adapter
from pilot_eval.stage0_sources import inside


class CutEngine(EvaluationEngine):
    def __init__(self, config, root):
        first=config['variant_order'][0]
        narrow={**config,'variant_order':[first], 'variants':{first:config['variants'][first]}}
        super().__init__(narrow,root)
        self.config=config
        self.current=first

    def select(self, variant):
        if variant == self.current: return
        if variant not in self.config['variants']: raise ValueError('Unknown cut variant.')
        proof=self.check_integrity(full=True)
        if not all(proof.get(k) is True for k in ('base_unchanged','adapters_unchanged')):
            raise IntegrityFailure('Prior adapter or base changed before switching.')
        directory=inside(self.root,self.config['variants'][variant]['adapter_path'])
        self.model.load_adapter(directory,adapter_name=variant,is_trainable=False)
        self.model.set_adapter(variant)
        try:verify_loaded_adapter(self.model,variant,directory)
        except ValueError as exc:raise IntegrityFailure(str(exc)) from exc
        self.model.delete_adapter(self.current)
        self.current=variant
        self.adapter_guard=self._versions(adapters=True)

    def generate(self,rows,variant,seed):
        self.select(variant)
        return super().generate(rows,variant,seed)

    def check_integrity(self,*,full=False):
        same=self._versions()==self.guard
        adapters=self._versions(adapters=True)==self.adapter_guard
        if full:
            from pilot_eval.sft_backend import frozen_weight_hash
            same=same and frozen_weight_hash(self.model)==self.initial_base
            try:
                verify_loaded_adapter(self.model,self.current,
                    inside(self.root,self.config['variants'][self.current]['adapter_path']))
            except ValueError:
                adapters=False
        return dict(base_unchanged=same,adapters_unchanged=adapters,
                    kind='full-base-and-current-adapter' if full else 'parameter-version-guard')


class CutBatchEngine(CutEngine, BatchEngine):
    """The original Stage 0 batch-agreement measurements (ADR 0014) on the switched cut adapter."""
    def compare(self, rows, variant, token_rows):
        self.select(variant)
        return BatchEngine.compare(self, rows, variant, token_rows)

    def divergence(self, rows, variant, index, reference, responses):
        self.select(variant)
        return BatchEngine.divergence(self, rows, variant, index, reference, responses)
