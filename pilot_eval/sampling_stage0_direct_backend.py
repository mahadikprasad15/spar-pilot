"""Direct-PEFT bounded smoke using the original deterministic FP32 engine."""
from pilot_eval.stage0_evaluation_backend import EvaluationDependencies, EvaluationEngine


class DirectDependencies(EvaluationDependencies):
    def engine(self, config, root):
        return DirectEngine(config, root)


class DirectEngine(EvaluationEngine):
    def smoke(self, rows):
        import hashlib
        torch=self.torch; results=[]
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            for arm in self.config['variant_order']:
                self.model.set_adapter(arm)
                for row in rows:
                    tokens=self.inputs[row['id']]
                    ids=torch.tensor([tokens],device='cuda:0',dtype=torch.long)
                    outputs=[]
                    for _ in range(2):
                        logits=self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1]
                        if logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all()):
                            raise ValueError('Direct PEFT smoke produced nonfinite or non-FP32 logits.')
                        outputs.append(logits.detach().cpu())
                    if not torch.equal(outputs[0],outputs[1]):
                        raise ValueError('Repeated same-runtime direct PEFT logits differ.')
                    results.append(dict(variant=arm,id=row['id'],vocabulary_size=len(outputs[0]),
                        logits_sha256=hashlib.sha256(outputs[0].numpy().tobytes()).hexdigest(), exact_repeat=True))
        torch.cuda.synchronize()
        return results
