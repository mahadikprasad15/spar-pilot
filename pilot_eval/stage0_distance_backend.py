"""One local untuned FP32 body forward per unique answer; bounded vocabulary head."""
import numpy as np

from pilot_eval.stage0_rollout_backend import RolloutDependencies,RolloutEngine


class DistanceDependencies(RolloutDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic scoring cannot launch CUDA.')
        return DistanceEngine(config,root)


class DistanceEngine(RolloutEngine):
    def __init__(self,config,root):
        super().__init__(dict(config,generation_settings=dict(seed=42)),root)
        self.body=self.backend.model.model;self.head=self.backend.model.get_output_embeddings()
        vocabulary,width=self.head.weight.shape
        if config['context_chunk']*(40*vocabulary+8*width)>config['workspace_bytes']:raise ValueError('Scoring output-head workspace exceeded.')

    def score(self,answer):
        torch=self.torch;ids=torch.tensor([answer['input_ids']],device='cuda:0',dtype=torch.long)
        if ids.shape[1]>self.backend.model.config.max_position_embeddings:raise ValueError('Teacher-forced context overflow; no truncation is allowed.')
        chunks={k:[] for k in ['surprise','entropy','rank']}
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            hidden=self.body(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
            for start in range(0,len(answer['positions']),self.config['context_chunk']):
                positions=answer['positions'][start:start+self.config['context_chunk']]
                target=torch.tensor(answer['target_ids'][start:start+len(positions)],device='cuda:0',dtype=torch.long)
                logits=self.head(hidden[0,positions])
                if logits.dtype!=torch.float32 or not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite/non-FP32 scoring logits.')
                lp=torch.log_softmax(logits.double(),dim=-1);at=torch.arange(len(target),device='cuda:0')
                chunks['surprise'].append((-lp[at,target]).cpu().numpy())
                chunks['entropy'].append((-(lp.exp()*lp).sum(-1)).cpu().numpy())
                chunks['rank'].append((1+(logits>logits[at,target,None]).sum(-1)).cpu().numpy())
        torch.cuda.synchronize()
        return {k:np.concatenate(v) for k,v in chunks.items()}
