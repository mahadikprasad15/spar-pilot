"""One local untuned FP32 body forward per unique answer; bounded vocabulary head."""
import numpy as np

from pilot_eval.stage0_numeric import require_eager_attention
from pilot_eval.stage0_rollout_backend import RolloutDependencies,RolloutEngine


class DistanceDependencies(RolloutDependencies):
    def engine(self,config,root):
        if config['execution_kind']!='real-cuda':raise ValueError('Synthetic scoring cannot launch CUDA.')
        return DistanceEngine(config,root)


class DistanceEngine(RolloutEngine):
    def __init__(self,config,root):
        super().__init__(dict(config,generation_settings=dict(seed=42)),root)
        require_eager_attention(self.backend.model)
        self.body=self.backend.model.model;self.head=self.backend.model.get_output_embeddings()
        vocabulary,width=self.head.weight.shape
        self.vocabulary_size=int(vocabulary)
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
                if logits.dtype!=torch.float32 or tuple(logits.shape)!=(len(target),self.vocabulary_size) or not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite/non-FP32 scoring logits.')
                full_logits=logits.double()
                lp=torch.log_softmax(full_logits,dim=-1);at=torch.arange(len(target),device='cuda:0')
                chunks['surprise'].append((-lp[at,target]).cpu().numpy())
                chunks['entropy'].append((-(lp.exp()*lp).sum(-1)).cpu().numpy())
                chunks['rank'].append((1+(full_logits>full_logits[at,target,None]).sum(-1)).cpu().numpy())
        torch.cuda.synchronize()
        return {k:np.concatenate(v) for k,v in chunks.items()}

    def vocabulary(self):
        tokenizer_size=len(self.backend.tokenizer)
        return dict(logit_vocabulary_size=self.vocabulary_size,tokenizer_vocabulary_size=tokenizer_size,
            padded_output_rows=max(0,self.vocabulary_size-tokenizer_size),includes_padded_rows=True,
            rank_and_softmax_same_full_vector=True,
            rank_definition='1 + count of strictly greater entries in the full FP64-promoted FP32 logit vector used for softmax')
