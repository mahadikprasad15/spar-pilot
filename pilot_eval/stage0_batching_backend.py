"""Bounded batch-versus-singleton checks on the unchanged FP32 decoder."""
from pilot_eval.stage0_evaluation_backend import EvaluationDependencies, EvaluationEngine


class BatchDependencies(EvaluationDependencies):
    def engine(self, config, root):
        return BatchEngine(config, root)


class BatchEngine(EvaluationEngine):
    def _packed(self, sequences):
        torch=self.torch;pad=self.config['decoding']['pad_token_id']
        width=max(map(len,sequences))
        ids=torch.tensor([[pad]*(width-len(s))+s for s in sequences],device='cuda:0')
        mask=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences],device='cuda:0')
        return dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0))

    def _body(self, inputs, **kwargs):
        return self.model.get_base_model().model(**inputs,return_dict=True,**kwargs)

    def compare(self, rows, variant, token_rows):
        torch=self.torch;by_id={r['id']:r for r in token_rows}
        data=[by_id[r['id']] for r in rows]
        # The final scored context predicts EOS. Do not include that target EOS
        # in the decoder input; all answer-prediction positions remain present.
        sequences=[r['input_ids'][:max(r['answer_positions'])+1] for r in data]
        inputs=self._packed(sequences);width=inputs['input_ids'].shape[1]
        self.model.set_adapter(variant);self.model.eval();head=self.model.get_output_embeddings()
        examples=[]
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            batched=self._body(inputs,use_cache=False).last_hidden_state
            prefix={k:v[:,:-1] for k,v in inputs.items()}
            cached=self._body(prefix,use_cache=True)
            step=self._body(dict(input_ids=inputs['input_ids'][:,-1:],attention_mask=inputs['attention_mask'],
                position_ids=inputs['position_ids'][:,-1:]),past_key_values=cached.past_key_values,use_cache=True).last_hidden_state
            for i,(row,sequence) in enumerate(zip(data,sequences)):
                single=self._body(self._packed([sequence]),use_cache=False).last_hidden_state
                offset=width-len(sequence);errors=[];positions=[];worst=[]
                for start in range(0,len(row['answer_positions']),32):
                    selected=row['answer_positions'][start:start+32]
                    a=head(single[0,selected]);b=head(batched[i,[p+offset for p in selected]])
                    if a.dtype!=torch.float32 or b.dtype!=torch.float32:
                        raise ValueError('Behavioral agreement must use the actual FP32 generation head.')
                    delta=(a.double()-b.double()).abs()
                    if not bool(torch.isfinite(delta).all()):raise ValueError('Nonfinite batch logit error.')
                    values,indices=delta.max(-1);errors.extend(values.cpu().tolist());positions.extend(selected)
                    for j,index in enumerate(indices.cpu().tolist()):
                        worst.append(dict(position=selected[j],vocabulary_row=index,
                            singleton=float(a[j,index]),batched=float(b[j,index])))
                a=head(batched[i,-1]);b=head(step[i,-1]);delta=(a.double()-b.double()).abs()
                if not bool(torch.isfinite(delta).all()):raise ValueError('Nonfinite cached-decode logit error.')
                index=int(delta.argmax());errors.append(float(delta[index]));positions.append(len(sequence)-1)
                worst.append(dict(position=len(sequence)-1,vocabulary_row=index,kind='cached-decode',
                    full=float(a[index]),cached=float(b[index])))
                examples.append(dict(id=row['id'],errors=errors,positions=positions,worst_coordinates=worst))
        torch.cuda.synchronize()
        return dict(max_absolute_error=max(e for r in examples for e in r['errors']),examples=examples)

    def divergence(self, rows, variant, index, reference, responses):
        """Check the first divergent decision at an identical supplied prefix."""
        torch=self.torch;actual=responses[index]['generated_token_ids']
        common=next((k for k,(a,b) in enumerate(zip(reference,actual)) if a!=b),min(len(reference),len(actual)))
        if common>=len(reference) or common>=len(actual):return dict(near_tie=False,reason='length-only mismatch')
        target=rows[index]['id'];self.model.set_adapter(variant)
        prefix=self.inputs[target]+reference[:common]
        width=max(len(self.inputs[r['id']]) for r in rows)
        pad=self.config['decoding']['pad_token_id'];ids=[];masks=[]
        for row,response in zip(rows,responses):
            prompt=self.inputs[row['id']];continuation=response['generated_token_ids'][:common]
            ids.append([pad]*(width-len(prompt))+prompt+continuation+[pad]*(common-len(continuation)))
            masks.append([0]*(width-len(prompt))+[1]*(len(prompt)+common))
        mask=torch.tensor(masks,device='cuda:0')
        packed=dict(input_ids=torch.tensor(ids,device='cuda:0'),attention_mask=mask,
            position_ids=(mask.cumsum(-1)-1).clamp(min=0))
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            hidden=self._body(self._packed([prefix]),use_cache=False).last_hidden_state
            logits=self.model.get_output_embeddings()(hidden[0,-1])
            prefill={k:v[:,:-1] for k,v in packed.items()}
            state=self._body(prefill,use_cache=True)
            cached=self._body(dict(input_ids=packed['input_ids'][:,-1:],attention_mask=mask,
                position_ids=packed['position_ids'][:,-1:]),past_key_values=state.past_key_values,use_cache=True)
            batch_logits=self.model.get_output_embeddings()(cached.last_hidden_state[index,-1])
            measured=float((logits.double()-batch_logits.double()).abs().max())
            error=self.config['batch_gate']
            top=logits.topk(2).values
            chosen_gap=float((logits[reference[common]]-logits[actual[common]]).abs())
            margin=float(top[0]-top[1])
            near=(measured<=error and margin<measured and chosen_gap<measured
                and float(logits[actual[common]])>=float(top[1]))
        return dict(near_tie=near,first_divergence=common,reference_token=reference[common],
            batch_token=actual[common],top2_margin=margin,chosen_gap=chosen_gap,
            measured_error=measured,batch_gate=error)
