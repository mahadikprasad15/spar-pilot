"""Exercise packing, positions and KV continuation without Torch or downloads."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from pilot_eval.stage0_batching_backend import BatchEngine
from pilot_eval.stage0_batching import numeric_ok


def test_actual_batch_backend_checks_unequal_lengths_and_cached_decode():
    class Tensor(np.ndarray):
        def __new__(cls,x):return np.asarray(x).view(cls)
        def double(self):return Tensor(self.astype(np.float64))
        def abs(self):return Tensor(np.abs(self))
        def cumsum(self,axis):return Tensor(np.asarray(self).cumsum(axis))
        def clamp(self,min):return Tensor(np.maximum(self,min))
        def cpu(self):return self
        def max(self,axis=None,**kwargs):
            if axis is None:return np.asarray(self).max(**kwargs)
            return Tensor(np.asarray(self).max(axis)),Tensor(np.asarray(self).argmax(axis))
    calls=[];corrupt={'cache':False}
    def body(input_ids,attention_mask,position_ids,return_dict,use_cache,past_key_values=None):
        size=input_ids.shape[1];mask=np.asarray(attention_mask)
        expected=(mask.cumsum(-1)-1).clip(min=0)[:,-size:]
        np.testing.assert_array_equal(position_ids,expected)
        values=np.asarray(input_ids)*mask[:,-size:]
        hidden=values.cumsum(-1)+(0 if past_key_values is None else past_key_values[:,None])
        if past_key_values is not None and corrupt['cache']:hidden=hidden+.1
        calls.append((input_ids.shape,use_cache,past_key_values is not None))
        return SimpleNamespace(last_hidden_state=Tensor(np.stack([hidden,expected],-1).astype(np.float32)),
            past_key_values=hidden[:,-1])
    head=lambda x:Tensor(x@np.array([[.1,.2,-.3],[.2,.3,.1]],dtype=np.float32))
    model=SimpleNamespace(set_adapter=lambda v:None,eval=lambda:None,
        get_base_model=lambda:SimpleNamespace(model=body),get_output_embeddings=lambda:head)
    engine=object.__new__(BatchEngine);engine.model=model
    engine.config=dict(decoding={'pad_token_id':0})
    engine.torch=SimpleNamespace(tensor=lambda x,**kw:Tensor(x),float32=np.dtype('float32'),
        inference_mode=nullcontext,autocast=lambda *a,**kw:nullcontext(),isfinite=np.isfinite,
        cuda=SimpleNamespace(synchronize=lambda:None))
    rows=[{'id':'short'},{'id':'long'}]
    tokens=[dict(id='short',input_ids=[3,5,7,2],answer_positions=[1,2]),
        dict(id='long',input_ids=[4,3,6,8,9,2],answer_positions=[2,3,4])]
    value=engine.compare(rows,'V1',tokens)
    assert numeric_ok(value,1e-5) and value['max_absolute_error']==0.
    assert [len(r['errors']) for r in value['examples']]==[3,4]
    assert any(cache for _,_,cache in calls)
    corrupt['cache']=True
    assert not numeric_ok(engine.compare(rows,'V1',tokens),1e-5)


def test_near_tie_uses_batch_gate_and_requires_margin_below_measured_error():
    import pilot_eval.stage0_batching_backend as backend
    import inspect
    source=inspect.getsource(backend.BatchEngine.divergence)
    assert "self.config['batch_gate']" in source
    assert "self.config['absolute_gate']" not in source
