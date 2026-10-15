"""Independent CPU oracles for the public readable-trace measurement interface."""
import numpy as np


def test_paired_stream_preserves_content_positions_sites_and_primary_pooling():
    from pilot_eval.trace_capture import summarize_pairs
    # Literal oracle: second example has one leading formatting token. All sites
    # are supplied in one record, as they will be from one hooked model forward.
    batches=[dict(example_ids=['legacy-a'],content_positions=[[0,1,2,3,4,5]],
        base={'embedding-output':np.zeros((1,6,2)),
              'block-14-output':np.array([[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]])},
        adapted={'embedding-output':np.zeros((1,6,2)),
                 'block-14-output':np.array([[[2,4],[3,5],[4,6],[5,7],[6,8],[7,9]]])}),
        dict(example_ids=['new-b'],content_positions=[[1,2,3,4,5,6]],
        base={'embedding-output':np.zeros((1,7,2)),
              'block-14-output':np.array([[[999,999],[2,4],[4,6],[6,8],[8,10],[10,12],[12,14]]])},
        adapted={'embedding-output':np.zeros((1,7,2)),
                 'block-14-output':np.array([[[-999,-999],[5,8],[7,10],[9,12],[11,14],[13,16],[15,18]]])})]
    result=summarize_pairs(iter(batches),sites=['embedding-output','block-14-output'],
                           content_tokens=6,retained_sites=['block-14-output'])
    assert result['example_ids']==['legacy-a','new-b']
    np.testing.assert_array_equal(result['position_delta_sums']['block-14-output'],[[4,6]]*6)
    np.testing.assert_array_equal(result['example_primary_delta']['block-14-output'],[[1,2],[3,4]])
    np.testing.assert_array_equal(result['example_primary_base']['block-14-output'],[[3,4],[6,8]])
    np.testing.assert_array_equal(result['retained_delta']['block-14-output'][1],[[3,4]]*6)
    # Base moment positions 1..5, both documents; sum and outer product derived
    # explicitly from the ten vectors, including off-diagonal terms.
    assert result['base_moments']['count']==10
    np.testing.assert_array_equal(result['base_moments']['sum'],[60,75])
    np.testing.assert_array_equal(result['base_moments']['outer_sum'],[[450,550],[550,675]])
    assert result['position_delta_sums']['block-14-output'].dtype==np.float64


def test_logit_lens_keeps_final_norm_gain_and_has_no_zero_vector_readout():
    import pytest
    from pilot_eval.trace_capture import logit_lens
    # RMS([1, 2]) = sqrt(2.5); norm gains [2, 3] give [2, 6]/sqrt(2.5).
    matrix=np.array([[1,0],[0,1],[1,-1]])
    scores=logit_lens([1,2],norm_weight=[2,3],unembedding=matrix,epsilon=0)
    np.testing.assert_allclose(scores,[1.2649110640673518,3.794733192202055,-2.5298221281347035],atol=1e-14)
    np.testing.assert_array_equal(np.argsort(scores),np.argsort(logit_lens([7,14],norm_weight=[2,3],unembedding=matrix,epsilon=0)))
    assert logit_lens([0,0],norm_weight=[2,3],unembedding=matrix,epsilon=0) is None
    with pytest.raises(ValueError,match='shape'):
        logit_lens([1,2],norm_weight=[1],unembedding=matrix,epsilon=0)


def test_retention_report_measures_conversion_without_claiming_qualification():
    import pytest
    from pilot_eval.trace_capture import retention_evidence
    original=np.array([[1.0001,2.0002],[3.0003,4.0004]],dtype=np.float64)
    evidence=retention_evidence(original,storage_dtype='float16')
    assert evidence['qualified'] is False
    assert evidence['max_absolute_error']==pytest.approx(.0004)
    assert evidence['worst_coordinate']==[1,1]
    assert evidence['source_dtype']=='float64'
    with pytest.raises(ValueError,match='Nonfinite'):
        retention_evidence([float('nan')],storage_dtype='float16')


def test_named_hooks_capture_embedding_and_pre_norm_blocks_and_always_remove_hooks():
    import pytest
    from pilot_eval.trace_capture import HookedTraceCapture
    class Module:
        def __init__(self,operation):self.operation=operation;self.hooks=[]
        def register_forward_hook(self,hook):
            self.hooks.append(hook)
            class Handle:
                def remove(inner):self.hooks.remove(hook)
            return Handle()
        def __call__(self,x):
            y=self.operation(x)
            for hook in self.hooks:hook(self,(x,),y)
            return y
    class Body:
        def __init__(self):
            self.embedding=Module(lambda ids:np.stack([ids,ids*2],axis=-1).astype(np.float32))
            self.layers=[Module(lambda x,i=i:x+i+1) for i in range(28)]
        def __call__(self,input_ids):
            x=self.embedding(input_ids)
            for block in self.layers:x=block(x)
            return x*100  # A final norm surrogate must not replace block-27 output.
    model=Body();capture=HookedTraceCapture(model.embedding,model.layers)
    result=capture.forward(lambda:model(np.array([[1,2]])))
    assert len(result)==29
    np.testing.assert_array_equal(result['embedding-output'],[[[1,2],[2,4]]])
    np.testing.assert_array_equal(result['block-14-output'],[[[121,122],[122,124]]])
    np.testing.assert_array_equal(result['block-27-output'],[[[407,408],[408,410]]])
    assert all(not module.hooks for module in [model.embedding,*model.layers])
    with pytest.raises(RuntimeError):capture.forward(lambda:(_ for _ in ()).throw(RuntimeError('interrupted')))
    assert all(not module.hooks for module in [model.embedding,*model.layers])


def test_nonfinite_hook_failure_names_the_site_and_coordinate():
    import pytest
    from pilot_eval.trace_capture import HookedTraceCapture,CaptureFailure
    class Module:
        def register_forward_hook(self,hook):
            self.hook=hook
            class Handle:
                def remove(inner):self.hook=None
            return Handle()
    module=Module();capture=HookedTraceCapture(module,[])
    def forward():module.hook(module,(),np.array([[[1.,float('nan')]]],dtype=np.float32))
    with pytest.raises(CaptureFailure) as failure:capture.forward(forward)
    assert failure.value.diagnostics['site']=='embedding-output'
    assert failure.value.diagnostics['coordinate']==[0,0,1]
    assert module.hook is None


def test_storage_sensitivity_rebuilds_primary_vector_for_paired_document_draws():
    from pilot_eval.trace_capture import storage_sensitivity
    # FP16 rounds these two near-equal components to 1, reversing the top-1
    # after cohort pooling by the declared ascending-ID tie rule.
    values=np.repeat(np.array([[[1.0001,0]],[[0,1.0002]]]),5,axis=1)
    result=storage_sensitivity(values,lambda v:v,storage_dtype='float16',draws=8,seed=42,top_k=1)
    assert result['qualified'] is False
    assert result['cohort']['top_k_overlap']==0
    assert result['cohort']['max_logit_error']>0
    assert result['bootstrap']['unit']=='whole-document'
    assert len(result['bootstrap']['draws'])==8
    for draw in result['bootstrap']['draws']:
        assert len(draw['indices'])==2
        assert draw['top_k_overlap']==(0 if set(draw['indices'])=={0,1} else 1)
    assert result==storage_sensitivity(values,lambda v:v,storage_dtype='float16',draws=8,seed=42,top_k=1)
