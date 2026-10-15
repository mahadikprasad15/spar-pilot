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
