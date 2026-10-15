import math
import numpy as np
import pytest
from pilot_eval.trace_vocabulary import TokenInventory


def test_scoring_requires_freeze_and_normalizes_unique_ranked_lists_and_zero_vectors(tmp_path):
    from pilot_eval.trace_scoring import score_vector
    inv=TokenInventory({0:' total',1:'Total',2:' apples',3:'####',4:'\ufffd',5:'<special>'},special_ids=[5])
    matrix=np.array([[3.,1.],[3.,1.],[2.,0.],[1.,0.],[9.,0.],[10.,0.]])
    ref={'ref_questions':dict(strings=['total','####'],word=['total'],format_number=['####'],
        scores=[dict(string='total',y_a=5,y_b=0,z=2.),dict(string='apples',y_a=1,y_b=5,z=-1.),dict(string='####',y_a=5,y_b=1,z=1.)])}
    readout=lambda v:matrix@v
    with pytest.raises(ValueError,match='freeze'):
        score_vector([1,0],readout,inv,references=ref)
    a=score_vector([1,0],readout,inv,references=ref,synthetic=True)
    b=score_vector([7.3,0],readout,inv,references=ref,synthetic=True)
    assert a['strings']==['total','apples','####']
    assert a['list_ids']==[0,2,3]  # duplicate-ID tie broken by ascending ID
    assert a['scores']==b['scores']
    assert a['scores']['ref_questions']['overlap']==pytest.approx(2/3)
    assert a['scores']['ref_questions']['overlap_word']==pytest.approx(1/3)
    assert a['scores']['ref_questions']['overlap_format']==pytest.approx(1/3)
    assert a['short_list'] is True
    zero=score_vector([0,0],readout,inv,references=ref,synthetic=True)
    assert zero['state']=='undefined' and zero['reason']=='zero-vector'
    empty=score_vector([1],lambda v:np.array([1.]),TokenInventory({0:'\ufffd'}),references=ref,synthetic=True)
    assert empty['valid_count']==0
    assert math.isnan(empty['scores']['ref_questions']['overlap'])


def test_scale_invariance_for_two_linear_lenses_and_matched_max_null():
    from pilot_eval.trace_scoring import score_vector, random_vectors, null_scores
    inv=TokenInventory({i:f' word{i}' for i in range(40)})
    ref={'ref_questions':dict(strings=[f'word{i}' for i in range(5)],word=[f'word{i}' for i in range(5)],format_number=[],scores=[])}
    for matrix in [np.arange(80,dtype=float).reshape(40,2),np.arange(80,dtype=float).reshape(40,2)[::-1]]:
        lens=lambda v:matrix@v
        a=score_vector([1,2],lens,inv,references=ref,synthetic=True)
        b=score_vector([7.3,14.6],lens,inv,references=ref,synthetic=True)
        assert a['strings']==b['strings'] and a['valid_count']==20
        assert a['scores']['ref_questions']['overlap']==b['scores']['ref_questions']['overlap']
    vectors=random_vectors(3,2,seed=9,positions=5)
    assert vectors.shape==(3,5,2)
    assert np.linalg.norm(vectors,axis=-1)==pytest.approx(np.ones((3,5)))
    scorer=lambda v:{'toy':float(v[0])}
    result=null_scores(scorer,dimension=2,draws=3,seed=9,positions=5)
    assert result['toy']==pytest.approx(vectors[:,:,0].max(axis=1))
    assert np.array_equal(result['toy'],null_scores(scorer,dimension=2,draws=3,seed=9,positions=5)['toy'])
    with pytest.raises(ValueError,match='covariance'):
        random_vectors(3,2,seed=9,covariance=np.zeros((2,2)))


def test_step_zero_gate_is_exact_and_rejects_even_small_nonzero_deltas():
    from pilot_eval.trace_scoring import check_zero_adapter
    assert check_zero_adapter(np.zeros((2,5,3)))['exact_zero'] is True
    with pytest.raises(ValueError,match='exactly zero'):
        check_zero_adapter(np.full((2,5,3),1e-15))


def test_reviewed_scoring_carries_prior_exposure_status(tmp_path):
    from pilot_eval.trace_scoring import score_vector
    from pilot_eval.trace_vocabulary import freeze_references,approve_references
    reference=dict(strings=['total'],word=['total'],format_number=[],scores=[])
    path=freeze_references(tmp_path/'refs',{key:reference for key in ['ref_questions','ref_gold','ref_grpo']},parameters={},input_files={})
    (path.parent/'top30.md').write_text('Synthetic preview; not a real reference vocabulary.')
    approve_references(path,reviewer='fixture',notes='Synthetic gate check.',exposure_status='seen-before-freeze')
    result=score_vector([1],lambda v:np.array([1.]),TokenInventory({0:'total'}),freeze_path=path)
    assert result['inference_status']=='exploratory-or-unverified-exposure'
