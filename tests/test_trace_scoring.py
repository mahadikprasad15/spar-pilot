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
