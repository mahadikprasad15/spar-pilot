"""Supplemental measurement contract: independent small distributions and masks."""
import numpy as np
import pytest


def test_next_target_views_and_full_vocabulary_kl_have_known_answers():
    from pilot_eval.token_measurement_math import prediction_positions, full_vocabulary_kl, summarize_kl
    rows = [dict(id='one', input_ids=[9, 4, 5, 6], attention_mask=[1]*4,
                 masks={'question':[False,True,False,False],
                        'solution':[False,False,True,True], 'user':[False]*4}),
            dict(id='two', input_ids=[9, 7], attention_mask=[1,1],
                 masks={'question':[False]*2,'solution':[False]*2,'user':[False,True]})]
    positions = prediction_positions(rows)
    assert positions == [dict(example=0,context=0,target=1,view='question',token_id=4),
                         dict(example=0,context=1,target=2,view='solution',token_id=5),
                         dict(example=0,context=2,target=3,view='solution',token_id=6),
                         dict(example=1,context=0,target=1,view='user',token_id=7)]
    p, q = np.log([[.8,.2],[.5,.5]]), np.log([[.5,.5],[.8,.2]])
    values = full_vocabulary_kl(p,q)
    np.testing.assert_allclose(values, [.19274475702175753,.22314355131420976],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(full_vocabulary_kl(p,p),[0.,0.])
    # Different lengths make the two weighting definitions observably distinct.
    summary=summarize_kl([dict(example=0,view='solution')]*2+[dict(example=1,view='solution')],
                         np.array([0.,0.,.6]),['one','two'])
    cell=next(r for r in summary if r['view']=='solution')
    assert cell['token_mean']==pytest.approx(.2)
    assert cell['example_mean']==pytest.approx(.3)
    assert cell['tokens']==3 and cell['examples']==2
