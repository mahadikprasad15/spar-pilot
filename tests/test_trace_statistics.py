import math
import numpy as np
import pytest


def test_declared_pvalues_holm_and_snr_match_hand_cases():
    from pilot_eval.trace_statistics import null_pvalue, c3_pvalue, holm, snr
    assert null_pvalue(2,np.zeros(1000))['p_value']==pytest.approx(1/1001)
    assert null_pvalue(0,np.zeros(1000))['p_value']==1
    assert c3_pvalue([1,2,3])==0.5
    assert c3_pvalue([-1,0,1])==1
    assert c3_pvalue([0,0,0])==1
    assert holm([.01,.04,.03,.2])==pytest.approx([.04,.09,.09,.2])
    # Mean=(2,0); SE=1, so SNR=2 for two example vectors (1,0), (3,0).
    assert snr([[1,0],[3,0]])==pytest.approx(2)
    assert math.isnan(snr([[1,0],[1,0]]))
    assert math.isnan(snr([[1,0]]))
    with pytest.raises(ValueError,match='finite'):
        null_pvalue(1,[0,math.nan])


def test_paired_bootstrap_aligns_ids_and_ncurves_preserve_membership():
    from pilot_eval.trace_statistics import bootstrap_compare, bootstrap_scores, ncurves
    left={'a':np.array([[1.,0.]]*5),'b':np.array([[3.,0.]]*5)}
    right={'b':left['b'].copy(),'a':left['a'].copy()}
    score=lambda v:float(v[0])
    result=bootstrap_compare(left,right,score,draws=20,seed=7)
    assert result['differences']==pytest.approx(np.zeros(20))
    assert result['p_value']==1 and result['interval']==[0.,0.]
    assert result['indices']==bootstrap_compare(left,right,score,draws=20,seed=7)['indices']
    single=bootstrap_scores(left,score,draws=20,seed=7)
    assert set(single['scores'])<={1.,2.,3.}
    assert single['observed']==2.
    curves=ncurves(left,score,sizes=[1,2],draws=4,seed=8,strata={'a':'legacy-150','b':'new-50'})
    full=[row for row in curves if row['n']==2]
    assert all(row['score']==2. and row['counts']=={'legacy-150':1,'new-50':1} for row in full)
    assert all(row['full_cohort'] for row in full)
    with pytest.raises(ValueError,match='document IDs'):
        bootstrap_compare(left,{'c':left['a']},score,draws=2)
