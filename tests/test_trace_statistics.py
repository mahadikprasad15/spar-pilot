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
