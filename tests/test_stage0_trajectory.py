"""Trajectory source audit, same-product pairing, CPU oracles and resume."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_document_bootstrap_preserves_token_and_example_weighting():
    from pilot_eval.stage0_trajectory_math import interval
    draws=np.array([[0,0],[1,1],[0,1]])
    result=interval([1.,9.],[1,3],draws)
    assert result['equal_example_mean']==2. and result['token_mean']==2.5
    assert result['bootstrap_unit']=='whole document' and result['interval_95']==[1.05,2.95]


def test_runtime_groups_require_session_identity_and_do_not_infer_compatibility():
    from pilot_eval.stage0_trajectory_math import comparison_group
    common=dict(runtime=dict(device='A100',session_id='one'),inputs_sha256='inputs',model_identity={'base_sha256':'base'},scheduling={'batch_size':1})
    assert comparison_group(common,'product-a')==comparison_group(common,'product-b')
    changed={**common,'runtime':dict(device='A100',session_id='two')}
    assert comparison_group(common,'product-a')!=comparison_group(changed,'product-b')
    unknown={**common,'runtime':dict(device='A100')}
    assert comparison_group(unknown,'product-a')!=comparison_group(unknown,'product-b')


def test_lengths_are_paired_medians_and_keep_cap_status():
    from pilot_eval.stage0_trajectory_math import lengths
    # Median of paired changes (1) differs from difference of medians (9).
    before=[dict(id=str(i),tokens=t,capped=False,prompt_sha256=str(i)) for i,t in enumerate([1,10,100])]
    after=[dict(id=str(i),tokens=t,capped=i==2,prompt_sha256=str(i)) for i,t in enumerate([2,100,101])]
    result=lengths(before,after,np.array([[0,1,2],[0,0,0],[2,2,2]]))
    assert result['median_paired_change']==1 and result['shorter_share']==0
    assert result['reference_cap_rate']==0 and result['edited_cap_rate']==1/3
    with pytest.raises(ValueError):lengths(before,after[:-1],np.array([[0,1,2]]))
