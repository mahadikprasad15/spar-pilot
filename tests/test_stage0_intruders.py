"""Gap/subspace geometry through saved CPU commands; no network or model loads."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main
from test_stage0_spectral import fixture, spectral, TinyWeights


def command(root,op,*args,dependencies=None):
    return main(['stage0-intruders-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


def prepared(root):
    cache,deps=fixture(root)
    assert spectral(root,'run',cache,dependencies=deps)==0
    assert command(root,'prepare','--spectral',cache,'--name','geometry')==0
    return root/'plans/geometry/stage0.intruders.json',deps


def test_public_measure_resume_report_and_verify(tmp_path):
    plan,deps=prepared(tmp_path)
    assert command(tmp_path,'run','--config',plan,'--max-new-units',1,dependencies=deps)==0
    directory=tmp_path/'runs/stage0/geometry/intruders'
    first=next((directory/'units').glob('*/complete.json'));before=first.read_bytes()
    assert not (directory/'complete.json').exists()
    assert command(tmp_path,'status','--config',plan)==0
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    assert first.read_bytes()==before
    assert command(tmp_path,'verify','--config',plan)==0
    result=json.loads((directory/'results/results.json').read_text())
    assert result['unit_count']==4 and result['module_count']==2
    assert result['arms']==['sft','grpo'] and not result['real_model_execution']
    assert result['small_update_skip'] is False
    for row in result['module_results']:
        assert len(row['candidates'])==40
        assert len(row['detection_indices'])==20
        assert len(row['levels'])==3
        assert row['primary']['status']=='undecidable-at-or-below-chance'
        assert row['primary']['eligible_indices']==[]
        assert row['levels']['50']['m']>=40
        assert row['levels']['50']['chance_norm']==math.sqrt(row['levels']['50']['m']/64)
    assert (directory/'results/candidates.csv').is_file()
    assert (directory/'report.html').is_file()
    assert (directory/'plots/rotation.svg').is_file()
    assert (directory/'plots/overlap.png').is_file()
    with np.load(next((directory/'units').glob('*/geometry.npz')),allow_pickle=False) as arrays:
        assert arrays['new_u'].shape[1]>=40 and arrays['new_v'].shape[1]>=40
        assert arrays['new_sigma'].shape==(64,)
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    (tmp_path/'weights/tiny.npz').write_bytes(b'changed')
    assert command(tmp_path,'verify','--config',plan)==1


def test_subspace_angles_ignore_internal_basis_rotation():
    from pilot_eval.stage0_intruder_math import principal_angles,overlap_norms,ratio
    old=np.eye(5)[:,:2]
    theta=.72
    mixing=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    np.testing.assert_allclose(principal_angles(old,old@mixing),[0,0],atol=1e-6)
    new=np.column_stack([np.eye(5)[:,0],np.cos(.4)*np.eye(5)[:,1]+np.sin(.4)*np.eye(5)[:,2]])
    np.testing.assert_allclose(principal_angles(old,new),[0,.4],atol=1e-7)
    np.testing.assert_allclose(overlap_norms(old,new),overlap_norms(old@mixing,new),atol=1e-15)
    assert ratio(2,0)==dict(value=None,status='infinite-positive-over-zero')
    assert ratio(0,0)==dict(value=None,status='undefined-zero-over-zero')
    assert ratio(2,4)==dict(value=.5,status='finite')


def test_diagonal_intruder_and_zero_update_oracles():
    from pilot_eval.stage0_intruder_math import measure
    # The base top-m contains exactly coordinates 0..39, chance sqrt(40/200)<.5.
    sigma=np.concatenate([np.arange(100.,60.,-1),np.full(160,1.)])
    w=np.diag(sigma);old=dict(sigma=sigma,u=np.eye(200)[:,:40],v=np.eye(200)[:,:40])
    ranks=dict(k30=10,k50=20,k70=30,d_out=200,d_in=200)
    a=np.zeros((1,200));b=np.zeros((200,1));a[0,100]=1;b[100,0]=150
    arrays,row=measure(w,a,b,old,ranks)
    assert row['update_operator_norm']==150
    assert row['primary']['status']=='eligible-intruders'
    assert row['primary']['eligible_indices']==[0]
    assert row['candidates'][0]['overlap_norm']['50']==0
    assert row['candidates'][0]['sigma']==151
    assert row['levels']['50']['rotation']['left_angles_radians'][-1]==pytest.approx(np.pi/2)
    # No update is measured; exact boundary ties remain explicit limitations.
    zeros,zero=measure(w,a,b*0,old,ranks)
    assert zero['zero_update'] and zero['update_operator_norm']==0
    assert zero['primary']['eligible_indices']==[]
    assert zero['levels']['50']['rotation']['max_left_degrees']==0
    assert zero['levels']['50']['update_over_gap']['status']=='finite'


def test_chance_boundary_and_strict_threshold_semantics():
    from pilot_eval.stage0_intruder_math import classify
    assert classify([.1,.5,.7],chance=.5,threshold=.5)['status']=='undecidable-at-or-below-chance'
    row=classify([.1,.5,.7],chance=.49,threshold=.5)
    assert row['eligible_indices']==[0] # .5 equality is not flagged
    assert classify([.1],chance=.49,threshold=.3)['eligible_indices']==[]
