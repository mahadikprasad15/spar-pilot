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


def test_zero_gaps_full_rank_and_invalid_subspaces_are_explicit():
    from pilot_eval.stage0_intruder_math import measure,diagnostics
    w=np.eye(64);a=np.zeros((1,64));b=np.zeros((64,1));a[0,-1]=1
    old=dict(sigma=np.ones(64),u=np.eye(64),v=np.eye(64))
    ranks=dict(k30=20,k50=32,k70=45,d_out=64,d_in=64)
    arrays,row=measure(w,a,b,old,ranks)
    assert row['levels']['50']['update_over_gap']==dict(value=None,status='undefined-zero-over-zero')
    assert row['levels']['50']['base_boundary']['status']=='exact-degeneracy'
    assert row['levels']['50']['rotation']['unique_boundary'] is False
    b[-1,0]=.01
    _,positive=measure(w,a,b,old,ranks)
    assert positive['levels']['50']['update_over_gap']==dict(value=None,status='infinite-positive-over-zero')
    full={**ranks,'k70':64};_,row=measure(w,a,b,old,full)
    assert row['levels']['70']['update_over_gap']['status']=='undefined-no-next-singular-value'
    short={**old,'u':old['u'][:,:39],'v':old['v'][:,:39]}
    with pytest.raises(ValueError,match='unavailable'):measure(w,a,b,short,ranks)
    invalid={**ranks,'k70':65}
    with pytest.raises(ValueError,match='clipping forbidden'):measure(w,a,b,old,invalid)
    broken={**old,'u':np.eye(64)*2}
    with pytest.raises(ValueError,match='orthogonality'):measure(w,a,b,broken,ranks)


def test_degenerate_boundary_does_not_become_a_stable_vector_claim():
    from pilot_eval.stage0_intruder_math import measure
    # Equal-energy boundary: subspace mixing across the boundary is nonunique.
    w=np.diag(np.r_[np.ones(40)*2,np.ones(24)])
    old=dict(sigma=np.r_[np.ones(40)*2,np.ones(24)],u=np.eye(64),v=np.eye(64))
    ranks=dict(k30=10,k50=20,k70=30,d_out=64,d_in=64)
    a=np.zeros((1,64));b=np.zeros((64,1));a[0,0]=1
    _,row=measure(w,a,b,old,ranks)
    assert row['levels']['50']['base_boundary']['gap']==0
    assert row['levels']['50']['rotation']['unique_boundary'] is False
    assert any(r['numerically_degenerate'] for r in row['candidates'])
    assert row['zero_update'] and row['small_update_skip'] is False


def test_mid_unit_failure_resume_and_corruption_rejection(tmp_path):
    plan,deps=prepared(tmp_path)
    class Broken(TinyWeights):
        def matrix(self,entry):raise RuntimeError('synthetic SVD input failure')
    assert command(tmp_path,'run','--config',plan,'--max-new-units',1,dependencies=Broken())==1
    directory=tmp_path/'runs/stage0/geometry/intruders'
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'
    assert not list((directory/'units').glob('*/complete.json'))
    assert (directory/'logs/errors.jsonl').exists()
    assert command(tmp_path,'run','--config',plan,'--max-new-units',1,dependencies=deps)==0
    folder=next((directory/'units').iterdir())
    # Partial payload in another unit is unsealed and can be recomputed.
    other=directory/'units/L00.down_proj-grpo';other.mkdir(exist_ok=True,parents=True)
    (other/'geometry.npz').write_bytes(b'unsealed')
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    (folder/'geometry.npz').write_bytes(b'corrupt')
    assert command(tmp_path,'verify','--config',plan)==1


def test_resealed_changed_geometry_metadata_does_not_verify(tmp_path):
    from pilot_eval.training import file_hash
    plan,deps=prepared(tmp_path)
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    directory=tmp_path/'runs/stage0/geometry/intruders';unit=next((directory/'units').iterdir())
    row=json.loads((unit/'result.json').read_text());row['primary']['status']='no-intruders'
    (unit/'result.json').write_text(json.dumps(row))
    seal=json.loads((unit/'complete.json').read_text());seal['files']['result.json']=file_hash(unit/'result.json')
    (unit/'complete.json').write_text(json.dumps(seal))
    marker=json.loads((directory/'complete.json').read_text())
    for p in ['result.json','complete.json']:marker['files'][str((unit/p).relative_to(directory))]=file_hash(unit/p)
    (directory/'complete.json').write_text(json.dumps(marker))
    assert command(tmp_path,'verify','--config',plan)==1


def test_live_writer_and_explicit_recovery(tmp_path):
    import os,platform
    from test_stage0_workflow import write_json
    plan,deps=prepared(tmp_path);directory=tmp_path/'runs/stage0/geometry/intruders'
    owner=directory/'writer.json';write_json(owner,dict(token='known',host=platform.node(),pid=os.getpid()))
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==1
    assert command(tmp_path,'recover-writer','--config',plan,'--owner-token','known','--review-notes','inspect')==1
    write_json(owner,dict(token='known',host='stopped-colab',pid=1234))
    assert command(tmp_path,'recover-writer','--config',plan,'--owner-token','wrong','--review-notes','stopped')==1
    assert command(tmp_path,'recover-writer','--config',plan,'--owner-token','known','--review-notes','stopped')==0
    assert command(tmp_path,'run','--config',plan,'--max-new-units',1,dependencies=deps)==0


def test_cpu_verify_needs_no_model_runtime_or_new_svd(tmp_path):
    import subprocess,sys
    plan,deps=prepared(tmp_path)
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    code='''
import sys
class Reject:
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','transformers','peft'}:raise AssertionError(name)
sys.meta_path.insert(0,Reject())
from pilot_eval.stage0_intruders import verify
assert verify(sys.argv[1],sys.argv[2])['geometry_complete']
'''
    subprocess.run([sys.executable,'-c',code,str(plan),str(tmp_path)],check=True)


def test_notebook_exposes_optional_cpu_geometry_without_gpu_work():
    import ast
    data=json.loads(Path('notebooks/stage-0-cpu-colab.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type']=='code')
    for op in ['intruders-prepare','intruders-run','intruders-status','intruders-verify']:assert op in code
    for c in data['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['execution_count'] is None and c['outputs']==[]


def test_source_change_during_svd_blocks_publication(tmp_path):
    plan,_=prepared(tmp_path)
    class Changing(TinyWeights):
        changed=False
        def matrix(self,entry):
            result=super().matrix(entry)
            if not self.changed:
                with open(entry['path'],'ab') as stream:stream.write(b'changed after read')
                self.changed=True
            return result
    assert command(tmp_path,'run','--config',plan,dependencies=Changing())==1
    directory=tmp_path/'runs/stage0/geometry/intruders'
    assert not (directory/'complete.json').exists()
    assert json.loads((directory/'meta/status.json').read_text())['state']=='failed'
