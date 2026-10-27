"""Conditional matrix catalogue public workflow and independent assignment oracles."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_eval.cli import main


def command(root,op,*args,dependencies=None):
    return main(['stage0-intruder-edits-'+op,'--output-root',str(root),*map(str,args)],dependencies=dependencies)


def test_assignment_matches_exhaustive_objective_and_index_ties():
    from pilot_eval.stage0_intruder_edit_math import match_controls
    sigma=np.array([12.,11.,10.,9.,8.,7.])
    flags=[0,2];controls=[1,3,4,5]
    result=match_controls(sigma,flags,controls)
    exhaustive=min((sum(abs(sigma[f]-sigma[c]) for f,c in zip(flags,p)),p)
                   for p in itertools.permutations(controls,len(flags)))
    assert tuple(r['control_index'] for r in result['pairs'])==exhaustive[1]
    assert result['total_absolute_difference']==exhaustive[0]
    ties=match_controls(np.ones(6),[0,1],[2,3,4,5])
    assert [r['control_index'] for r in ties['pairs']]==[2,3]
    assert match_controls(sigma,[0,2],[1])['status']=='insufficient-controls'
    zeros=match_controls(np.zeros(5),[0],[1,2])
    assert zeros['pairs'][0]['ratio_status']=='undefined-zero-flagged-singular-value'
    assert not zeros['pairs'][0]['ratio_retained']


def test_halving_general_matrix_and_ratio_masks_without_rematching():
    from pilot_eval.stage0_intruder_edit_math import edit_matrix,module_selection
    sigma=np.array([10.,9.,8.,7.,6.])
    w=np.diag(sigma);u=v=np.eye(5)
    edited,evidence=edit_matrix(w,u,v,sigma,[0,2])
    np.testing.assert_array_equal(edited,np.diag([5.,9.,4.,7.,6.]).astype(np.float32))
    assert [r['after'] for r in evidence['singular_values']]==[5.,4.]
    row=dict(primary=dict(status='eligible-intruders',eligible_indices=[0,2]),
        levels={'50':dict(chance_norm=.1,m=40)},
        candidates=[dict(index=i,sigma=s,overlap_norm={'50':0. if i in [0,2] else 1.}) for i,s in enumerate([10.,8.,1.,.1])])
    selection=module_selection(row)
    assert [p['control_index'] for p in selection['matching']['pairs']]==[1,3]
    assert selection['selections']['ratio']==[0]
    assert selection['selections']['ratio-control']==[1]
    assert selection['selections']['full']==selection['selections']['matched']==[0,2]


def public_sources(root,eligible):
    from test_stage0_spectral import fixture,spectral
    from pilot_eval.stage0_edits import prepare as prepare_edits,run as run_edits
    from pilot_eval.stage0_intruders import prepare as prepare_geometry,run as run_geometry
    if not eligible:
        cache,deps=fixture(root)
    else:
        # A rank-one diagonal update creates one new leading direction, outside
        # the old top-40 subspace, in a 200-dimensional output space.
        from test_stage0_workflow import sources,command as audit,tensor_file,write_json
        from test_stage0_spectral import MODULES,TinyWeights
        from pilot_eval.training import file_hash
        from pilot_eval.workflow import _hash
        selection=sources(root,width=200,output=200)
        for arm in ['sft','grpo']:
            source=root/'inputs'/arm/'activation.prepared.json'
            config=json.loads(source.read_text());checkpoint=root/'inputs'/arm/'64'
            from pilot_eval.stage0_sources import factors
            current=factors(checkpoint)
            arrays={}
            for module in current:
                a=np.zeros((1,200),dtype=np.float32);b=np.zeros((200,1),dtype=np.float32)
                a[0,100]=150;b[100,0]=1 if arm=='sft' else 0
                arrays[module+'.lora_A.weight']=a;arrays[module+'.lora_B.weight']=b
            tensor_file(checkpoint/'adapter_model.safetensors',arrays)
            seal=json.loads((checkpoint/'complete.json').read_text())
            seal['files']['adapter_model.safetensors']=file_hash(checkpoint/'adapter_model.safetensors')
            write_json(checkpoint/'complete.json',seal)
            config['source_evidence']['checkpoints']['64'].update(complete_sha256=file_hash(checkpoint/'complete.json'),adapter_sha256=seal['files']['adapter_model.safetensors'])
            write_json(source,config)
            write_json(source.parent/'prepare-complete.json',dict(config_sha256=_hash(config),files={str(source.relative_to(root)):file_hash(source)}))
        assert audit(root,'prepare','--selection',selection,'--name','audit')==0
        assert audit(root,'audit','--config',root/'plans/audit/stage0.plan.json')==0
        weights=root/'weights';weights.mkdir();w=np.diag(np.r_[np.arange(100,60,-1),np.ones(160)]).astype(np.float32)
        np.savez(weights/'tiny.npz',**{m:w for m in MODULES})
        deps=TinyWeights()
        assert main(['stage0-spectral-prepare','--audit',str(root/'plans/audit/stage0.plan.json'),'--weights-dir',str(weights),
                     '--name','basis','--output-root',str(root)],dependencies=deps)==0
        cache=root/'plans/basis/stage0.spectral.json'
    assert spectral(root,'run',cache,dependencies=deps)==0
    edit=prepare_edits(cache,root,'edits');run_edits(edit,root)
    geometry=prepare_geometry(cache,root,'geometry');run_geometry(geometry,root,dependencies=deps,progress=False)
    assert command(root,'prepare','--edits',edit,'--geometry',geometry,'--name','conditional')==0
    return root/'plans/conditional/stage0.intruder-edits.json',deps


def test_verified_not_applicable_publishes_no_matrices(tmp_path):
    plan,deps=public_sources(tmp_path,False)
    class NoMatrix:
        def matrix(self,*args):raise AssertionError('not-applicable must not read a base matrix')
    assert command(tmp_path,'run','--config',plan,dependencies=NoMatrix())==0
    directory=tmp_path/'runs/stage0/conditional/intruder-edits'
    result=json.loads((directory/'results/catalogue.json').read_text())
    assert result['state']=='not-applicable' and result['variants']=={}
    assert result['arms']['sft']['status']=='not-applicable'
    assert not list(directory.rglob('*.npy'))
    assert command(tmp_path,'verify','--config',plan)==0
    assert command(tmp_path,'run','--config',plan,dependencies=NoMatrix())==0


def test_eligible_arm_full_matched_and_sensitivity_catalogues_resume(tmp_path):
    plan,deps=public_sources(tmp_path,True)
    assert command(tmp_path,'run','--config',plan,'--max-new-units',1,dependencies=deps)==0
    directory=tmp_path/'runs/stage0/conditional/intruder-edits'
    unit=next((directory/'units').glob('*/complete.json'));before=unit.read_bytes()
    assert command(tmp_path,'status','--config',plan)==0
    assert command(tmp_path,'run','--config',plan,dependencies=deps)==0
    assert before==unit.read_bytes()
    assert command(tmp_path,'verify','--config',plan)==0
    result=json.loads((directory/'results/catalogue.json').read_text())
    assert result['state']=='complete' and result['arms']['grpo']['status']=='not-applicable'
    assert set(result['variants'])=={'V6-sft-unhalved','V6-sft-full','V6-sft-matched','V6c-sft-matched','V6-sft-ratio','V6c-sft-ratio'}
    assert result['variants']['V6-sft-full']['modules_edited']==2
    assert result['variants']['V6-sft-ratio']['status']=='not-applicable-no-retained-pairs'
    # Original flagged 151 vs best nonflagged 100 => ratio < .8. No rematching.
    pair=result['matching'][0]['matching']['pairs'][0]
    assert pair['flagged_index']==0 and pair['control_index']==1
    assert pair['sigma_ratio']==pytest.approx(100/151) and not pair['ratio_retained']
    variant=result['variants']['V6-sft-full']
    assert variant['representation']=='mixed-full-matrices' and not variant['gpu_qualified']
    matrix=next(iter(variant['merged_modules'].values()))
    value=np.load(tmp_path/matrix['path'],allow_pickle=False)
    assert value.dtype==np.float32 and value[100,100]==75.5
    assert variant['adapter_disable_modules']==sorted(variant['merged_modules'])
    assert (directory/'report.html').exists()


def test_mixed_scope_restores_exact_weights_flags_and_adapter_after_exception(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from pilot_eval.stage0_merged_backend import merged_scope
    from pilot_eval.training import file_hash
    class Tensor:
        def __init__(self,value):self.value=np.array(value,dtype=np.float32);self.dtype='float32';self.device='cpu';self.shape=self.value.shape;self._version=0
        def to(self,**kwargs):return self
    class Layer:
        def __init__(self,weight):self.base_layer=SimpleNamespace(weight=Tensor(weight));self._disable_adapters=False;self.merged=False
    layers={'edited':Layer([[1.,0.],[0.,1.]]),'untouched':Layer([[2.,0.],[0.,2.]])}
    class Model:
        active_adapters=['old']
        def set_adapter(self,name):self.active_adapters=[name]
        def get_submodule(self,name):return layers[name]
        def named_modules(self):return layers.items()
    model=Model();original=layers['edited'].base_layer.weight
    path=tmp_path/'edited.npy';np.save(path,np.eye(2,dtype=np.float32)*3)
    definition=dict(variant='V6-sft-full',status='ready-for-qualification',mode='full',
        merged_modules={'edited':dict(path='edited.npy',sha256=file_hash(path),shape=[2,2],dtype='float32')},
        adapter_disable_modules=['edited'])
    torch=SimpleNamespace(float32='float32',from_numpy=Tensor,nn=SimpleNamespace(Parameter=lambda t,requires_grad: t))
    with pytest.raises(RuntimeError,match='injected'):
        with merged_scope(model,definition,tmp_path,torch,adapter_name='sft'):
            assert model.active_adapters==['sft']
            assert layers['edited']._disable_adapters and not layers['untouched']._disable_adapters
            assert layers['edited'].base_layer.weight is not original
            np.testing.assert_array_equal(layers['edited'].base_layer.weight.value,np.eye(2)*3)
            raise RuntimeError('injected')
    assert model.active_adapters==['old'] and layers['edited'].base_layer.weight is original
    assert not layers['edited']._disable_adapters
    unhalved=dict(definition,mode='unhalved')
    with merged_scope(model,unhalved,tmp_path,torch,adapter_name='sft',negative_double_adapter=True):
        assert not layers['edited']._disable_adapters
    assert layers['edited'].base_layer.weight is original
    broken={**definition,'merged_modules':{**definition['merged_modules'],'missing':definition['merged_modules']['edited']},
            'adapter_disable_modules':['edited','missing']}
    with pytest.raises(KeyError):
        with merged_scope(model,broken,tmp_path,torch,adapter_name='sft'):pass
    assert layers['edited'].base_layer.weight is original and model.active_adapters==['old']
