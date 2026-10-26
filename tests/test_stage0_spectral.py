"""Public spectral workflows with independently known diagonal-matrix answers."""
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot_eval.cli import main
from test_stage0_workflow import command, sources, tensor_file, write_json
from pilot_eval.workflow import _hash


MODULES = ['base_model.model.model.layers.0.self_attn.q_proj',
           'base_model.model.model.layers.0.mlp.down_proj']


class TinyWeights:
    """Explicit synthetic backend, not a claim about the real Qwen checkpoint."""
    calls = []

    def inventory(self, directory, audit):
        path = directory/'tiny.npz'
        return dict(execution_kind='synthetic-cpu-fixture', model=audit['identity']['model'],
                    model_revision=audit['identity']['model_revision'],
                    base_sha256=audit['identity']['base_sha256'],
                    parents={str(path):hashlib.sha256(path.read_bytes()).hexdigest()},
                    modules={m:dict(path=str(path), key=m) for m in MODULES})

    def matrix(self, entry):
        self.calls.append(entry['key'])
        with np.load(entry['path'],allow_pickle=False) as arrays:
            return arrays[entry['key']]


def fixture(root, *, isotropic=False):
    selection=sources(root,width=64,output=64)
    assert command(root,'prepare','--selection',selection,'--name','audit')==0
    audit=root/'plans/audit/stage0.plan.json'
    assert command(root,'audit','--config',audit)==0
    weights=root/'weights';weights.mkdir()
    matrix=np.eye(64,dtype=np.float32) if isotropic else np.diag(np.arange(1,65,dtype=np.float32))
    np.savez(weights/'tiny.npz',**{m:matrix for m in MODULES})
    deps=TinyWeights();deps.calls=[]
    assert main(['stage0-spectral-prepare','--audit',str(audit),'--weights-dir',str(weights),
                 '--name','basis','--output-root',str(root)],dependencies=deps)==0
    return root/'plans/basis/stage0.spectral.json',deps


def spectral(root,op,plan,*extra,dependencies=None):
    return main(['stage0-spectral-'+op,'--config',str(plan),'--output-root',str(root),*map(str,extra)],
                dependencies=dependencies)


def test_spectral_cache_matches_known_spectrum_and_keeps_trained_results_unrevealed(tmp_path):
    plan,deps=fixture(tmp_path)
    assert spectral(tmp_path,'run',plan,dependencies=deps)==0
    run=tmp_path/'runs/stage0/basis/spectral'
    summary=json.loads((run/'results/cache.json').read_text())
    assert summary['modules']==2 and summary['trained_overlap_revealed'] is False
    assert not (run/'reports').exists()
    expected=np.arange(64,0,-1,dtype=np.float64)
    row=summary['module_results'][0]
    for level in [30,50,70]:
        k=int(np.searchsorted(np.cumsum(expected**2), np.sum(expected**2)*level/100)+1)
        assert row[f'k{level}']==k
    with np.load(run/row['arrays_path'],allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays['sigma'],expected)
        assert arrays['u'].shape[1]==max(40,row['k70'])
        assert arrays['v'].shape[1]==max(40,row['k70'])
    assert row['reconstruction_relative_error'] < 1e-6
    assert spectral(tmp_path,'verify',plan)==0


def test_cache_resume_skips_completed_modules_and_rejects_changed_weights(tmp_path):
    plan,deps=fixture(tmp_path)
    assert spectral(tmp_path,'run',plan,'--max-new-units',1,dependencies=deps)==0
    assert len(deps.calls)==1
    assert spectral(tmp_path,'run',plan,dependencies=deps)==0
    assert len(deps.calls)==2
    assert spectral(tmp_path,'run',plan,dependencies=deps)==0
    assert len(deps.calls)==2
    with (tmp_path/'weights/tiny.npz').open('ab') as stream:stream.write(b'changed')
    assert spectral(tmp_path,'verify',plan)==1


def test_projection_report_requires_explicit_reveal_and_reports_undefined_zero(tmp_path):
    plan,deps=fixture(tmp_path,isotropic=True)
    assert spectral(tmp_path,'run',plan,dependencies=deps)==0
    review=tmp_path/'review.json'
    write_json(review,dict(reveal_trained_overlaps=False,prediction_status='exploratory',notes='Not approved'))
    assert spectral(tmp_path,'report',plan,'--review',review,'--name','projections')==1
    write_json(review,dict(reveal_trained_overlaps=True,prediction_status='exploratory',notes='Approved reveal'))
    assert spectral(tmp_path,'report',plan,'--review',review,'--name','projections')==0
    report=tmp_path/'reports/projections'
    rows=[json.loads(line) for line in (report/'results/projections.jsonl').read_text().splitlines()]
    zero=next(r for r in rows if r['arm']=='sft' and r['step']==0 and r['energy_level']==50)
    assert zero['pi_out'] is None and zero['undefined_out_reason']=='zero factor'
    endpoint=next(r for r in rows if r['arm']=='sft' and r['step']==64 and r['energy_level']==50)
    k=endpoint['k']
    assert abs(endpoint['pi_out']-k/64)<1e-10
    assert abs(endpoint['pi_in']-sum(np.arange(1,k+1,dtype=float)**2)/sum(np.arange(1,65,dtype=float)**2))<1e-10
    assert (report/'plots/principal-overlap.svg').is_file()
    assert (report/'complete.json').is_file()
    assert spectral(tmp_path,'report',plan,'--review',review,'--name','projections')==0


def test_failed_gaussian_gate_stops_report_without_destroying_cache(tmp_path):
    plan,deps=fixture(tmp_path)
    assert spectral(tmp_path,'run',plan,dependencies=deps)==0
    review=tmp_path/'review.json'
    write_json(review,dict(reveal_trained_overlaps=True,prediction_status='exploratory',notes='Known negative fixture'))
    assert spectral(tmp_path,'report',plan,'--review',review,'--name','failed-projections')==1
    assert not (tmp_path/'reports/failed-projections/complete.json').exists()
    assert (tmp_path/'reports/failed-projections/logs/errors.jsonl').is_file()
    assert spectral(tmp_path,'verify',plan)==0


def test_bad_weight_shape_or_zero_spectrum_preserves_failure(tmp_path):
    plan,deps=fixture(tmp_path)
    class Broken(TinyWeights):
        def matrix(self, entry):return np.zeros((64,64),dtype=np.float32)
    assert spectral(tmp_path,'run',plan,dependencies=Broken())==1
    run=tmp_path/'runs/stage0/basis/spectral'
    assert not (run/'complete.json').exists()
    assert json.loads((run/'meta/status.json').read_text())['state']=='failed'
    assert (run/'logs/errors.jsonl').is_file()


def test_cache_verification_rejects_changed_retained_basis(tmp_path):
    plan,deps=fixture(tmp_path)
    assert spectral(tmp_path,'run',plan,dependencies=deps)==0
    run=tmp_path/'runs/stage0/basis/spectral'
    row=json.loads((run/'results/cache.json').read_text())['module_results'][0]
    (run/row['arrays_path']).write_bytes(b'not a basis')
    assert spectral(tmp_path,'verify',plan)==1


def local_snapshot(root):
    """Known PEFT parameter names and FP32 bytes, independent of the inventory code."""
    selection=sources(root,width=64,output=64)
    weights=root/'local-weights';weights.mkdir()
    arrays={};named={}
    source=json.loads((root/'inputs/sft/activation.prepared.json').read_text())
    for layer in range(28):
        for projection in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
            group='self_attn' if projection in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp'
            base=f'model.layers.{layer}.{group}.{projection}'
            value=np.eye(64,dtype=np.float32)
            arrays[base+'.weight']=value
            named[f'base_model.model.model.layers.{layer}.{group}.{projection}.base_layer.weight']=value
    bias=np.arange(64,dtype=np.float32)
    arrays['model.layers.0.self_attn.q_proj.bias']=bias
    named['base_model.model.model.layers.0.self_attn.q_proj.base_layer.bias']=bias
    embedding=np.ones((5,64),dtype=np.float32)
    arrays['model.embed_tokens.weight']=embedding
    arrays['lm_head.weight']=embedding.copy()
    named['base_model.model.model.embed_tokens.weight']=embedding
    norm=np.ones(64,dtype=np.float32)
    arrays['model.norm.weight']=norm
    named['base_model.model.model.norm.weight']=norm
    digest=hashlib.sha256()
    for name,value in sorted(named.items()):
        digest.update(name.encode()+b'\0');digest.update(str(value.shape).encode());digest.update(value.tobytes())
    base_hash=digest.hexdigest()
    tensor_file(weights/'model.safetensors',arrays)
    write_json(weights/'config.json',dict(model_type='qwen2',num_hidden_layers=28,tie_word_embeddings=True))
    for arm in ['sft','grpo']:
        path=root/'inputs'/arm/'activation.prepared.json'
        config=json.loads(path.read_text());config['source_evidence']['base_sha256']=base_hash
        write_json(path,config)
        write_json(path.parent/'prepare-complete.json',dict(config_sha256=_hash(config),
            files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert command(root,'prepare','--selection',selection,'--name','real-reader')==0
    audit=root/'plans/real-reader/stage0.plan.json'
    assert command(root,'audit','--config',audit)==0
    return weights,audit,base_hash


def test_actual_local_weight_reader_matches_peft_parameter_hash_without_torch(tmp_path):
    weights,audit,expected=local_snapshot(tmp_path)
    assert main(['stage0-spectral-prepare','--audit',str(audit),'--weights-dir',str(weights),
                 '--name','local-basis','--output-root',str(tmp_path)])==0
    plan=tmp_path/'plans/local-basis/stage0.spectral.json'
    cfg=json.loads(plan.read_text())
    assert cfg['inventory']['base_sha256']==expected
    assert cfg['inventory']['parameters_hashed']==199
    assert len(cfg['inventory']['modules'])==196
    assert spectral(tmp_path,'run',plan,'--max-new-units',1)==0
    assert not (tmp_path/'runs/stage0/local-basis/spectral/complete.json').exists()


def test_weight_identity_mismatch_refuses_cache_before_svd(tmp_path):
    weights,audit,_=local_snapshot(tmp_path)
    with (weights/'model.safetensors').open('r+b') as stream:
        stream.seek(-4,2);stream.write(np.float32(2).tobytes())
    assert main(['stage0-spectral-prepare','--audit',str(audit),'--weights-dir',str(weights),
                 '--name','bad-basis','--output-root',str(tmp_path)])==1
    assert not (tmp_path/'plans/bad-basis/stage0.spectral.json').exists()
