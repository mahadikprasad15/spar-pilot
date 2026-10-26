"""Complete CPU edit catalogue through the saved command boundary."""
import json
from pathlib import Path

import numpy as np

from pilot_eval.cli import main
from pilot_eval.stage0_sources import factors
from test_stage0_spectral import fixture, spectral, MODULES


VARIANTS={'V0a','V0b','V1','V2','V3-0','V3-1','V3-2','V4','V4-input','V5'}


def edits(root,operation,plan,*extra):
    return main(['stage0-edits-'+operation,'--config',str(plan),'--output-root',str(root),*map(str,extra)])


def prepared(root):
    cache,deps=fixture(root,isotropic=True)
    assert spectral(root,'run',cache,dependencies=deps)==0
    assert main(['stage0-edits-prepare','--spectral',str(cache),'--output-root',str(root),'--name','edits'])==0
    return root/'plans/edits/stage0.edits.json'


def test_catalogue_reconstructs_principal_edits_size_controls_and_reference_identity(tmp_path):
    plan=prepared(tmp_path)
    assert edits(tmp_path,'run',plan)==0
    run=tmp_path/'runs/stage0/edits/edits'
    result=json.loads((run/'results/catalogue.json').read_text())
    assert set(result['variants'])==VARIANTS
    assert result['gpu_qualified'] is False and result['execution_kind']=='synthetic-cpu-fixture'
    paths={key:tmp_path/row['adapter_path'] for key,row in result['variants'].items()}
    original=factors(tmp_path/'inputs/sft/64')
    for variant,arm in [('V0a','sft'),('V0b','grpo')]:
        for file in ['adapter_config.json','adapter_model.safetensors']:
            assert (paths[variant]/file).read_bytes()==(tmp_path/f'inputs/{arm}/64'/file).read_bytes()
    output={key:factors(path) for key,path in paths.items()}
    # Isotropic fixture's k50=32 and identity singular bases.
    for module in MODULES:
        a,b=original[module]
        expected_b=b.copy();expected_b[:32]=0
        expected_a=a.copy();expected_a[:,:32]=0
        np.testing.assert_array_equal(output['V1'][module][0],a)
        np.testing.assert_allclose(output['V1'][module][1],expected_b,atol=1e-6)
        np.testing.assert_allclose(output['V2'][module][0],expected_a,atol=1e-6)
        np.testing.assert_array_equal(output['V2'][module][1],b)
        for edited,control in [('V1','V4'),('V2','V4-input')]:
            ea,eb=output[edited][module];ca,cb=output[control][module]
            np.testing.assert_allclose(np.linalg.norm(ea)*np.linalg.norm(eb),np.linalg.norm(ca)*np.linalg.norm(cb),rtol=1e-6)
            ratio=cb[b!=0]/b[b!=0]
            np.testing.assert_allclose(ratio,ratio[0],rtol=1e-6)
        assert not np.array_equal(output['V3-0'][module][1],output['V3-1'][module][1])
    # The synthetic backend touches two modules; all other payloads stay unchanged.
    untouched=next(module for module in original if module not in MODULES)
    for variant in output:
        source=factors(tmp_path/f"inputs/{'grpo' if variant in ['V0b','V5'] else 'sft'}/64")
        for x,y in zip(output[variant][untouched],source[untouched]):np.testing.assert_array_equal(x,y)
    assert edits(tmp_path,'verify',plan)==0
    assert not list((run/'results').glob('*projection*'))


def test_interruption_reuses_module_units_and_damaged_adapter_is_rejected(tmp_path):
    plan=prepared(tmp_path)
    assert edits(tmp_path,'run',plan,'--max-new-units',1)==0
    run=tmp_path/'runs/stage0/edits/edits'
    assert not (run/'complete.json').exists()
    receipts=list((run/'units').glob('*/complete.json'))
    assert len(receipts)==1
    saved=receipts[0].read_bytes()
    assert edits(tmp_path,'run',plan)==0
    assert receipts[0].read_bytes()==saved
    catalogue=json.loads((run/'results/catalogue.json').read_text())
    payload=tmp_path/catalogue['variants']['V1']['adapter_path']/'adapter_model.safetensors'
    payload.write_bytes(payload.read_bytes()+b'changed')
    assert edits(tmp_path,'verify',plan)==1
    assert edits(tmp_path,'run',plan)==1


def test_zero_rank_edit_is_byte_preserving_and_dense_update_oracle_agrees():
    from pilot_eval.stage0_edit_math import edit_module
    a=np.array([[1.,2.,3.]],dtype=np.float32);b=np.array([[2.],[4.]],dtype=np.float32)
    u=np.zeros((2,0),dtype=np.float32);v=np.zeros((3,0),dtype=np.float32)
    output,evidence,bases=edit_module(a,b,u,v)
    for va,vb in output.values():
        assert va.tobytes()==a.tobytes() and vb.tobytes()==b.tobytes()
    u=np.array([[1.],[0.]],dtype=np.float32);v=np.array([[1.],[0.],[0.]],dtype=np.float32)
    output,evidence,bases=edit_module(a,b,u,v)
    np.testing.assert_allclose(output['V1'][1]@output['V1'][0],(np.eye(2)-u@u.T)@(b@a))
    np.testing.assert_allclose(output['V2'][1]@output['V2'][0],(b@a)@(np.eye(3)-v@v.T))
    np.testing.assert_allclose(evidence['V4']['update_norm'],evidence['V1']['update_norm'],rtol=1e-6)


def test_changed_source_or_cache_cannot_be_used_for_edit_resume(tmp_path):
    plan=prepared(tmp_path)
    assert edits(tmp_path,'run',plan,'--max-new-units',1)==0
    basis=next((tmp_path/'runs/stage0/basis/spectral/units').glob('*/basis.npz'))
    basis.write_bytes(b'invalid')
    assert edits(tmp_path,'run',plan)==1


def test_cpu_notebook_has_public_edit_prepare_run_and_verify_cells():
    import ast
    notebook=json.loads(Path('notebooks/stage-0-cpu-colab.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    for operation in ['edits-prepare','edits-run','edits-verify']:
        assert "run_stage0('"+operation+"'" in code
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[] and cell['execution_count'] is None
