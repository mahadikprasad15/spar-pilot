"""Public lens inspection/readout with independent CPU tensor archives."""
import io,json,pickle,zipfile
from pathlib import Path
import numpy as np
import pytest


def archive(path,*,width=2):
    # Encode an independent tiny PyTorch-compatible pickle/storage fixture.
    # GLOBAL names are data in this byte stream; no Torch module is imported.
    def string(s):
        data=s.encode();return b'X'+len(data).to_bytes(4,'little')+data
    tensor=(b'ctorch._utils\n_rebuild_tensor_v2\n(('
        +string('storage')+b'ctorch\nFloatStorage\n'+string('0')+string('cpu')+b'K\x04tQK\x00K\x02K\x02\x86K\x02K\x01\x86\x89ccollections\nOrderedDict\n)RtR')
    payload=b'\x80\x02}('+string('J')+b'}K\x00'+tensor+b's'+string('n_prompts')+b'K\x10'+string('source_layers')+b']K\x00a'+string('d_model')+b'K'+bytes([width])+b'u.'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('lens/data.pkl',payload);z.writestr('lens/byteorder','little')
        z.writestr('lens/data/0',np.array([[1,2],[3,4]],dtype='<f4').tobytes())


def test_public_archive_inspection_and_transport_are_cpu_only_and_hash_bound(tmp_path):
    from pilot_eval.trace_lenses import inspect_lens,read_matrix,transport
    from pilot_eval.training import file_hash
    path=tmp_path/'lens.pt';archive(path)
    output=inspect_lens(path,tmp_path,'lens-inspect',expected_sha256=file_hash(path),
        provenance=dict(model='fixture',model_revision=None,target='final-block-output-before-norm',source_site='block-output'),
        expected_model='fixture',expected_revision='pin',expected_width=2,primary_block=0)
    result=json.loads(output.read_text())
    assert result['status']=='insufficient-provenance' and not result['qualified']
    assert result['source_layers']==[0] and result['d_model']==2
    matrix=read_matrix(path,0,expected_sha256=file_hash(path))
    np.testing.assert_array_equal(transport([2,5],matrix),[12,26])
    assert result['matrices']['0']['finite'] is True
    assert inspect_lens(path,tmp_path,'lens-inspect',expected_sha256=file_hash(path),
        provenance=dict(model='fixture',model_revision=None,target='final-block-output-before-norm',source_site='block-output'),
        expected_model='fixture',expected_revision='pin',expected_width=2,primary_block=0)==output
    with pytest.raises(ValueError,match='hash'):read_matrix(path,0,expected_sha256='0'*64)


def test_archive_loader_refuses_unrecognized_pickle_globals_without_execution(tmp_path):
    from pilot_eval.trace_lenses import archive_inventory
    path=tmp_path/'unsafe.pt'
    with zipfile.ZipFile(path,'w') as z:z.writestr('lens/data.pkl',b'cos\nsystem\n(Vecho forbidden\ntR.')
    with pytest.raises(ValueError,match='Unsupported pickle global'):archive_inventory(path)


def test_inspection_seal_cannot_omit_required_result(tmp_path):
    from pilot_eval.trace_lenses import inspect_lens,verify_lens_inspection
    from pilot_eval.training import file_hash
    path=tmp_path/'lens.pt';archive(path)
    result=inspect_lens(path,tmp_path,'missing-seal',expected_sha256=file_hash(path),
        provenance=dict(model='fixture'),expected_model='fixture',expected_revision='pin',expected_width=2,primary_block=0)
    marker=result.parent.parent/'complete.json'
    payload=json.loads(marker.read_text());payload['files'].pop('results/results.json')
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='required'):verify_lens_inspection(result,tmp_path)
