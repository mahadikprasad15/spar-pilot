"""Hash-bound CPU inspection of the restricted upstream Jacobian tensor format.

No Torch import, remote code or arbitrary pickle globals. A readable archive is
not a qualified scientific instrument: source provenance and numerical gates
are recorded separately.
"""
from collections import OrderedDict
from dataclasses import dataclass
import io
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _save_frozen,_hash
from pilot_eval.run import _write_json,_write_state
from pilot_eval.trace_corpus import _inside
from pilot_eval.sft import safe_name


@dataclass(frozen=True)
class Storage:
    key:str
    dtype:str
    count:int


@dataclass(frozen=True)
class Tensor:
    storage:Storage
    offset:int
    shape:tuple
    stride:tuple


def _tensor(storage,offset,shape,stride,requires_grad,hooks):
    if not isinstance(storage,Storage) or requires_grad or hooks:
        raise ValueError('Unsupported tensor reconstruction metadata.')
    if (type(offset) is not int or offset<0 or not isinstance(shape,tuple) or not isinstance(stride,tuple)
            or len(shape)!=2 or len(stride)!=2 or any(type(i) is not int or i<1 for i in shape)
            or offset!=0 or stride!=(shape[1],1) or storage.count!=shape[0]*shape[1]):
        raise ValueError('Only dense complete contiguous matrices are supported.')
    return Tensor(storage,offset,shape,stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return _tensor
        if (module,name)==('collections','OrderedDict'):return OrderedDict
        if module=='torch' and name in ['HalfStorage','FloatStorage','DoubleStorage']:
            return {'HalfStorage':'f2','FloatStorage':'f4','DoubleStorage':'f8'}[name]
        raise ValueError(f'Unsupported pickle global: {module}.{name}')

    def persistent_load(self,value):
        if (not isinstance(value,tuple) or len(value)!=5 or value[0]!='storage'
                or value[1] not in ['f2','f4','f8'] or not isinstance(value[2],str)
                or not value[2].isdigit() or value[3]!='cpu' or type(value[4]) is not int
                or not 0<value[4]<=100000000):
            raise ValueError('Unsupported tensor storage metadata.')
        return Storage(value[2],value[1],value[4])


def archive_inventory(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries.')
        pickles=[n for n in names if n.endswith('/data.pkl')]
        if len(pickles)!=1 or archive.getinfo(pickles[0]).file_size>1000000:
            raise ValueError('Unexpected tensor archive metadata.')
        data=MetadataReader(io.BytesIO(archive.read(pickles[0]))).load()
        if (not isinstance(data,dict) or set(data)!={'J','n_prompts','source_layers','d_model'}
                or type(data['d_model']) is not int or not 1<=data['d_model']<=8192
                or type(data['n_prompts']) is not int or data['n_prompts']<1
                or not isinstance(data['J'],dict) or not 1<=len(data['J'])<=128
                or any(type(k) is not int or k<0 for k in data['J'])
                or data['source_layers']!=sorted(data['J'])):
            raise ValueError('Unexpected Jacobian lens schema.')
        prefix=pickles[0][:-len('data.pkl')]
        order=archive.read(prefix+'byteorder').decode() if prefix+'byteorder' in names else 'little'
        if order not in ['little','big']:raise ValueError('Unsupported byte order.')
        matrices={}
        for layer,tensor in data['J'].items():
            if not isinstance(tensor,Tensor) or tensor.shape!=(data['d_model'],data['d_model']):
                raise ValueError('Lens matrix dimension mismatch.')
            storage_name=prefix+'data/'+tensor.storage.key
            if archive.getinfo(storage_name).file_size!=tensor.storage.count*np.dtype(tensor.storage.dtype).itemsize:
                raise ValueError('Lens storage byte count mismatch.')
            matrices[layer]=dict(path=storage_name,dtype=('<' if order=='little' else '>')+tensor.storage.dtype,
                                 shape=list(tensor.shape),count=tensor.storage.count)
        return dict(d_model=data['d_model'],n_prompts=data['n_prompts'],source_layers=data['source_layers'],matrices=matrices)


def read_matrix(path,layer,*,expected_sha256):
    if file_hash(path)!=expected_sha256:raise ValueError('Lens file hash mismatch.')
    metadata=archive_inventory(path)
    if layer not in metadata['matrices']:raise ValueError('Requested source block is unavailable.')
    row=metadata['matrices'][layer]
    with zipfile.ZipFile(path) as archive:
        matrix=np.frombuffer(archive.read(row['path']),dtype=row['dtype']).reshape(row['shape']).astype(np.float64)
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite lens matrix.')
    return matrix


def transport(vector,matrix):
    value=np.asarray(vector,dtype=np.float64);matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.ndim!=2 or value.shape[-1:]!=(matrix.shape[1],) or not np.isfinite(value).all() or not np.isfinite(matrix).all():
        raise ValueError('Invalid transport shape or nonfinite values.')
    return value@matrix.T


def inspect_lens(path,output_root,name,*,expected_sha256,provenance,expected_model,
                 expected_revision,expected_width=1536,primary_block=14):
    safe_name(name);root=Path(output_root).resolve();path=_inside(root,path)
    directory=root/'runs/pilot-5'/name/'lens-inspection'
    config=dict(schema_version=1,lens_path=str(path.relative_to(root)),sha256=expected_sha256,
        provenance=provenance,expected_model=expected_model,expected_revision=expected_revision,
        expected_width=expected_width,primary_block=primary_block,implementation_sha256=file_hash(Path(__file__)))
    with run_lock(directory):
        _save_frozen(directory/'config.json',config)
        if file_hash(path)!=expected_sha256:raise ValueError('Lens file hash mismatch.')
        if (directory/'complete.json').exists():
            verify_lens_inspection(directory/'results/results.json',root)
            return directory/'results/results.json'
        _save_frozen(directory/'meta/run_manifest.json',config);_write_state(directory,'running',0,1)
        try:
            metadata=archive_inventory(path);matrices={}
            with zipfile.ZipFile(path) as archive:
                for layer,row in metadata['matrices'].items():
                    values=np.frombuffer(archive.read(row['path']),dtype=row['dtype'])
                    matrices[str(layer)]=dict(shape=row['shape'],dtype=row['dtype'],finite=bool(np.isfinite(values).all()),
                        storage_sha256=__import__('hashlib').sha256(values.tobytes()).hexdigest())
            reasons=[];incompatible=[]
            if metadata['d_model']!=expected_width:incompatible.append('hidden-width mismatch')
            if primary_block not in metadata['source_layers']:incompatible.append('primary source block unavailable')
            if not all(r['finite'] for r in matrices.values()):incompatible.append('nonfinite matrix')
            if provenance.get('model')!=expected_model:incompatible.append('model identity mismatch')
            revision=provenance.get('model_revision')
            if revision and revision!=expected_revision:incompatible.append('base-model revision mismatch')
            if not revision:reasons.append('exact fitting base-model revision not recorded')
            if provenance.get('source_site')!='block-output':reasons.append('source-site convention not established')
            if provenance.get('target')!='final-block-output-before-norm':reasons.append('target representation unresolved')
            if provenance.get('conflicts'):reasons.extend(provenance['conflicts'])
            status='incompatible' if incompatible else 'insufficient-provenance' if reasons else 'pending-numerical-qualification'
            result=dict(schema_version=1,status=status,qualified=False,d_model=metadata['d_model'],n_prompts=metadata['n_prompts'],
                source_layers=metadata['source_layers'],primary_available=primary_block in metadata['source_layers'],
                matrices=matrices,reasons=incompatible+reasons,lens_sha256=expected_sha256,
                source_site_requirement='decoder block output before final model norm',
                readout_convention='transport J @ v then model final norm and unembedding; target provenance must agree',
                failure_action='preserve evidence; no fitting, logit-only primary or paid model work automatically authorized')
            _write_json(directory/'results/results.json',result)
            _write_json(directory/'complete.json',dict(config_sha256=_hash(config),files={n:file_hash(directory/n) for n in ['config.json','meta/run_manifest.json','results/results.json']}))
            _write_state(directory,status,1,1)
            return directory/'results/results.json'
        except BaseException as exc:
            _write_state(directory,'failed',0,1,str(exc));raise


def verify_lens_inspection(result_path,output_root):
    root=Path(output_root).resolve();result_path=_inside(root,result_path);directory=result_path.parent.parent
    config=json.loads((directory/'config.json').read_text());marker=json.loads((directory/'complete.json').read_text())
    if set(marker['files'])!={'config.json','meta/run_manifest.json','results/results.json'}:
        raise ValueError('Inspection seal omits required payloads.')
    if result_path != directory/'results/results.json':
        raise ValueError('Unexpected inspection result path.')
    if marker['config_sha256']!=_hash(config):raise ValueError('Inspection config identity mismatch.')
    if config['implementation_sha256']!=file_hash(Path(__file__)):raise ValueError('Inspection implementation changed.')
    for name,expected in marker['files'].items():
        if file_hash(_inside(directory,name))!=expected:raise ValueError('Lens inspection payload hash mismatch.')
    if file_hash(_inside(root,config['lens_path']))!=config['sha256']:raise ValueError('Lens file hash mismatch.')
    return json.loads(result_path.read_text())
