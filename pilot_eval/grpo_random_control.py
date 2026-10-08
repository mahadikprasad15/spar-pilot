"""One isolated random rank-1 intervention, matched to effective weight norms."""
import json
from pathlib import Path
import numpy as np
from pilot_eval.activation_prepare import PROJECTIONS
from pilot_eval.training import file_hash
from pilot_eval.workflow import _save_frozen
from pilot_eval.run import _write_json


def _factors(source):
    from safetensors.numpy import load_file
    config=json.loads((source/'adapter_config.json').read_text())
    if (config.get('r')!=1 or config.get('lora_alpha')!=1 or config.get('lora_dropout')!=0
            or config.get('rank_pattern') or config.get('alpha_pattern') or config.get('use_dora')
            or config.get('use_rslora') or config.get('bias','none')!='none'):
        raise ValueError('random control requires plain rank-1 alpha-1 dropout-zero LoRA')
    arrays=load_file(str(source/'adapter_model.safetensors'))
    modules=[];required=set()
    for layer in range(28):
        for projection in PROJECTIONS:
            stem=f"base_model.model.model.layers.{layer}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            ka,kb=stem+'.lora_A.weight',stem+'.lora_B.weight'
            required.update([ka,kb])
            if ka not in arrays or kb not in arrays: raise ValueError('incomplete 196-module factor mapping')
            a,b=arrays[ka],arrays[kb]
            if (a.ndim!=2 or b.ndim!=2 or a.shape[0]!=1 or b.shape[1]!=1
                    or min(a.shape+b.shape)<1 or a.dtype!=np.float32 or b.dtype!=np.float32
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError('invalid/nonfinite/non-FP32 rank-1 source factors')
            norm=float(np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64)))
            if not np.isfinite(norm): raise ValueError('nonfinite effective update norm')
            modules.append(dict(layer=layer,projection=projection,a_key=ka,b_key=kb,
                                a_shape=list(a.shape),b_shape=list(b.shape),target_norm=norm,scaling=1.))
    if set(arrays)!=required: raise ValueError('unexpected adapter weights outside 196 rank-1 modules')
    return config,arrays,modules


def _random_factors(modules):
    rng=np.random.default_rng(42)
    arrays={}
    directions=[]
    for row in modules:
        a=rng.normal(size=row['a_shape']);b=rng.normal(size=row['b_shape'])
        a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        # Unit directions are retained even when the effective update is zero.
        directions.append(dict(layer=row['layer'],projection=row['projection'],
                               input_unit=a.tolist(),output_unit=b.tolist()))
        arrays[row['a_key']]=a.astype(np.float32)
        arrays[row['b_key']]=(b*row['target_norm']).astype(np.float32)
    return arrays,directions


def construct_control(source, destination):
    """Save/reverify factors before any measurement; never mutate the trained adapter."""
    from safetensors.numpy import load_file, save_file
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination==source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('control must have a separate directory from its source')
    config,original,modules=_factors(source)
    identity=dict(protocol='pilot4-random-rank1-v1',seed=42,realizations=1,
                  rng='numpy-PCG64; isolated generator; Gaussian normalized in float64',
                  order='layer ascending; q,k,v,o,gate,up,down',
                  source_config_sha256=file_hash(source/'adapter_config.json'),
                  source_adapter_sha256=file_hash(source/'adapter_model.safetensors'),modules=modules,
                  interpretation='weight-norm-matched random intervention; not a null distribution')
    arrays,directions=_random_factors(modules)
    destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if set(saved['files']) != {'adapter_config.json','adapter_model.safetensors','construction.json'}:
            raise ValueError('random control seal omits required payloads')
        if saved['identity']!=identity: raise ValueError('random control source identity changed')
        for name,digest in saved['files'].items():
            if file_hash(destination/name)!=digest: raise ValueError('random control payload changed')
    else:
        # Caller owns the workflow run lock; interrupted, unsealed payloads may be replaced.
        save_file(arrays,str(destination/'adapter_model.safetensors'))
        _save_frozen(destination/'adapter_config.json',config)
        _save_frozen(destination/'construction.json',dict(**identity,directions=directions))
    saved=load_file(str(destination/'adapter_model.safetensors'))
    if set(saved)!=set(arrays) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
        raise ValueError('random control does not reproduce isolated seed-42 construction')
    for row in modules:
        a,b=saved[row['a_key']],saved[row['b_key']]
        realized=float(np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64)))
        if (not np.isclose(np.linalg.norm(a.astype(np.float64)),1.,atol=1e-6,rtol=1e-5)
                or not np.isclose(realized,row['target_norm'],atol=1e-6,rtol=1e-5)
                or (row['target_norm']==0 and np.count_nonzero(b))
                or (row['target_norm']>0 and realized==0)):
            raise ValueError('random control norm/unit-direction reconstruction failed')
    if (file_hash(source/'adapter_config.json')!=identity['source_config_sha256']
            or file_hash(source/'adapter_model.safetensors')!=identity['source_adapter_sha256']):
        raise ValueError('trained source changed during control construction')
    if not marker.exists():
        _write_json(marker,dict(identity=identity,files={name:file_hash(destination/name) for name in
            ['adapter_config.json','adapter_model.safetensors','construction.json']}))
    return identity
