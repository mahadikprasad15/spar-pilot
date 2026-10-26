"""Pinned local weight inventory and PEFT-compatible base hash, without a model."""
import hashlib
import json
from pathlib import Path

import numpy as np

from pilot_eval.activation_prepare import PROJECTIONS
from pilot_eval.trace_readout_assets import read_weight, weight_header
from pilot_eval.training import file_hash


class LocalWeights:
    def inventory(self, directory, audit):
        directory=Path(directory).resolve()
        config_path=directory/'config.json'
        config=json.loads(config_path.read_text())
        if config.get('model_type')!='qwen2' or config.get('num_hidden_layers')!=28:
            raise ValueError('Expected the selected 28-block Qwen2 weight snapshot.')
        parents={str(config_path):file_hash(config_path)}
        index=directory/'model.safetensors.index.json'
        if index.exists():
            mapping=json.loads(index.read_text())['weight_map']
            parents[str(index)]=file_hash(index)
        else:
            single=directory/'model.safetensors'
            _,header=weight_header(single)
            mapping={key:single.name for key in header if key!='__metadata__'}
        for relative in sorted(set(mapping.values())):
            path=(directory/relative).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('Weight shard escapes selected snapshot.')
            print(f'Stage 0 weights: verifying {path.name}.',flush=True)
            parents[str(path)]=file_hash(path)
        modules={}
        for layer in range(28):
            for projection in PROJECTIONS:
                group='self_attn' if projection in PROJECTIONS[:4] else 'mlp'
                key=f'model.layers.{layer}.{group}.{projection}.weight'
                if key not in mapping:
                    raise ValueError('Weight snapshot lacks a required adapted linear module.')
                module='base_model.model.'+key[:-len('.weight')]
                modules[module]=dict(path=str(directory/mapping[key]),key=key)
        # Match frozen_weight_hash's sorted PEFT names, FP32 values and tied parameter deduplication.
        parameters={}
        for key,relative in mapping.items():
            if key=='lm_head.weight' and config.get('tie_word_embeddings'):
                continue
            parts=key.split('.')
            peft=key
            if len(parts)>=5 and parts[-2] in PROJECTIONS and parts[-1] in ['weight','bias']:
                peft='.'.join(parts[:-1]+['base_layer',parts[-1]])
            parameters['base_model.model.'+peft]=(key,directory/relative)
        if config.get('tie_word_embeddings') and 'model.embed_tokens.weight' not in mapping:
            raise ValueError('Tied embedding source is missing.')
        digest=hashlib.sha256()
        print('Stage 0 weights: computing the recorded FP32 base-parameter hash.',flush=True)
        for name,(key,path) in sorted(parameters.items()):
            values=read_weight(path,key)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite base weights.')
            digest.update(name.encode()+b'\0')
            digest.update(str(tuple(values.shape)).encode())
            digest.update(values.astype('<f4',copy=False).tobytes())
        base=digest.hexdigest()
        if base!=audit['identity']['base_sha256']:
            raise ValueError('Local base-parameter hash differs from the audited source.')
        return dict(execution_kind='cpu-local-pinned-weights', model=audit['identity']['model'],
                    model_revision=audit['identity']['model_revision'],base_sha256=base,
                    parameters_hashed=len(parameters),parents=parents,modules=modules,
                    provenance='FP32 parameter hash matched; no model/network construction')

    def matrix(self, entry):
        return read_weight(entry['path'],entry['key'])
