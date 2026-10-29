"""Real CPU autograd fixture for the public endpoint workflow; no Hub packages.

Not a Qwen/PEFT qualification. All 196 saved rank-one factors participate in a
small next-token model, and the real production loss/optimizer loop is reused.
"""
import json
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace

from pilot_eval.stage0_sources import factors
from pilot_eval.trace_readout_assets import weight_header,read_weight


def tensor_file(path,arrays):
    header={};chunks=[];offset=0
    for name,array in arrays.items():
        raw=array.astype('<f4').tobytes();header[name]=dict(dtype='F32',shape=list(array.shape),data_offsets=[offset,offset+len(raw)])
        chunks.append(raw);offset+=len(raw)
    encoded=json.dumps(header).encode();path.write_bytes(len(encoded).to_bytes(8,'little')+encoded+b''.join(chunks))


class Engine:
    def __init__(self,initial):
        torch.manual_seed(42);self.initial=initial;self.modules=sorted(factors(initial));self.tokenizer=SimpleNamespace(pad_token_id=0)
        class Tiny(torch.nn.Module):
            def __init__(self,values):
                super().__init__()
                self.register_parameter('base',torch.nn.Parameter(torch.arange(128,dtype=torch.float32)/100,requires_grad=False))
                self.lora_A=torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(values[n][0].copy())) for n in sorted(values)])
                self.lora_B=torch.nn.ParameterList([torch.nn.Parameter(torch.from_numpy(values[n][1].copy())) for n in sorted(values)])
            def forward(self,input_ids,**kwargs):
                # Coefficients vary by token; both factors receive true gradients.
                token=input_ids.float().unsqueeze(-1)
                delta=sum((a.sum()*b.sum() for a,b in zip(self.lora_A,self.lora_B)))/196
                direction=torch.linspace(-1,1,128,device=input_ids.device)
                return SimpleNamespace(logits=self.base+delta*direction*(1+token/100))
        self.model=Tiny(factors(initial))
    def load_adapter(self,path):
        values=factors(path)
        with torch.no_grad():
            for n,a,b in zip(self.modules,self.model.lora_A,self.model.lora_B):
                a.copy_(torch.from_numpy(values[n][0].copy()));b.copy_(torch.from_numpy(values[n][1].copy()))
    def save_adapter(self,path):
        path.mkdir(parents=True,exist_ok=True)
        (path/'README.md').write_text('Incidental PEFT-style model card; not an inference/recovery payload.')
        (path/'adapter_config.json').write_bytes((self.initial/'adapter_config.json').read_bytes())
        arrays={}
        for n,a,b in zip(self.modules,self.model.lora_A,self.model.lora_B):
            arrays[n+'.lora_A.weight']=a.detach().numpy();arrays[n+'.lora_B.weight']=b.detach().numpy()
        tensor_file(path/'adapter_model.safetensors',arrays)
    def close(self):pass


class Dependencies:
    def __init__(self,stop=None,device='CPU tiny fixture'):self.stop=stop;self.device=device;self.opened=0
    def open(self,gold,rows,initial):
        self.opened+=1
        return Engine(initial),dict(device=self.device,dtype='float32',tf32=False,versions={'torch':torch.__version__},model_context_limit=512)
    def after_step(self,step):
        if step==self.stop:raise RuntimeError('fixture interruption after sealed optimizer step')
