"""Scientific training/recovery through public workflow, without Hub downloads."""
import json
from pathlib import Path
import pytest
from pilot_eval.cli import main
from test_grpo_preflight import fixture, collect, Dependencies as PreflightDependencies


def frozen(root):
    path,baseline,options=fixture(root);deps=PreflightDependencies()
    assert collect(root,path,baseline,options,deps)==0
    review=root/'review.json'
    review.write_text(json.dumps(dict(reviewed=True,notes='Checked evidence',margin=.1,
        preregistration=dict(author='researcher',written_at='2026-10-08T00:00:00+00:00',prediction='No drop',falsifier='Drop exceeds margin'),
        monitors={k:dict(threshold=1.,action='flag') for k in ['cap_fraction','dead_group_fraction','length_ratio_change_8_steps']})))
    assert main(['grpo-freeze','--config',str(root/'plans/flight/grpo.preflight.json'),'--review',str(review),
        '--name','frozen','--output-root',str(root)])==0
    return root/'plans/frozen/grpo.frozen.json'


class Engine:
    def __init__(self, deps):self.step=0;self.deps=deps
    def runtime(self):return dict(device='controlled GPU',dtype='float32',adapter_dtype='float32',deterministic_algorithms=True,tf32=False,total_memory_bytes=8*2**30)
    def train_window(self, rows, step):
        self.deps.calls.append(step)
        if step==self.deps.fail_at:raise RuntimeError('injected interruption')
        self.step=step
        from pilot_eval.grpo_baseline import _score
        records=[dict(step=step,advantage=0.,**_score(r,dict(text='#### 1',token_ids=[7],generated_token_ids=[7,1],stop_reason='eos'),d)) for r in rows for d in range(8)]
        return records,dict(loss=0.,learning_rate=.0001,pre_clip_gradient_norm=0.,gradient_consistency=dict(defined=False,reason='zero_gradient',cosine=None),adapter_norm=float(step),parameter_movement=1.,seconds=.1)
    def save_checkpoint(self, directory, step):
        from pilot_eval.training import CHECKPOINT_FILES
        directory.mkdir(parents=True,exist_ok=True)
        for name in CHECKPOINT_FILES:(directory/name).write_text(json.dumps(dict(global_step=step)))
        (directory/'prior-gradient.json').write_text(json.dumps(dict(step=step)))
    def restore_checkpoint(self, directory):self.step=json.loads((directory/'trainer_state.json').read_text())['global_step'];self.deps.restored.append(self.step)
    def check_integrity(self):return dict(base_before='base',base_after='base',base_unchanged=True)
    def close(self):pass


class Dependencies:
    def __init__(self):self.calls=[];self.restored=[];self.loads=0;self.fail_at=None
    def load_training(self, plan, settings, directory):self.loads+=1;return Engine(self)


def train(root,path,deps):
    return main(['grpo-train','--config',str(path),'--name','arm','--length-change-definition','step-lag-8','--output-root',str(root)],dependencies=deps)


def test_public_training_consumes_exact_groups_and_reuses_completed_run(tmp_path):
    path=frozen(tmp_path);deps=Dependencies()
    assert train(tmp_path,path,deps)==0
    plan=json.loads((tmp_path/'plans/arm/grpo.training.json').read_text());directory=tmp_path/plan['run_path']
    result=json.loads((directory/'results/results.json').read_text())
    assert result['optimizer_steps']==64 and result['training_prompts']==512 and result['accepted_completions']==4096
    assert result['checkpoints']==[0,8,16,32,64]
    draws=[json.loads(s) for s in (directory/'results/responses.jsonl').read_text().splitlines()]
    assert len({r['draw_id'] for r in draws})==4096
    assert deps.calls==list(range(1,65))
    assert train(tmp_path,path,deps)==0 and deps.loads==1
