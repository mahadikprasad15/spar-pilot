"""Public preflight/freeze gates with controlled GPU dependencies; no downloads."""
import copy
import json
from pathlib import Path
import pytest
from pilot_eval.cli import main
from test_grpo_baseline import prepared, Boundary, command


def fixture(root):
    path, baseline_settings = prepared(root)
    assert command(root, path, baseline_settings, Boundary()) == 0
    settings = dict(scorer='gsm8k-flexible-v3', loss='dapo', cap_rule=dict(method='ceil-p99-plus', extra_tokens=3),
                    generation_groups=1, backward_groups=1, evaluation_batch_size=2,
                    gradient_checkpointing=True, diagnostic_seed=142, memory_margin_gib=1,
                    top_p=1.0, top_k=0)
    options = root / 'preflight-settings.json'; options.write_text(json.dumps(settings))
    return path, root / 'plans/sampling/baseline.config.json', options


class Engine:
    calls = 0
    mismatch = False
    capacity_fails = False
    def runtime(self):
        return dict(device='controlled GPU', dtype='float32', adapter_dtype='float32',
                    deterministic_algorithms=True, tf32=False, total_memory_bytes=8 * 2**30)
    def zero_equivalence(self, rows, settings):
        return [dict(id=r['id'], baseline_token_ids=[7], zero_token_ids=[8 if self.mismatch else 7]) for r in rows]
    def trial(self, rows, settings, seed, directory):
        self.calls += 1
        return dict(seed=seed, state_sha256='identical', base_before='base', base_after='base',
                    zero_initialized=True, steps=[dict(step=i+1, loss=0., gradient_norm=1., seconds=.1) for i in range(2)],
                    responses=[dict(id=r['id'], draw=d, text='#### 1', token_ids=[7], reward=1,
                                    stop_reason='eos') for r in rows for d in range(8)], elapsed_seconds=2.)
    def capacity(self, rows, settings, directory):
        if self.capacity_fails: raise RuntimeError('CUDA out of memory')
        return dict(generation_seconds=1., backward_seconds=2., elapsed_seconds=4.,
                    peak_allocated_bytes=2**30, peak_reserved_bytes=2**30,
                    free_memory_bytes=6 * 2**30, synthetic_full_cap=True,
                    completions=64, completion_tokens=settings['completion_limit'], base_unchanged=True)
    def close(self): pass


class Dependencies:
    def __init__(self): self.engine = Engine(); self.loads=0
    def verify_controls(self, root, name):
        from pilot_eval.grpo_preflight import verify_controls
        from pilot_eval import grpo_controls, grpo_algorithm, grpo_checks
        from pilot_eval.training import file_hash
        from pilot_eval.workflow import _hash
        directory=root/'runs/diagnostics/pilot-4'/name;directory.mkdir(parents=True,exist_ok=True)
        config=dict(task_sha256=file_hash(grpo_controls.TASK_PATH), algorithm_sha256=file_hash(Path(grpo_algorithm.__file__)),
                    controls_sha256=file_hash(Path(grpo_controls.__file__)), checks_sha256=file_hash(Path(grpo_checks.__file__)))
        (directory/'config.json').write_text(json.dumps(config))
        (directory/'results').mkdir(exist_ok=True)
        result=dict(passed=True,algorithm_checks=dict(passed=True),runs=[dict(seed=seed,sign=sign,passed=True,signed_change=.4) for seed in [42,43] for sign in [1,-1]])
        (directory/'results/results.json').write_text(json.dumps(result))
        files={str(p.relative_to(directory)):file_hash(p) for p in directory.rglob('*') if p.is_file() and p.name!='complete.json'}
        (directory/'complete.json').write_text(json.dumps(dict(config_sha256=_hash(config),files=files)))
        return verify_controls(root,name)
    def load_preflight(self, plan, settings, directory):
        self.loads += 1
        return self.engine


def collect(root, path, baseline, options, deps):
    return main(['grpo-preflight', '--config', str(path), '--baseline-config', str(baseline),
                 '--controls-name', 'controls', '--settings', str(options), '--name', 'flight',
                 '--output-root', str(root)], dependencies=deps)


def test_public_preflight_freeze_and_verify_require_review_and_preserve_sources(tmp_path):
    path, baseline, options = fixture(tmp_path); deps = Dependencies()
    originals = {p:p.read_bytes() for p in [path,baseline]}
    assert collect(tmp_path,path,baseline,options,deps) == 0
    flight = tmp_path / 'plans/flight/grpo.preflight.json'
    config = json.loads(flight.read_text()); directory = tmp_path / config['run_path']
    result = json.loads((directory / 'results/results.json').read_text())
    assert result['passed'] and config['settings']['completion_limit'] == 4
    assert result['workload']['training_completions'] == 4096
    assert result['workload']['checkpoint_recovery_max_redo_steps'] == 32
    assert len(config['optimizer_groups']) == 64
    assert deps.engine.calls == 2
    assert collect(tmp_path,path,baseline,options,deps) == 0 and deps.loads == 1
    review = tmp_path / 'review.json'
    values = dict(reviewed=True, notes='Reviewed evidence', margin=.10,
                  preregistration=dict(author='researcher', written_at='2026-10-08T00:00:00+00:00',
                                      prediction='No material drop', falsifier='Drop exceeds margin'),
                  monitors={k:dict(threshold=t, action='pause') for k,t in
                            [('cap_fraction',.02),('dead_group_fraction',.8),('length_ratio_change_8_steps',.5)]})
    review.write_text(json.dumps({**values,'reviewed':False}))
    args=['grpo-freeze','--config',str(flight),'--review',str(review),'--name','frozen','--output-root',str(tmp_path)]
    assert main(args) == 1
    review.write_text(json.dumps(values)); assert main(args) == 0
    frozen = tmp_path / 'plans/frozen/grpo.frozen.json'
    assert main(['grpo-check-ready','--config',str(frozen),'--output-root',str(tmp_path)]) == 0
    assert all(p.read_bytes() == b for p,b in originals.items())
    # A changed input policy is not mixed with cached evidence.
    edited=json.loads(options.read_text()); edited['top_p']=.9; options.write_text(json.dumps(edited))
    assert collect(tmp_path,path,baseline,options,deps) == 1 and deps.loads == 1
    (directory/'results/results.json').write_text('{}')
    assert main(['grpo-check-ready','--config',str(frozen),'--output-root',str(tmp_path)]) == 1


@pytest.mark.parametrize('failure',['zero','repeat','memory','censored','missing'])
def test_failed_preflight_cannot_publish_success(tmp_path,failure):
    path,baseline,options=fixture(tmp_path); deps=Dependencies()
    if failure=='zero': deps.engine.mismatch=True
    if failure=='repeat':
        original=deps.engine.trial
        def varied(*args):
            result=original(*args); result['state_sha256']=str(deps.engine.calls); return result
        deps.engine.trial=varied
    if failure=='memory': deps.engine.capacity_fails=True
    if failure=='censored':
        config=json.loads(baseline.read_text()); summary=tmp_path/config['run_path']/'results/results.json'
        value=json.loads(summary.read_text());value['p99_censored']=True;summary.write_text(json.dumps(value))
    if failure=='missing':
        value=json.loads(options.read_text());value['scorer']=None;options.write_text(json.dumps(value))
    assert collect(tmp_path,path,baseline,options,deps)==1
    assert not list(tmp_path.glob('runs/pilot-4/*/preflight/flight/complete.json'))


def test_interrupted_capacity_reuses_verified_zero_and_trial_units(tmp_path):
    path,baseline,options=fixture(tmp_path);deps=Dependencies();deps.engine.capacity_fails=True
    assert collect(tmp_path,path,baseline,options,deps)==1
    assert deps.engine.calls==2
    deps.engine.capacity_fails=False
    assert collect(tmp_path,path,baseline,options,deps)==0
    assert deps.engine.calls==2


def test_real_preflight_backend_runs_disposable_trials_on_tiny_offline_model(tmp_path):
    from test_grpo_algorithm import torch_stack
    torch=torch_stack();torch.set_num_threads(1)
    from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from types import SimpleNamespace
    from pilot_eval.grpo_gpu import GPUPreflight
    from pilot_eval.sft import optimizer_settings,adapter_settings
    from pilot_eval.workflow import _hash
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=2,hidden_size=16,intermediate_size=32,num_hidden_layers=28,
        num_attention_heads=2,num_key_value_heads=2,attention_dropout=0,tie_word_embeddings=False,
        pad_token_id=None,bos_token_id=0,eos_token_id=1,attn_implementation='eager'))
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'bad':0,'good':1},unk_token='bad')),
                                     pad_token='bad',unk_token='bad',eos_token='good')
    tokenizer.padding_side='left';tokenizer.chat_template='tiny-test-template'
    sampler=SimpleNamespace(backend=SimpleNamespace(model=model,tokenizer=tokenizer),close=lambda:None)
    plan=dict(seed=42,optimizer=optimizer_settings(),adapter=adapter_settings(),
              source_prompt_contract=dict(chat_template_sha256=_hash(tokenizer.chat_template)))
    settings=dict(generation_groups=1,backward_groups=1,evaluation_batch_size=2,completion_limit=3,
                  top_p=1.,top_k=0,gradient_checkpointing=True)
    engine=GPUPreflight(plan,settings,tmp_path,sampler=sampler)
    rows=[dict(id=f'tiny:{i}',prompt='bad',input_ids=[0],prompt_tokens=1,gold='#### 1') for i in range(16)]
    zero=engine.zero_equivalence(rows[:2],settings)
    assert all(r['baseline_token_ids']==r['zero_token_ids'] for r in zero)
    first=engine.trial(rows,settings,142,tmp_path/'a')
    second=engine.trial(rows,settings,142,tmp_path/'b')
    assert first['state_sha256']==second['state_sha256']
    assert first['responses']==second['responses'] and len(first['responses'])==128
    assert first['base_before']==first['base_after']
    assert (tmp_path/'a/optimizer-state.pt').exists()
    engine.close()
