"""The real saved reference workflow, with only tokenizer/data access substituted."""
import hashlib
import json
from pathlib import Path
from test_trace_corpus import Sources, review, digest, CharacterTokenizer


def make_sources(root):
    from pilot_eval.trace_corpus import prepare_corpus
    deps=Sources(root)
    corpus=prepare_corpus(deps.path,root,'corpus',review=review(),dependencies=deps)
    rows=[dict(id=f'q-{i}',source_index=i,question='total apples '*6,gold='answer #### 5 '*6,prompt='prompt') for i in range(512)]
    training=root/'inputs/training.json';training.write_text(json.dumps(rows))
    sft=root/'sft.json';sft.write_text(json.dumps(dict(model=deps.config['model'],model_revision='model-pin',tokenizer_revision='tokenizer-pin',training_items_path='inputs/training.json',training_items_sha256=digest(rows))))
    directory=root/'runs/grpo';(directory/'results').mkdir(parents=True)
    records=[dict(draw_id=f'draw-{i}',training_source_index=i//8,gold=rows[i//8]['gold'],prompt='prompt',text='reason answer '*6) for i in range(4096)]
    responses=directory/'results/responses.jsonl';responses.write_text(''.join(json.dumps(row)+'\n' for row in records))
    config=dict(model=deps.config['model'],model_revision='model-pin',tokenizer_revision='tokenizer-pin',run_path='runs/grpo')
    grpo=root/'grpo.json';grpo.write_text(json.dumps(config))
    (directory/'complete.json').write_text(json.dumps(dict(config_sha256=digest(config),files={'results/responses.jsonl':hashlib.sha256(responses.read_bytes()).hexdigest()})))
    return deps,corpus,sft,grpo


def test_reference_workflow_builds_three_frozen_references_and_reuses_completed_inputs(tmp_path):
    from pilot_eval.trace_reference_workflow import prepare_reference_run, verify_reference_run
    from pilot_eval.trace_vocabulary import TokenInventory
    deps,corpus,sft,grpo=make_sources(tmp_path)
    class Inventory(TokenInventory):
        def encode(self,text,add_special_tokens=False):return CharacterTokenizer().encode(text)
    class Dependencies:
        def load_inventory(self,config,root):return Inventory({i:chr(i-10) for i in range(42,137)})
        def stream_fineweb(self,config,revision):
            for i in range(10000):
                yield deps.document(i) if i<2000 else {'id':f'doc-{i}','text':f'{i:04d} '+'background '*100}
    path=prepare_reference_run(corpus,sft,grpo,tmp_path,'reference-test',dependencies=Dependencies())
    result=verify_reference_run(path,tmp_path)
    assert result['reference_names']==['ref_questions','ref_gold','ref_grpo']
    assert result['background_tokens']==2000000
    assert result['training_examples']==512 and result['grpo_completions']==4096
    assert result['reveal_allowed'] is False
    assert (path.parent/'top30.md').is_file()
    class NoAccess:
        def __getattr__(self,name):raise AssertionError('Completed preparation must reuse existing inputs.')
    assert prepare_reference_run(corpus,sft,grpo,tmp_path,'reference-test',dependencies=NoAccess())==path


def test_public_reference_verify_and_review_commands_work_in_fresh_cpu_process(tmp_path):
    import subprocess,sys
    from pilot_eval.trace_reference_workflow import prepare_reference_run
    from pilot_eval.trace_vocabulary import TokenInventory
    deps,corpus,sft,grpo=make_sources(tmp_path)
    class Inventory(TokenInventory):
        def encode(self,text,add_special_tokens=False):return CharacterTokenizer().encode(text)
    class Dependencies:
        def load_inventory(self,config,root):return Inventory({i:chr(i-10) for i in range(42,137)})
        def stream_fineweb(self,config,revision):
            for i in range(10000):yield deps.document(i) if i<2000 else {'id':f'doc-{i}','text':f'{i:04d} '+'background '*100}
    path=prepare_reference_run(corpus,sft,grpo,tmp_path,'cli-reference',dependencies=Dependencies())
    cmd=[sys.executable,'-m','pilot_eval','traces-reference-verify','--config',str(path),'--output-root',str(tmp_path)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stderr+result.stdout
    assert json.loads(result.stdout)['reference_review']=='pending'
    reviewed=subprocess.run([sys.executable,'-m','pilot_eval','traces-reference-review','--config',str(path),
        '--reviewer','Prasad','--notes','Read the synthetic reference preview.','--exposure-status','unknown',
        '--output-root',str(tmp_path)],capture_output=True,text=True)
    assert reviewed.returncode==0,reviewed.stderr+reviewed.stdout
    assert json.loads(reviewed.stdout)['inference_status']=='exploratory-or-unverified-exposure'
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['reference_review']=='reviewed'
