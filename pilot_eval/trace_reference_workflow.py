"""CPU saved workflow for approved Pilot 5 references; never reads learned deltas."""
from collections import Counter
import json
import subprocess
from pathlib import Path
from pilot_eval.trace_corpus import load_corpus,HFCorpusDependencies,_inside
from pilot_eval.trace_vocabulary import TokenInventory,prepare_background,build_reference,freeze_references,load_references
from pilot_eval.training import file_hash,run_lock
from pilot_eval.workflow import _hash,_save_frozen
from pilot_eval.run import _write_json,_write_state
from pilot_eval.sft import safe_name


def _implementation():
    directory=Path(__file__).parent
    return {name:file_hash(directory/name) for name in ('trace_vocabulary.py','trace_reference_workflow.py')}


def approved_procedure():
    """Values approved in design v1.1 plus preflight section 6, 15 October 2026."""
    return dict(version='pilot5-overlap-v1.1-preflight',approval='Prasad: approve the preflight defaults, 2026-10-15',
        normalization='NFKC-strip-lower; exclude all added/special IDs, empty and U+FFFD',
        alpha0=1000,sensitivity_alpha0=[100,10000],minimum_task_count=5,positive_z_only=True,reference_size=200,
        primary_reference='ref_questions',primary_vector=dict(form='raw',block=14,step=64,positions=[0,1,2,3,4],pooling='per-document-then-cohort'),
        ranking=dict(sign='positive',unique_normalized_strings=20,maximum_rank=1000,ties='ascending-token-id',zero_valid='undefined'),
        null=dict(draws=1000,seed=20261015,unit_norm=True,real_arrays='after-qualified-inputs-before-learned-publication'),
        correction=dict(c1_c2_family=4,c3_family=2,method='Holm',alpha=.05),
        bootstrap=dict(primary_draws=1000,c3_draws=2000,stratum_draws=2000,seed=20261016,quantile_method='linear',c3='two-sided-declared-bootstrap-tail-formula'),
        ncurves=dict(sizes=[25,50,100,200],draws=200,seed=20261017),
        matched_legacy=dict(size=50,draws=1000,seed=20261020),
        snr='norm(mean)/sqrt(sum squared document deviations/(N*(N-1))); undefined N<2 or SE=0',
        grading=dict(seed=20261018,expected_lists_per_mode=106,external_calls=False,naming_before_relevance=True,kappa_warning=.4),
        primary_missing_lens='stop and seek separate scope approval',user_form='deferred',assistant_form='optional separate execution')


class ReferenceDependencies(HFCorpusDependencies):
    def load_inventory(self,corpus,root):
        return TokenInventory.from_file(_inside(root,corpus['tokenizer_files']['tokenizer.json']['path']))


def verify_reference_run(path,output_root):
    root=Path(output_root).resolve();path=_inside(root,path);directory=path.parent.parent
    complete=directory/'complete.json'
    if not complete.is_file():raise ValueError('Reference preparation is not complete.')
    for relative,expected in json.loads(complete.read_text())['files'].items():
        if file_hash(_inside(directory,relative))!=expected:raise ValueError('Reference run hash mismatch.')
    payload=load_references(path);config=json.loads((directory/'config.json').read_text())
    if config['implementation']!=_implementation():
        raise ValueError('Reference builder implementation changed; inspect the old execution before reuse.')
    reviewed=(path.parent/'review.json').exists()
    if reviewed:load_references(path,require_review=True)
    return dict(reference_names=['ref_questions','ref_gold','ref_grpo'],background_tokens=2000000,
                procedure_plan=str(root/config['procedure_plan']),
                training_examples=512,grpo_completions=4096,reference_sizes={k:len(v['strings']) for k,v in payload['references'].items()},
                reveal_allowed=False,reference_review='reviewed' if reviewed else 'pending',
                reference_gate_passed=reviewed,remaining_gate='qualified capture and readout consumer',
                freeze_path=str(path),procedure=config['procedure'])


def prepare_reference_run(corpus_path,sft_path,grpo_path,output_root,name,*,dependencies=None):
    """Prepare three references from sealed text inputs and an independent background."""
    safe_name(name);root=Path(output_root).resolve()
    corpus_path,sft_path,grpo_path=(_inside(root,p) for p in (corpus_path,sft_path,grpo_path))
    directory=root/'runs/pilot-5'/name/'analysis/ticket03';freeze=directory/'references/freeze.json'
    with run_lock(directory):
        if (directory/'complete.json').exists():
            verify_reference_run(freeze,root)
            saved=json.loads((directory/'config.json').read_text())
            if saved['source_paths']!=[str(p.relative_to(root)) for p in (corpus_path,sft_path,grpo_path)]:
                raise ValueError('Reference source selection changed; use a separately reviewed name.')
            return freeze
        corpus,measurement=load_corpus(corpus_path,root)
        sft=json.loads(sft_path.read_text());grpo=json.loads(grpo_path.read_text())
        for key in ('model','model_revision','tokenizer_revision'):
            if sft[key]!=corpus[key] or grpo[key]!=corpus[key]:raise ValueError('Reference source model/tokenizer identity mismatch.')
        training_path=_inside(root,sft['training_items_path']);training=json.loads(training_path.read_text())
        if _hash(training)!=sft['training_items_sha256'] or len(training)!=512 or len({r['source_index'] for r in training})!=512:
            raise ValueError('The exact 512 gold training examples are required.')
        run=_inside(root,grpo['run_path']);responses=run/'results/responses.jsonl';seal_path=run/'complete.json'
        seal=json.loads(seal_path.read_text())
        if seal['config_sha256']!=_hash(grpo) or seal['files'].get('results/responses.jsonl')!=file_hash(responses):
            raise ValueError('Accepted GRPO response seal mismatch.')
        records=[json.loads(line) for line in responses.read_text().splitlines()];by={r['source_index']:r for r in training}
        if (len(records)!=4096 or len({r['draw_id'] for r in records})!=4096
                or set(r['training_source_index'] for r in records)!=set(by)
                or any(r['gold']!=by[r['training_source_index']]['gold'] or r['prompt']!=by[r['training_source_index']]['prompt'] for r in records)):
            raise ValueError('Accepted GRPO completions do not match the shared training cohort.')
        parents={'corpus':corpus_path,'sft':sft_path,'grpo':grpo_path,'training':training_path,'responses':responses,'training_seal':seal_path,
                 'measurement':_inside(root,corpus['items_path'])}
        parents.update({f'tokenizer-{key}':_inside(root,value['path']) for key,value in corpus['tokenizer_files'].items()})
        config=dict(schema_version=1,procedure=approved_procedure(),source_paths=[str(p.relative_to(root)) for p in (corpus_path,sft_path,grpo_path)],
                    implementation=_implementation(),
                    procedure_plan=f'plans/{name}-overlap-v1/overlap.config.json',run_path=str(directory.relative_to(root)),
                    parent_hashes={key:file_hash(p) for key,p in parents.items()},prior_exposure=corpus['review']['prior_exposure'],
                    reviewer='Prasad',review_date='2026-10-15',verification_scope='sealed text/input parents; not a full optimizer/checkpoint audit')
        existing=directory/'config.json'
        if existing.exists():
            config['origin_git_commit']=json.loads(existing.read_text())['origin_git_commit']
        else:
            revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).parent.parent,
                                    capture_output=True,text=True)
            config['origin_git_commit']=revision.stdout.strip() if revision.returncode==0 else 'unavailable'
        _save_frozen(directory/'config.json',config);_write_state(directory,'running',0,3)
        _save_frozen(directory/'meta/run_manifest.json',config)
        plan_path=_inside(root,config['procedure_plan'])
        _save_frozen(plan_path,config)
        parents['procedure_plan']=plan_path
        parents['run_manifest']=directory/'meta/run_manifest.json'
        try:
            deps=dependencies or ReferenceDependencies()
            inventory=deps.load_inventory(corpus,root)
            bgdir=directory/'background'
            stream=None if (bgdir/'complete.json').exists() else deps.stream_fineweb(corpus['fineweb']['config'],corpus['fineweb']['revision'])
            background=prepare_background(bgdir,stream,inventory,measurement_indices=[r['source_index'] for r in measurement],
                measurement_sources=[{key:r.get(key) for key in ('source_index','source_id','full_document_sha256')} for r in measurement],
                tokenizer_identity=dict(revision=corpus['tokenizer_revision'],files=corpus['tokenizer_files'],fineweb=corpus['fineweb']['revision']))
            parents['background']=bgdir/'results/background.json';parents['background_seal']=bgdir/'complete.json'
            counts=Counter()
            for row in background['documents']:counts.update(inventory.count(row['token_ids']))
            task_sources={'ref_questions':[row['question'] for row in training],'ref_gold':[row['gold'] for row in training],'ref_grpo':[row['text'] for row in records]}
            references={};sensitivities={str(alpha):{} for alpha in (100,10000)}
            for index,(key,texts) in enumerate(task_sources.items()):
                print(f'References: counting {key}, {len(texts)} source texts.',flush=True)
                task=Counter()
                for text in texts:task.update(inventory.count(inventory.encode(text,add_special_tokens=False)))
                count_path=directory/'counts'/f'{key}.json'
                _save_frozen(count_path,dict(task=dict(task),background=dict(counts)))
                parents[key+'-counts']=count_path
                ref=build_reference(dict(task),dict(counts));ref['scores']=[r for r in ref['scores'] if r['y_a']+r['y_b']>=5]
                references[key]=ref
                for alpha in (100,10000):
                    extra=build_reference(dict(task),dict(counts),alpha0=alpha)
                    selected=set(extra['strings']);extra['scores']=[r for r in extra['scores'] if r['string'] in selected]
                    sensitivities[str(alpha)][key]=extra
                _write_state(directory,'running',index+1,3)
            parameters=dict(procedure=config['procedure'],sensitivities=sensitivities,source_counts={key:len(value) for key,value in task_sources.items()})
            freeze_references(freeze.parent,references,parameters=parameters,input_files=parents)
            lines=['# Task-enriched references: review before learned readout reveal','']
            for key,ref in references.items():
                lines+=['## '+key,f"Reference size: {len(ref['strings'])}; short-reference flag: {ref['short_reference']}",'','| String | y_A | y_B | z |','|---|---:|---:|---:|']
                selected={r['string']:r for r in ref['scores']}
                for word in ref['strings'][:30]:
                    row=selected[word];display=word.replace('|','\\|').replace('\n','\\n')
                    lines.append(f"| {display} | {row['y_a']} | {row['y_b']} | {row['z']:.9g} |")
                lines.append('')
            (freeze.parent/'top30.md').write_text('\n'.join(lines)+'\n')
            paths=[directory/'config.json',directory/'meta/run_manifest.json',freeze,freeze.parent/'freeze-seal.json',freeze.parent/'top30.md']
            _write_json(directory/'complete.json',{'files':{str(p.relative_to(directory)):file_hash(p) for p in paths}})
            _write_state(directory,'awaiting-review',3,3)
            return freeze
        except BaseException as exc:
            _write_state(directory,'failed',0,3,str(exc));raise
