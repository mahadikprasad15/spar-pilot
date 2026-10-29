"""Public batch agreement -> review -> freeze -> paired evaluation, CPU only."""
import json

import pytest

from test_stage0_evaluation import frozen_root, source, TinyEvaluation
from pilot_eval.cli import main
from pilot_eval.training import file_hash
from pilot_eval.workflow import _hash
from pilot_eval.stage0_workflow import record_budget
from pilot_eval.stage0_budget import settle, reservations


def command(root, op, *args, dependencies=None):
    return main(['stage0-batch-' + op, '--output-root', str(root), *map(str, args)],
                dependencies=dependencies)


def observed(root, event, total):
    path = root / (event + '.json')
    path.write_text(json.dumps(dict(event_id=event, session_id='session-a', total_units=total,
                    observed_at='2026-10-29T00:00:00+00:00', note='fixture usage')))
    record_budget(root / 'plans/audit/stage0.plan.json', root, path)


def prepared(root,legacy_lock=False):
    observed(root, 'release-old', 1.03)
    for rid, row in reservations(root).items():
        if row['state'] == 'active': settle(root, rid)
    qpath = root / 'plans/qualify/stage0.qualification.json'
    q = json.loads(qpath.read_text())
    profile = root / 'capacity'
    profile.mkdir()
    config = dict(protocol='stage0-batch-performance-profile-v1',runtime=json.loads((root / q['run_path'] / 'meta/runtime.json').read_text()),
                  qualification_sha256=file_hash(qpath), variants=q['variants'],
                  decoding={**q['decoding'], 'use_cache': True})
    (profile / 'config.json').write_text(json.dumps(config))
    (profile / 'summary.json').write_text(json.dumps([dict(batch_size=16, memory={'fits': True},
                    exact_singleton_token_agreement=True, production_authorized=False)]))
    files={'summary.json': file_hash(profile / 'summary.json')}
    if legacy_lock:files['writer.json']=_hash('temporary writer removed at successful exit')
    (profile / 'complete.json').write_text(json.dumps(dict(config_sha256=_hash(config), files=files)))
    assert command(root, 'prepare', '--qualification', qpath, '--profile', profile,
                   '--name', 'batch16', '--batch-size', 16) == 0
    return root / 'plans/batch16/stage0.batch.json'


class BatchOracle(TinyEvaluation):
    def __init__(self, fail_after=None, error=1e-7):
        super().__init__(); self.failure = fail_after; self.error = error; self.units = 0
    def compare(self, rows, variant, token_rows):
        if self.failure is not None and self.units >= self.failure:
            raise RuntimeError('injected batch qualification interruption')
        self.units += 1
        return dict(max_absolute_error=self.error,
                    examples=[dict(id=row['id'], errors=[self.error], positions=[0]) for row in rows])
    def generate(self, rows, variant, seed):
        self.calls.append((variant, [r['id'] for r in rows]))
        return [dict(text='1 2', token_ids=[1, 2], generated_token_ids=[1, 2], stop_reason='eos') for row in rows]
    def check_integrity(self, full=False):
        return dict(base_unchanged=True, adapters_unchanged=True)


def test_legacy_profile_lock_is_not_a_durable_input_but_missing_data_still_fails(source):
    plan=prepared(source,legacy_lock=True)
    assert command(source,'status','--config',plan)==0
    profile=source/'capacity'
    (profile/'writer.json').write_text('{}')
    assert command(source,'status','--config',plan)==1
    (profile/'writer.json').unlink()
    (profile/'summary.json').unlink()
    assert command(source,'status','--config',plan)==1


def test_public_batch_agreement_resumes_and_preserves_scope(source):
    plan = prepared(source)
    config = json.loads(plan.read_text())
    assert config['qualified_sizes'] == [16, 6]
    approval = source / 'batch-budget.json'
    approval.write_text(json.dumps(dict(reviewer='test', notes='fixture', session_id='session-a',
                    max_units_per_hour=1., max_seconds=600., source_reviewed=True)))
    engine = BatchOracle()
    assert command(source, 'run', '--config', plan, '--approval', approval,
                   '--max-new-units', 1, dependencies=engine) == 0
    directory = source / config['run_path']
    sealed_path = next((directory / 'units').glob('*/complete.json'))
    sealed = sealed_path.read_bytes()
    observed(source, 'after-first', 1.04)
    for rid, row in reservations(source).items():
        if row['state'] == 'active': settle(source, rid)
    assert command(source, 'run', '--config', plan, '--approval', approval, dependencies=engine) == 0
    assert command(source, 'verify', '--config', plan) == 0
    report = json.loads((directory / 'results/agreement.json').read_text())
    assert report['eligible_for_review'] and not report['production_authorized']
    assert report['absolute_gate'] == pytest.approx(6e-6)
    assert sealed_path.read_bytes() == sealed
    assert report['variant_count'] == 10
    assert report['forecast']['batch_sizes'] == [16, 6]
    review=source/'batch-review.json'
    review.write_text(json.dumps(dict(reviewer='test',notes='exact saved evidence',
        agreement_sha256=file_hash(directory/'complete.json'),absolute_gate=report['absolute_gate'])))
    assert command(source,'approve','--config',plan,'--review',review)==0
    observed(source,'after-agreement',1.05)
    for rid,row in reservations(source).items():
        if row['state']=='active':settle(source,rid)
    production=source/'batch-production.json'
    production.write_text(json.dumps(dict(reviewer='test',notes='bounded production',
        agreement_sha256=file_hash(directory/'complete.json'),max_units_per_hour=1.,
        max_seconds=max(60.,report['forecast']['conservative_total_seconds']),session_id='session-a',source_reviewed=True)))
    assert command(source,'freeze','--config',plan,'--review',production)==0
    execution=plan.parent/'stage0.execution.json'
    assert main(['stage0-evaluation-prepare','--output-root',str(source),'--execution',str(execution),
        '--batch-size','16','--batches-per-unit','1'])==0
    evaluation=json.loads((plan.parent/'stage0.evaluation.json').read_text())
    assert len(evaluation['units'])==100
    assert len(evaluation['units'][-1]['ids'])==6
    assert sum(len(u['ids']) for u in evaluation['units'])==1500
    assert main(['stage0-evaluation-run','--output-root',str(source),'--config',str(plan.parent/'stage0.evaluation.json'),
        '--max-new-units','1'],dependencies=TinyEvaluation())==0
    assert main(['stage0-evaluation-prepare','--output-root',str(source),'--execution',str(execution),
        '--batch-size','8'])==1
    fixed=[]
    for corpus in ['gsm8k','fineweb']:
        for i in range(150):
            masks=dict(question=[False,True,False],solution=[False,False,True],user=[False]*3) if corpus=='gsm8k' else dict(question=[False]*3,solution=[False]*3,user=[True]*3)
            fixed.append(dict(id=f'{corpus}:{i}',corpus=corpus,input_ids=[1,2,3],attention_mask=[1]*3,masks=masks))
    inputs=source/'fixed';inputs.mkdir();(inputs/'items.json').write_text(json.dumps(fixed))
    prepared_inputs=inputs/'activation.prepared.json'
    prepared_inputs.write_text(json.dumps(dict(items_path='fixed/items.json',items_sha256=_hash(fixed),
        model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='pinned',tokenizer_revision='pinned',dtype='float32')))
    assert main(['stage0-drift-prepare','--output-root',str(source),'--execution',str(execution),
        '--inputs',str(prepared_inputs),'--name','internals'])==0
    drift=source/'plans/internals/stage0.drift.json'
    assert main(['stage0-findings-prepare','--output-root',str(source),'--evaluation',str(plan.parent/'stage0.evaluation.json'),
        '--drift',str(drift),'--name','joined'])==0
    joined=json.loads((source/'plans/joined/stage0.findings.json').read_text())
    assert joined['inference_schedules']['behaviour_batch_size']==16
    assert joined['inference_schedules']['internals_batch_size']==1
    receipt=source/evaluation['batch_agreement']['receipt_path'];receipt.write_text('{}')
    assert main(['stage0-findings-prepare','--output-root',str(source),'--evaluation',str(plan.parent/'stage0.evaluation.json'),
        '--drift',str(drift),'--name','tampered'])==1


def test_numeric_gate_failure_saves_evidence_and_never_widens_it(source):
    plan = prepared(source)
    approval = source / 'batch-budget.json'
    approval.write_text(json.dumps(dict(reviewer='test', notes='fixture', session_id='session-a',
                    max_units_per_hour=1., max_seconds=600., source_reviewed=True)))
    assert command(source, 'run', '--config', plan, '--approval', approval,
                   dependencies=BatchOracle(error=1e-3)) == 1
    directory = source / json.loads(plan.read_text())['run_path']
    assert list((directory / 'units').glob('*/complete.json'))
    assert not (directory / 'complete.json').exists()
    assert json.loads((directory / 'meta/status.json').read_text())['state'] == 'failed'
