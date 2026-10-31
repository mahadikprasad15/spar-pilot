"""Public dollar-budget controls; no provider calls or model imports."""
import json
from pathlib import Path

import pytest

from pilot_eval.cli import main
from pilot_eval.workflow import _hash


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def setup(root):
    selection=write(root/'selection.json',dict(sft='missing-sft.json',grpo='missing-grpo.json'))
    assert main(['sampling-sft-prepare','--selection',str(selection),'--output-root',str(root),'--name','budget-test'])==0
    plan=root/'plans/budget-test/sampling-sft.plan.json'
    assert cli(root,'init','--config',plan)==0
    return plan


def cli(root,operation,*args,**kwargs):
    return main(['sampling-sft-budget-'+operation,'--output-root',str(root),*map(str,args)],**kwargs)


def gpu(job_id='job',rate='1',hours='1',stage='sampling'):
    return dict(job_id=job_id,kind='gpu',stage=stage,rate_usd_per_hour=rate,
                estimated_hours=hours,setup_hours='0.1')


def reserve(root,plan,request,approval=None):
    path=write(root/(request['job_id']+'-request.json'),request)
    args=['--config',plan,'--request',path]
    if approval is not None:args+=['--approval',write(root/'review.json',approval)]
    return cli(root,'reserve',*args)


def ledger(root):
    from pilot_eval.sampling_sft_budget import status
    return status(root)


def settle(root,job_id,cost,event='bill'):
    observation=write(root/(event+'.json'),dict(event_id=event,actual_usd=cost,
        evidence='provider billing observation',reviewer='human',notes='Includes setup, idle and work.'))
    return cli(root,'settle','--job-id',job_id,'--observation',observation)


def test_public_budget_reserves_reconciles_and_reuses_without_double_charging(tmp_path):
    plan=setup(tmp_path)
    request=gpu()
    assert reserve(tmp_path,plan,request)==0
    value=ledger(tmp_path)
    assert value['reserved_usd']=='1.600000'
    assert value['spent_usd']=='0.000000'
    assert value['stage_totals']['sampling']['committed_usd']=='1.600000'
    assert reserve(tmp_path,plan,request)==0
    assert ledger(tmp_path)['reserved_usd']=='1.600000'
    assert settle(tmp_path,'job','0.75')==0
    assert settle(tmp_path,'job','0.75')==0
    assert ledger(tmp_path)['spent_usd']=='0.750000'
    assert ledger(tmp_path)['reserved_usd']=='0.000000'
    assert cli(tmp_path,'status')==0


def test_new_plan_name_cannot_reset_the_shared_cap(tmp_path):
    plan=setup(tmp_path)
    assert reserve(tmp_path,plan,gpu('a',hours='4'))==0
    assert main(['sampling-sft-prepare','--selection',str(tmp_path/'selection.json'),
       '--output-root',str(tmp_path),'--name','another'])==0
    other=tmp_path/'plans/another/sampling-sft.plan.json'
    assert cli(tmp_path,'init','--config',other)==0
    assert reserve(tmp_path,other,gpu('b',hours='3'))==1
    assert ledger(tmp_path)['reserved_usd']=='6.100000'


def test_80_percent_requires_current_bound_review_but_cannot_raise_cap(tmp_path):
    plan=setup(tmp_path)
    request=gpu('large',hours='5.3') #8.05 with setup, above80% sampling
    assert reserve(tmp_path,plan,request)==1
    value=ledger(tmp_path)
    approval=dict(reviewer='Prasad',notes='Reviewed measured estimate',budget_sha256=value['sha256'],
                  request_sha256=_hash(request),review_80_percent=True)
    assert reserve(tmp_path,plan,request,approval)==0
    assert settle(tmp_path,'large','8.0','large-bill')==0
    assert reserve(tmp_path,plan,gpu('next',hours='0.1'))==1
    request=gpu('over',hours='7')
    approval.update(budget_sha256=ledger(tmp_path)['sha256'],request_sha256=_hash(request))
    assert reserve(tmp_path,plan,request,approval)==1


def test_uncertain_reservation_survives_restart_and_changed_request_is_rejected(tmp_path):
    plan=setup(tmp_path)
    assert reserve(tmp_path,plan,gpu())==0
    assert cli(tmp_path,'init','--config',plan)==0
    changed=gpu();changed['estimated_hours']='0.01'
    assert reserve(tmp_path,plan,changed)==1
    assert ledger(tmp_path)['reserved_usd']=='1.600000'


def test_storage_lifetime_extension_and_stage_allocation(tmp_path):
    plan=setup(tmp_path)
    request=dict(job_id='volume',kind='storage',stage='storage_shared',volume_id='volume-1',
                 rate_usd_per_month='4',months='1')
    assert reserve(tmp_path,plan,request)==0
    assert ledger(tmp_path)['stage_totals']['storage_shared']['reserved_usd']=='4.000000'
    extension=write(tmp_path/'extension.json',dict(additional_months='0.5'))
    assert cli(tmp_path,'extend','--job-id','volume','--request',extension)==0
    assert ledger(tmp_path)['reserved_usd']=='6.000000'
    bad=gpu('misclassified',stage='storage_shared')
    assert reserve(tmp_path,plan,bad)==1


def test_actual_overrun_is_recorded_truthfully_and_blocks_new_work(tmp_path):
    plan=setup(tmp_path)
    assert reserve(tmp_path,plan,gpu())==0
    assert settle(tmp_path,'job','11')==0
    assert ledger(tmp_path)['spent_usd']=='11.000000'
    assert ledger(tmp_path)['cap_exceeded'] is True
    assert reserve(tmp_path,plan,gpu('new',hours='0.01'))==1


def test_tampered_or_missing_ledger_cannot_be_reinitialized(tmp_path):
    plan=setup(tmp_path)
    assert reserve(tmp_path,plan,gpu())==0
    path=tmp_path/'runs/sampling-sft/budget/budget.json'
    value=json.loads(path.read_text());value['events']=[];write(path,value)
    assert cli(tmp_path,'status')==1
    assert cli(tmp_path,'init','--config',plan)==1
    path.unlink()
    assert cli(tmp_path,'init','--config',plan)==1


@pytest.mark.parametrize('field,value',[('estimated_hours','NaN'),('rate_usd_per_hour','-1'),
                                       ('setup_hours',True),('estimated_hours','Infinity')])
def test_nonfinite_negative_boolean_estimates_are_rejected(tmp_path,field,value):
    plan=setup(tmp_path);request=gpu();request[field]=value
    assert reserve(tmp_path,plan,request)==1


def test_total_and_stage_caps_count_all_outstanding_reservations(tmp_path):
    plan=setup(tmp_path)
    requests=[gpu('s',hours='6.5'),gpu('t',hours='3.8',stage='training'),
       dict(job_id='v',kind='storage',stage='storage_shared',volume_id='v',rate_usd_per_month='9',months='1')]
    for request in requests:
        approval=dict(reviewer='human',notes='Reviewed80%',budget_sha256=ledger(tmp_path)['sha256'],
                      request_sha256=_hash(request),review_80_percent=True)
        assert reserve(tmp_path,plan,request,approval)==0
    assert ledger(tmp_path)['committed_usd']=='24.650000'
    request=gpu('more',hours='0.3',stage='training')
    approval.update(budget_sha256=ledger(tmp_path)['sha256'],request_sha256=_hash(request))
    assert reserve(tmp_path,plan,request,approval)==1


def test_partial_storage_bill_retains_expected_remaining_lifetime(tmp_path):
    plan=setup(tmp_path)
    request=dict(job_id='volume',kind='storage',stage='storage_shared',volume_id='v',
                 rate_usd_per_month='4',months='1')
    assert reserve(tmp_path,plan,request)==0
    observation=write(tmp_path/'partial.json',dict(event_id='week1',actual_usd='1.2',covered_months='0.25',
        evidence='provider bill for first quarter month',reviewer='human',notes='Volume remains allocated'))
    assert cli(tmp_path,'settle','--job-id','volume','--observation',observation)==0
    assert cli(tmp_path,'settle','--job-id','volume','--observation',observation)==0
    current=ledger(tmp_path)
    assert current['spent_usd']=='1.200000'
    assert current['reserved_usd']=='3.000000'
    assert current['committed_usd']=='4.200000'


def test_simultaneous_reservations_cannot_overbook_a_stage(tmp_path):
    import subprocess,sys
    plan=setup(tmp_path)
    paths=[write(tmp_path/f'{i}.json',gpu(str(i),hours='4')) for i in range(2)]
    script='from pilot_eval.cli import main; import sys; raise SystemExit(main(sys.argv[1:]))'
    children=[subprocess.Popen([sys.executable,'-c',script,'sampling-sft-budget-reserve','--config',str(plan),
              '--output-root',str(tmp_path),'--request',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for path in paths]
    codes=[]
    for child in children:
        child.communicate(timeout=30);codes.append(child.returncode)
    assert sorted(codes)==[0,1]
    assert ledger(tmp_path)['reserved_usd']=='6.100000'


def test_interim_gpu_cost_observation_does_not_release_the_reservation(tmp_path):
    plan=setup(tmp_path)
    assert reserve(tmp_path,plan,gpu())==0
    observation=write(tmp_path/'interim.json',dict(event_id='meter',actual_usd='2',
        evidence='provider running-total meter',reviewer='human',notes='Cumulative charges; Pod still running'))
    assert cli(tmp_path,'observe','--job-id','job','--observation',observation)==0
    assert cli(tmp_path,'observe','--job-id','job','--observation',observation)==0
    assert ledger(tmp_path)['spent_usd']=='2.000000'
    assert ledger(tmp_path)['reserved_usd']=='1.600000'


def test_explicit_request_bound_waiver_preserves_costs_and_allows_launch(tmp_path):
    from pilot_eval.sampling_sft_budget import authorize_launch
    plan = setup(tmp_path)
    request = gpu('waived', hours='20')
    assert reserve(tmp_path, plan, request) == 1
    approval = dict(reviewer='Prasad', notes='User withdrew caps; preserve accounting.',
                    waive_budget_limits=True, budget_sha256=ledger(tmp_path)['sha256'],
                    request_sha256=_hash(request))
    stale = dict(approval, request_sha256='wrong')
    assert reserve(tmp_path, plan, request, stale) == 1
    assert reserve(tmp_path, plan, request, approval) == 0
    summary = authorize_launch(tmp_path, 'waived', request, pod_id='fixture-pod')
    assert summary['cap_exceeded'] is True
    assert summary['reserved_usd'] == '30.100000'
    assert summary['jobs']['waived']['budget_limits_waived'] is True
    assert summary['jobs']['waived']['launch_recorded'] is True
    assert reserve(tmp_path, plan, gpu('ordinary')) == 1
