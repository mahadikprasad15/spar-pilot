"""ADR 0017: matched-128 jobs keep reservations and time caps but no cumulative cap."""
import json

import pytest


def ledger(root,total,event='reading'):
    from test_stage0_workflow import write_json
    from pilot_eval.stage0_workflow import record_budget
    path=root/f'{event}.json';write_json(path,dict(event_id=event,session_id='s',total_units=total,
        observed_at='2026-10-30T12:00:00+00:00',note='synthetic cumulative reading'))
    record_budget(root/'plans/example/stage0.plan.json',root,path)


def approval(**changes):
    return dict(dict(reviewer='test',notes='bounded job',session_id='s',source_reviewed=True,
                     max_units_per_hour=1.6,max_seconds=3600.),**changes)


@pytest.fixture
def root(tmp_path):
    from test_stage0_workflow import prepared
    prepared(tmp_path)
    return tmp_path


def test_reservation_is_allowed_beyond_twenty_units_and_records_the_policy(root):
    from pilot_eval.stage0_job_budget import reserve,POLICY
    ledger(root,21.)
    row=reserve(root,'job',approval())
    assert row['state']=='active' and row['starting_units']==21. and row['reserved_units']==1.6
    assert row['budget_policy']==POLICY and 'ADR 0017' in POLICY


def test_reservation_still_requires_an_observation_one_active_job_and_a_bounded_approval(root):
    from pilot_eval.stage0_job_budget import reserve
    with pytest.raises(ValueError,match='observation'):reserve(root,'job',approval())
    ledger(root,21.)
    for bad in [approval(max_seconds=0),approval(source_reviewed=False),approval(max_units_per_hour=float('inf'))]:
        with pytest.raises(ValueError,match='Approve'):reserve(root,'job',bad)
    reserve(root,'job',approval())
    with pytest.raises(ValueError,match='active reservation'):reserve(root,'other',approval())


def test_capped_stage0_reservations_are_unchanged(root):
    from pilot_eval.stage0_budget import reserve
    ledger(root,21.)
    with pytest.raises(ValueError,match='20-unit'):reserve(root,'job',approval())
