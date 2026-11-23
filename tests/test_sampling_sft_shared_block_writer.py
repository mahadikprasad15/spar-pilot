import json

from pilot_eval import sampling_sft_generation_v2 as generation
from pilot_eval import sampling_sft_pilot_v2 as pilot
from pilot_eval.sampling_sft_buffered_journal import BlockWriter
from test_sampling_sft_instrument import source_template
from test_sampling_sft_pilot import prepared, Dependencies as PilotDependencies
from test_sampling_sft_generation import Dependencies as GenerationDependencies


def test_pilot_and_generation_use_same_versioned_block_adapter(tmp_path,source_template,monkeypatch):
    calls=[]
    class Spy(BlockWriter):
        def record(self,value):calls.append(self.__class__);return super().record(value)
    monkeypatch.setattr(pilot,'BlockWriter',Spy)
    monkeypatch.setattr(generation,'BlockWriter',Spy)
    old=prepared(tmp_path,source_template)
    instrument=tmp_path/json.loads(old.read_text())['instrument_plan']
    replay=pilot.prepare(instrument,tmp_path,'shared-pilot')
    pilot.run(replay,tmp_path,'uninterrupted',dependencies=PilotDependencies(),stop_after_blocks=1)
    assert calls and all(c is Spy for c in calls)
    calls.clear()
    full=generation.prepare(old,tmp_path,'shared-generation')
    generation.run(full,tmp_path,0,dependencies=GenerationDependencies(),stop_after_blocks=1)
    assert calls and all(c is Spy for c in calls)
