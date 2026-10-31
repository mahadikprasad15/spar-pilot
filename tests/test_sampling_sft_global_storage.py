"""Provider boundary: v2 storage discovery, no live cloud calls."""
import io
import json
from types import SimpleNamespace
import pytest
from pilot_eval.sampling_sft_provider import RunPodProvider


def test_global_volume_inspection_uses_v2_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY', 'private-test-token')
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps(dict(id='pod', networkVolumeId=None, costPerHr=0.59))))
    def fetch(request, timeout):
        assert request.full_url == 'https://api.runpod.io/v2/pods/pod'
        assert request.get_header('Authorization') == 'Bearer private-test-token'
        return io.BytesIO(json.dumps(dict(id='pod',status='RUNNING',cost=0.59,
            mounts={'global':[dict(volumeId='global-volume',path='/workspace')]},
            env={'SECRET':'do-not-export'})).encode())
    monkeypatch.setattr('urllib.request.urlopen', fetch)
    result=RunPodProvider().inspect('pod')
    assert result['storageVolumeId']=='global-volume'
    assert result['storageMountPath']=='/workspace'
    assert result['storageType']=='global'
    assert result['desiredStatus']=='RUNNING'
    assert 'env' not in result and 'private-test-token' not in json.dumps(result)


@pytest.mark.parametrize('current', [
    dict(id='another',mounts={'global':[dict(volumeId='v',path='/workspace')]}),
    dict(id='pod',mounts={}),
    dict(id='pod',mounts={'global':[dict(volumeId='v')]}),
    dict(id='pod',mounts={'global':[dict(volumeId='v',path='/workspace')],
                          'network':[dict(volumeId='w',path='/other')]}),
])
def test_v2_inspection_rejects_wrong_or_ambiguous_storage(monkeypatch,current):
    monkeypatch.setenv('RUNPOD_API_KEY','private-test-token')
    monkeypatch.setattr('subprocess.run',lambda *a,**kw:SimpleNamespace(stdout=json.dumps(dict(id='pod'))))
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:io.BytesIO(json.dumps(current).encode()))
    with pytest.raises(ValueError):
        RunPodProvider().inspect('pod')
