"""Narrow RunPod CLI adapter: inspect and stop only. No rent/delete API."""
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import time


class RunPodProvider:
    host_id=platform.node()

    def own_pod_id(self):
        value=os.environ.get('RUNPOD_POD_ID')
        if not value:raise ValueError('RunPod job must run on its own Pod (RUNPOD_POD_ID missing).')
        return value

    def inspect(self,pod_id):
        result=subprocess.run(['runpodctl','pod','get',pod_id],check=True,capture_output=True,text=True,timeout=30)
        value=json.loads(result.stdout)
        value=value.get('data',value)
        if not isinstance(value,dict) or value.get('id')!=pod_id:
            raise ValueError('Unrecognized RunPod CLI identity/schema; qualify the installed CLI before use.')
        return {key:value.get(key) for key in ['id','desiredStatus','runtimeStatus','networkVolumeId','costPerHr']}

    def stop(self,pod_id):
        if self.own_pod_id()!=pod_id:raise ValueError('Refusing to stop any Pod other than this job’s own Pod.')
        subprocess.run(['runpodctl','pod','stop',pod_id],check=True,timeout=30,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        return dict(id=pod_id,state='stop-command-returned') # Never claim self-confirmed shutdown.

    def run_job(self,request,root,log,timeout):
        environment=dict(os.environ,SAMPLING_SFT_ARTIFACT_ROOT=str(root))
        started=time.monotonic()
        log.parent.mkdir(parents=True,exist_ok=True)
        timed_out=False
        with log.open('wb') as stream:
            process=subprocess.Popen(request['command'],cwd=Path(__file__).resolve().parents[1],env=environment,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                returncode=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:returncode=process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    returncode=process.wait(timeout=5)
            except BaseException:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                raise
            finally:
                # The command has ended: no owned background inference worker may remain.
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        return dict(returncode=returncode,elapsed_seconds=time.monotonic()-started,timed_out=timed_out)
