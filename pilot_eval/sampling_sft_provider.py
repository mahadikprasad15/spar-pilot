"""Narrow RunPod CLI adapter: inspect and stop only. No rent/delete API."""
import json
import os
import platform
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
        return {key:value.get(key) for key in ['id','desiredStatus','runtimeStatus','networkVolumeId']}

    def stop(self,pod_id):
        if self.own_pod_id()!=pod_id:raise ValueError('Refusing to stop any Pod other than this job’s own Pod.')
        subprocess.run(['runpodctl','pod','stop',pod_id],check=True,timeout=30)
        return dict(id=pod_id,state='stop-command-returned') # Never claim self-confirmed shutdown.

    def run_job(self,request,root,log,timeout):
        environment=dict(os.environ,SAMPLING_SFT_ARTIFACT_ROOT=str(root))
        started=time.monotonic()
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('wb') as stream:
            result=subprocess.run(request['command'],cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        return dict(returncode=result.returncode,elapsed_seconds=time.monotonic()-started)
