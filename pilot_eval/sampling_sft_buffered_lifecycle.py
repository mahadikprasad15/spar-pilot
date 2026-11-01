"""Prepare isolated v2 lifecycle jobs; preparation never launches a GPU."""
import json
import sys
from pathlib import Path

from pilot_eval import sampling_sft as source
from pilot_eval import sampling_sft_budget as budget
from pilot_eval import sampling_sft_jobs as jobs
from pilot_eval import sampling_sft_pilot_v2 as pilot
from pilot_eval import sampling_sft_generation_v2 as generation
from pilot_eval.sampling_sft_sources import inside
from pilot_eval.sft import safe_name


def prepare(config, output_root, operation, name, volume_id, rate, hours):
    root = Path(output_root).resolve()
    config = inside(root, config); safe_name(name)
    if operation not in ('pilot-reference','pilot-kill','pilot-resume','generation-k5'):
        raise ValueError('Unknown buffered operation.')
    if not volume_id.strip(): raise ValueError('Network volume identity required.')
    budget.number(rate,True); budget.number(hours,True)
    module = generation if operation == 'generation-k5' else pilot
    _,plan,folder = module.load(config,root)
    source_plan = json.loads((root/plan['source_plan']).read_text())
    job_path = root/source_plan['run_path']/'jobs'/name/'job.plan.json'
    if operation == 'generation-k5':
        child = ['sampling-sft-buffered-generation-run','--config',str(config),'--k','5']
        base = plan['run_path']+'/k5'
        outputs = ['complete.json','results.json','attempts.jsonl','dispositions.jsonl']
    else:
        lane = 'uninterrupted' if operation == 'pilot-reference' else 'resumed'
        child = ['sampling-sft-buffered-pilot-run','--config',str(config),'--lane',lane]
        if operation == 'pilot-kill': child += ['--interrupt-after-block','2']
        base = plan['run_path']+'/'+lane
        outputs = ['blocks/001/complete.json','writer.json'] if operation == 'pilot-kill' else ['complete.json','results.json','responses.jsonl']
    request = dict(job_id=name,kind='gpu',stage='sampling',mode='runpod',
        rate_usd_per_hour=str(rate),estimated_hours=str(hours),setup_hours='0',
        network_volume_id=volume_id,command=[sys.executable,'-m','pilot_eval',*child,
        '--job',str(job_path),'--output-root',str(root)],
        outputs=[base+'/'+name for name in outputs])
    request_path = folder/'pipeline'/'requests'/(name+'.json')
    source.frozen(request_path,request)
    prepared = jobs.prepare(root/plan['source_plan'],root,request_path)
    return {'state':'prepared','operation':operation,'job':str(prepared),
            'approval':'must bind current ledger at execution', 'launched':False}


def execute(config, output_root, operation, job_path, reviewer, *, kill_job=None):
    """Run one explicitly supplied prepared job; never automatically retry."""
    from pilot_eval.sampling_sft_pipeline import create_job_approval
    import signal
    root=Path(output_root).resolve(); config=inside(root,config); job_path=inside(root,job_path)
    if operation not in ('pilot-reference','pilot-kill','pilot-resume','generation-k5'):
        raise ValueError('Unknown buffered operation.')
    module=generation if operation=='generation-k5' else pilot
    _,plan,folder=module.load(config,root)
    _,job,job_dir=jobs._load(job_path,root)
    argv=job['request']['command']
    expected='sampling-sft-buffered-generation-run' if operation=='generation-k5' else 'sampling-sft-buffered-pilot-run'
    lane='uninterrupted' if operation=='pilot-reference' else 'resumed'
    def argument(name):
        if argv.count(name)!=1 or argv.index(name)+1>=len(argv):raise ValueError('Job arguments do not match stage.')
        return argv[argv.index(name)+1]
    request=job['request']
    if (job['source_plan']!=plan['source_plan'] or request.get('mode')!='runpod'
        or request.get('kind')!='gpu' or request.get('stage')!='sampling'
        or len(argv)<4 or argv[3]!=expected
        or inside(root,argument('--config'))!=config or inside(root,argument('--job'))!=job_path
        or inside(root,argument('--output-root'))!=root or '--stop-after-blocks' in argv
        or (operation=='generation-k5' and argument('--k')!='5')
        or (operation!='generation-k5' and argument('--lane')!=lane)
        or (operation=='pilot-kill' and argument('--interrupt-after-block')!='2')
        or (operation!='pilot-kill' and '--interrupt-after-block' in argv)):
        raise ValueError('Job does not match buffered stage.')
    if (job_dir/'launch.json').exists():raise ValueError('Prior launch requires inspection and reconciliation.')
    if operation in ('pilot-kill','pilot-resume'):
        reference=pilot.verify_lane(folder/'uninterrupted',plan)
        if reference['execution_kind']!='real-cuda':raise ValueError('Real-CUDA uninterrupted pilot required.')
    if operation=='pilot-resume':
        if kill_job is None:raise ValueError('Provide controlled kill job evidence.')
        _,killed,killed_dir=jobs._load(inside(root,kill_job),root)
        command=killed['request']['command']
        if (killed['source_plan']!=plan['source_plan'] or len(command)<4 or command[3]!='sampling-sft-buffered-pilot-run'
            or command.count('--config')!=1 or inside(root,command[command.index('--config')+1])!=config
            or command.count('--lane')!=1 or command[command.index('--lane')+1]!='resumed'
            or command.count('--interrupt-after-block')!=1 or command[command.index('--interrupt-after-block')+1]!='2'
            or json.loads((killed_dir/'execution.json').read_text()).get('returncode')!=-signal.SIGKILL):
            raise ValueError('Controlled kill receipt does not match this pilot.')
        blocks=pilot.blocks(folder/'resumed',plan)
        if len(blocks)!=2:raise ValueError('Expected exactly two sealed blocks before restart.')
        marker=folder/'resumed/writer.json'
        if marker.exists():
            import platform
            owner=json.loads(marker.read_text())
            if owner.get('host')!=platform.node():raise ValueError('Writer host differs; inspect ownership manually.')
            pilot.recover_writer(config,root,'resumed',owner['token'],
                'Controlled external SIGKILL verified; releasing only a dead writer for this replay.')
    if operation=='generation-k5':generation.check_writer_replay(root,plan)
    approval=create_job_approval(root,job_path,reviewer)
    return jobs.run(job_path,root,approval)
