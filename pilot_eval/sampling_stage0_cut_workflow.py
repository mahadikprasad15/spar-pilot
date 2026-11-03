"""Standalone principal-cut build, paired singleton evaluation and private export."""
import json
from pathlib import Path


def export(directory, root, *, repo, client=None):
    """Upload only a verified complete run and verify the exact remote archive bytes."""
    import hashlib
    import tarfile
    import tempfile
    from pilot_eval.sampling_sft_hf import Hub, private
    from pilot_eval.training import file_hash
    from pilot_eval.run import _write_json
    directory=Path(directory).resolve(); root=Path(root).resolve()
    if not directory.is_relative_to(root): raise ValueError('Run outside artifact root.')
    if not (directory/'complete.json').is_file(): raise ValueError('Cannot export incomplete run.')
    verify(directory)
    client=client or Hub();private(client,repo)
    files=sorted(p for p in directory.rglob('*') if p.is_file() and not p.is_symlink()
                 and p.name != 'hf-upload.json' and not p.name.endswith('.tmp'))
    manifest={str(p.relative_to(root)):file_hash(p) for p in files}
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    receipt=directory/'meta/hf-upload.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['manifest_sha256']!=identity or saved['repo']!=repo: raise ValueError('Export identity changed.')
        if file_hash(client.fetch(repo,saved['archive'],saved['revision']))!=saved['archive_sha256']:
            raise ValueError('Remote archive readback changed.')
        return saved
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        archive=Path(temporary)/'bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            for path in files: stream.add(path,arcname=str(path.relative_to(root)),recursive=False)
        if any(file_hash(p)!=manifest[str(p.relative_to(root))] for p in files):
            raise ValueError('Local artifact changed during export.')
        index=Path(temporary)/'index.json';index.write_text(json.dumps(dict(files=manifest,manifest_sha256=identity),sort_keys=True))
        remote=f'exports/cuts-{identity}'
        archive_name=remote+'/bundle.tar.gz'; index_name=remote+'/index.json'
        archive_hash=file_hash(archive); index_hash=file_hash(index)
        revision=client.publish(repo,{archive_name:archive,index_name:index})
        if not isinstance(revision,str) or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('HF export lacks exact commit revision.')
        if file_hash(client.fetch(repo,archive_name,revision))!=archive_hash or file_hash(client.fetch(repo,index_name,revision))!=index_hash:
            raise ValueError('HF readback verification failed.')
        result=dict(repo=repo,revision=revision,archive=archive_name,archive_sha256=archive_hash,
                    index=index_name,index_sha256=index_hash,manifest_sha256=identity)
        _write_json(receipt,result)
        return result


def verify(directory):
    """Verify complete aggregate and every response unit without loading a model."""
    import hashlib
    from pilot_eval.training import file_hash
    from pilot_eval.workflow import _hash
    from pilot_eval.stage0_evaluation_records import OUTPUT_FIELDS, records
    directory=Path(directory)
    if not (directory/'complete.json').is_file(): raise ValueError('Run is not complete.')
    config=json.loads((directory/'config.json').read_text())
    items=json.loads((directory/'inputs/items.json').read_text())
    inputs=json.loads((directory/'inputs/tokenized-prompts.json').read_text())
    if [r['id'] for r in inputs] != [r['id'] for r in items]: raise ValueError('Input identity changed.')
    if any(t.get('prompt_sha256')!=_hash(i['prompt']) or t.get('attention_mask')!=[1]*len(t['input_ids']) for t,i in zip(inputs,items)):
        raise ValueError('Prompt identity changed.')
    rh=file_hash(directory/'meta/runtime.json'); ih=file_hash(directory/'inputs/tokenized-prompts.json'); items_hash=file_hash(directory/'inputs/items.json'); rows=[]
    for variant in config['variant_order']:
        for index,item in enumerate(items):
            key=hashlib.sha256((variant+'\0'+str(item['id'])).encode()).hexdigest()
            unit=dict(key=key,variant=variant,ids=[item['id']],seed=config['seed']+index//16)
            folder=directory/'units'/key; payload=folder/'responses.jsonl'; proof=folder/'integrity.json'
            expected=dict(config_sha256=_hash(config),runtime_sha256=rh,input_sha256=ih,items_sha256=items_hash,
                          responses_sha256=file_hash(payload),integrity_sha256=file_hash(proof))
            if json.loads((folder/'complete.json').read_text())!=expected: raise ValueError('Response unit changed.')
            evidence=json.loads(proof.read_text())
            if any(not all(evidence[s].get(k) is True for k in ('base_unchanged','adapters_unchanged')) for s in ('before','after')):
                raise ValueError('Response integrity failed.')
            saved=[json.loads(line) for line in payload.read_text().splitlines()]
            if saved!=records(config,unit,[item],[{k:r[k] for k in OUTPUT_FIELDS} for r in saved],runtime_sha256=rh,input_sha256=ih):
                raise ValueError('Response scoring changed.')
            rows.extend(saved)
    response=directory/'results/responses.jsonl';summary=directory/'results/results.json'
    expected=dict(config_sha256=_hash(config),items_sha256=items_hash,responses_sha256=file_hash(response),results_sha256=file_hash(summary))
    if json.loads((directory/'complete.json').read_text())!=expected: raise ValueError('Completed result seal changed.')
    if [json.loads(x) for x in response.read_text().splitlines()]!=rows: raise ValueError('Aggregate changed.')
    return json.loads(summary.read_text())
