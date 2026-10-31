"""Read-only analysis of saved pilot lanes; never certifies exact replay."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(folder):
    folder = Path(folder)
    issues, differences, source_hashes = [], [], {}
    lanes = {}
    try:
        plan_hash = json.loads((folder/'plan-seal.json').read_text())['config_sha256']
        replay = json.loads((folder/'replay.json').read_text())
        source_hashes['plan-seal.json'] = digest(folder/'plan-seal.json')
        source_hashes['replay.json'] = digest(folder/'replay.json')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append({'issue':'missing or invalid plan/replay seal', 'detail':str(exc)})
        plan_hash, replay = None, {}
    for lane in ('uninterrupted', 'resumed'):
        base = folder / lane
        requests, decisions, blocks, records = defaultdict(list), defaultdict(list), set(), {}
        previous = None
        for directory in sorted((base / 'blocks').glob('*')):
            if not directory.is_dir() or not directory.name.isdigit():
                issues.append({'lane': lane, 'path': str(directory), 'issue': 'unexpected block entry'})
                continue
            block = int(directory.name)
            blocks.add(block)
            seal_path = directory / 'complete.json'
            if seal_path.exists():
                try:
                    source_hashes[str(seal_path.relative_to(folder))] = digest(seal_path)
                    seal = json.loads(seal_path.read_text())
                    if seal['block'] != block or seal['previous'] != previous or seal['config_sha256'] != plan_hash:
                        issues.append({'lane': lane, 'block': block+1, 'issue': 'broken seal chain'})
                    for name, expected in seal['files'].items():
                        file = directory / name
                        if Path(name).name != name or not file.is_file() or digest(file) != expected:
                            issues.append({'lane': lane, 'block': block+1, 'file': name, 'issue': 'sealed file missing or hash mismatch'})
                    env = base / ('env.' + seal['attempt'] + '.json')
                    if not env.is_file() or digest(env) != seal['env_sha256']:
                        issues.append({'lane': lane, 'block': block+1, 'issue': 'environment hash mismatch'})
                    previous = digest(seal_path)
                except (KeyError, ValueError, OSError) as exc:
                    issues.append({'lane': lane, 'block': block+1, 'issue': 'invalid seal', 'detail': str(exc)})
            else:
                issues.append({'lane': lane, 'block': block+1, 'issue': 'missing seal'})
            path = directory / 'scientific.jsonl'
            if path.is_file(): source_hashes[str(path.relative_to(folder))] = digest(path)
            try:
                records[block] = [json.loads(line) for line in path.read_text().splitlines()]
                for line_no, row in enumerate(records[block], 1):
                    if 'decision' in row:
                        d = row['decision']; key = (block, d['problem_id'], d['step'])
                        decisions[key].append(d)
                    else:
                        reqs, outs = row['requests'], row['outputs']
                        if len(reqs) != len(outs):
                            issues.append({'lane': lane, 'block': block+1, 'line': line_no, 'issue': 'request/output cardinality'})
                        for index, req in enumerate(reqs):
                            key = (block, row['kind'], req['problem_id'], req['step'])
                            requests[key].append((req, outs[index] if index < len(outs) else None, line_no, index))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issues.append({'lane': lane, 'block': block+1, 'issue': 'invalid scientific records', 'detail': str(exc)})
        try:
            result_path = base / 'results.json'
            source_hashes[str(result_path.relative_to(folder))] = digest(result_path)
            final_seal_path = base / 'complete.json'
            if not final_seal_path.is_file():
                issues.append({'lane': lane, 'issue': 'missing final seal'})
            else:
                source_hashes[str(final_seal_path.relative_to(folder))] = digest(final_seal_path)
                final_seal = json.loads(final_seal_path.read_text())
                parent_hashes = [v for k, v in replay.get('parents', {}).items() if k.endswith('/'+lane+'/complete.json')]
                if (final_seal.get('last_block') != previous or final_seal.get('config_sha256') != plan_hash
                    or final_seal.get('files', {}).get('results.json') != digest(result_path)
                    or parent_hashes != [digest(final_seal_path)]):
                    issues.append({'lane': lane, 'issue': 'final results seal mismatch'})
            traces = json.loads(result_path.read_text())['traces']
            trace_map = defaultdict(list)
            for trace in traces: trace_map[trace['problem_id']].append(trace)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({'lane': lane, 'issue': 'invalid final results', 'detail': str(exc)})
            trace_map = {}
        if blocks != set(range(len(blocks))): issues.append({'lane': lane, 'issue': 'noncontiguous blocks'})
        lanes[lane] = (requests, decisions, trace_map, blocks, records)
    a, b = lanes['uninterrupted'], lanes['resumed']
    summaries = defaultdict(Counter)
    score_count, maximum, first = 0, 0.0, None
    unmatched = 0
    for key in sorted(set(a[0]) | set(b[0])):
        left, right = a[0].get(key, []), b[0].get(key, [])
        block, kind, pid, step = key
        context = {'block': block+1, 'problem_id': pid, 'kind': kind, 'step': step}
        if len(left) != 1 or len(right) != 1 or left[0][0] != right[0][0]:
            unmatched += len(left)+len(right)
            issues.append({**context, 'issue': 'missing, duplicate, or different request', 'left_count': len(left), 'right_count': len(right),
                           'left_requests':[v[0] for v in left], 'right_requests':[v[0] for v in right]})
            continue
        req, x, line, index = left[0]; y = right[0][1]
        context.update(record_line=line, request_index=index, request=req)
        if not isinstance(x, dict) or not isinstance(y, dict):
            issues.append({**context, 'issue': 'missing output'}); continue
        if kind == 'score':
            if (x.get('token_ids') != y.get('token_ids') or x.get('positions') != y.get('positions')
                or not isinstance(x.get('logprobs'), list) or not isinstance(y.get('logprobs'), list)
                or len(x['logprobs']) != len(y['logprobs'])
                or len(x['logprobs']) != len(x.get('token_ids', []))
                or len(x['logprobs']) != len(x.get('positions', []))
                or x.get('token_ids') != req.get('response_ids')):
                issues.append({**context, 'issue': 'score token/position/length mismatch'}); continue
            for i, (v, w) in enumerate(zip(x['logprobs'], y['logprobs'])):
                if type(v) not in (int, float) or type(w) not in (int, float) or not math.isfinite(v) or not math.isfinite(w):
                    issues.append({**context, 'issue': 'nonnumeric score', 'token_index': i}); continue
                if v != w:
                    diff = {**context, 'category': 'score', 'token_index': i, 'token_id': x['token_ids'][i],
                            'position': x['positions'][i], 'left': v, 'right': w, 'abs_difference': abs(v-w)}
                    differences.append(diff); score_count += 1; summaries[block+1]['score_values'] += 1
                    maximum = max(maximum, abs(v-w))
                    if first is None or (diff['block'], diff['record_line'], diff['request_index'], i) < (first['block'], first['record_line'], first['request_index'], first['token_index']): first = diff
            other_x = {k:v for k,v in x.items() if k != 'logprobs'}
            other_y = {k:v for k,v in y.items() if k != 'logprobs'}
            if other_x != other_y: differences.append({**context, 'category':'score_other_output', 'left':other_x, 'right':other_y})
        elif kind == 'generate':
            if x != y:
                summaries[block+1]['generation_outputs'] += 1
                if x.get('token_ids') != y.get('token_ids'): summaries[block+1]['generation_token_outputs'] += 1
                differences.append({**context, 'category':'generation_output', 'left':x, 'right':y})
        else: issues.append({**context, 'issue': 'unknown request kind'})
    accepted = proposals = 0
    for key in sorted(set(a[1]) | set(b[1])):
        left, right = a[1].get(key, []), b[1].get(key, [])
        block, pid, step = key
        context = {'block':block+1, 'problem_id':pid, 'step':step}
        if len(left) != 1 or len(right) != 1:
            issues.append({**context, 'issue':'missing or duplicate decision', 'left_count':len(left), 'right_count':len(right)}); continue
        x, y = left[0], right[0]
        if x != y:
            summaries[block+1]['decisions'] += 1
            if x.get('accepted') != y.get('accepted'): accepted += 1; summaries[block+1]['accepted'] += 1
            if any(x.get(k) != y.get(k) for k in ('cut','old_ids','proposal_ids')): proposals += 1; summaries[block+1]['proposals'] += 1
            differences.append({**context, 'category':'decision', 'left':x, 'right':y})
    final_count = 0
    final_fields = Counter()
    for pid in sorted(set(a[2]) | set(b[2])):
        x, y = a[2].get(pid, []), b[2].get(pid, [])
        if len(x) != 1 or len(y) != 1:
            issues.append({'problem_id':pid, 'issue':'missing or duplicate final trace'}); continue
        if x[0] != y[0]:
            final_count += 1
            for field in ('response_ids', 'text', 'end_state', 'automatic_flexible_v3', 'automatic_strict'):
                if x[0].get(field) != y[0].get(field): final_fields[field] += 1
            differences.append({'category':'final_trace', 'problem_id':pid, 'left':x[0], 'right':y[0]})
    if a[3] != b[3]: issues.append({'issue':'different block sets', 'left':sorted(a[3]), 'right':sorted(b[3])})
    return {'scope':'descriptive saved-artifact analysis only; exact replay gate unchanged',
            'scores':{'different_values':score_count, 'max_abs_difference':maximum, 'first_difference':first},
            'decisions':{'accepted_differences':accepted, 'proposal_differences':proposals},
            'final_traces':{'different_problems':final_count, 'field_differences':dict(final_fields)},
            'blocks':{str(n):{'phase':'before_interruption' if n<=2 else 'after_interruption',
                          'ordered_records_equal':a[4].get(n-1) == b[4].get(n-1) if n-1 in a[4] and n-1 in b[4] else None,
                          'left_records':len(a[4][n-1]) if n-1 in a[4] else None,
                          'right_records':len(b[4][n-1]) if n-1 in b[4] else None, **summaries[n]}
                      for n in sorted({v+1 for v in a[3] | b[3]})},
            'structure':{'unmatched_requests':unmatched, 'issues':issues, 'integrity_valid':not any(
                any(term in issue['issue'] for term in ('seal', 'hash', 'noncontiguous', 'invalid scientific', 'invalid final')) for issue in issues)},
            'source_hashes':source_hashes, 'differences':differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', required=True, type=Path)
    parser.add_argument('--name', required=True, help='New diagnostic directory name')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.','..'):
        parser.error('name must be one directory component')
    folder = args.pilot_dir.resolve()
    if not (folder/'uninterrupted').is_dir() or not (folder/'resumed').is_dir():
        parser.error('pilot directory must contain both lanes')
    result = diagnose(folder)
    destination = folder/'diagnostics'/args.name
    destination.mkdir(parents=True, exist_ok=False)
    differences = result.pop('differences')
    (destination/'differences.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in differences))
    result['provenance'] = {'pilot_dir':str(folder), 'created_utc':datetime.now(timezone.utc).isoformat(),
                            'diagnostic_sha256':digest(Path(__file__)),
                            'replay_sha256':digest(folder/'replay.json') if (folder/'replay.json').is_file() else None}
    (destination/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(destination/'summary.json')


if __name__ == '__main__': main()
