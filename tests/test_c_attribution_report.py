"""Ticket 13: the C-attribution report, figures and catalog registration, on a real CPU chain run.

The chain runs once on the tiny real-format world (``c_attribution_realpath_fixture``); every report number is
checked against the sealed artifact it comes from, not recomputed by the test's own logic where avoidable."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

import c_attribution_realpath_fixture as world
import test_c_attribution_realpath as rp
from pilot_eval import c_attribution as ca
from pilot_eval import c_attribution_generation as cg
from pilot_eval import c_attribution_report as cr
from pilot_eval import c_attribution_stages as cs
from pilot_eval import catalog as cat

ARMS = ('online-grpo', 'offline-grpo')


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('report')
    mp = pytest.MonkeyPatch()
    mp.setenv('HF_HOME', str(tmp / 'hf-home'))
    mp.setenv('HF_HUB_OFFLINE', '1')
    w = world.build(tmp / 'world')
    c, _ = rp.chain(w, tmp)
    summary = c.run()
    mp.undo()
    assert summary['state'] == 'complete', (summary['failures'], summary.get('stop_reason'))
    local = tmp / 'local'
    return dict(w=w, local=local, sub=local / 'c-attribution', tmp=tmp,
                report=json.loads((local / cs.REPORT_DIR / 'report.json').read_text()),
                markdown=(local / cs.REPORT_DIR / 'report.md').read_text())


def _j(path):
    return json.loads(Path(path).read_text())


def test_the_header_carries_how_to_read_the_sign_convention_and_the_labels(run):
    header = run['report']['header']
    assert len(header['how_to_read']) == 5 and any('greedy' in s for s in header['how_to_read'])
    assert [r['quantity'] for r in header['sign_convention']] == [r['quantity'] for r in ca.SIGN_CONVENTION]
    assert header['labels'] == list(cg.LABELS)
    assert run['report']['schema'] == 'c-attribution-report-v2'


def test_the_attribution_summary_traces_to_the_sealed_aggregate(run):
    sub = run['sub']
    for arm in ARMS:
        a = run['report']['attribution'][arm]
        counts = _j(sub / 'rows' / arm / 'manifest.json')['counts']
        assert a['J_bar'] == counts['mean_reward'] and a['informative_questions'] == counts['groups']['mixed']
        sealed = _j(sub / 'aggregate' / arm / 'adapters.json')['adapters']
        rows = {r['index']: r for r in a['adapters']['rows']}
        assert len(rows) == len(sealed['dJdlambda'])
        boot = _j(sub / 'aggregate/comparison/comparison.json')['adapter_bootstrap']['arms'][arm]
        for i, d in enumerate(sealed['dJdlambda']):
            assert rows[i]['dJdlambda'] == d and rows[i]['removal_J'] == sealed['removal_J'][i]
            assert rows[i]['p2.5'] == boot['dJdlambda']['p2.5'][i] and rows[i]['rank_p97.5'] == boot['rank']['p97.5'][i]
            assert [rows[i]['layer'], rows[i]['projection']] == list(sealed['names'][i])
        assert [r['dJdlambda'] for r in a['adapters']['rows']] == sorted(sealed['dJdlambda'], reverse=True)
        totals = sealed['totals']
        assert a['adapters']['totals'] == totals
        top10 = sorted(sealed['dJdlambda'], reverse=True)[:10]
        assert a['adapters']['top10_share_of_gross'] == pytest.approx(sum(abs(x) for x in top10) / totals['gross'])
        with np.load(sub / 'aggregate' / arm / 'neurons.npz') as npz:
            d = npz['dJdlambda']
        per_layer = _j(sub / 'aggregate' / arm / 'bootstrap.json')['neurons']['per_layer']
        for li, row in enumerate(a['reads_per_layer']):
            assert row['dJdlambda_sum'] == pytest.approx(float(d[li].sum()))
            assert row['gross'] == pytest.approx(float(np.abs(d[li]).sum()))
            assert row['p2.5'] == per_layer['p2.5'][li]
            assert row['quantiles']['p50'] == pytest.approx(float(np.percentile(d[li], 50)))
        identity = _j(sub / 'aggregate' / arm / 'identity.json')['layers']
        assert a['identity']['max_relative_to_gross'] == max(v['relative_to_gross'] for v in identity.values())
        assert a['harmful_adapters_reported_only'] == _j(sub / 'freeze' / arm / f'{arm}.json')['harmful_adapters_reported_only']


def test_stability_qualification_and_provenance_sections(run):
    sub, report = run['sub'], run['report']
    for arm in ARMS:
        assert report['stability'][arm] == _j(sub / 'aggregate' / arm / 'split.json')['stability']
    qual = _j(sub / 'qualification/qualification.json')
    q = report['qualification']
    assert q['checks'] == qual['checks'] and q['signoff']['decision'] == 'accept'
    assert q['numerics'] == _j(sub / 'qualification/numerics.json')
    for arm in ARMS:
        d, s = qual['details'][arm], q['arms'][arm]
        assert s['replay']['status'] == 'not-run'
        assert s['finite_difference']['median_rel_err'] == pytest.approx(
            float(np.median([i['rel_err'] for i in d['finite_difference']['items'] if i['informative']])))
        assert s['finite_difference']['floors'] == d['finite_difference']['floors']
        assert s['cached_vs_forced']['per_token_abs'] == d['cached_vs_forced']['per_token_abs']
        assert s['batch_invariance']['max_fraction'] == d['batch_invariance']['max_fraction']
        assert s['noop']['passed'] is True and s['processors']['passed'] is True
        assert s['memory']['fraction'] == pytest.approx(d['memory']['peak_bytes'] / d['memory']['total_bytes'])
        prov = report['provenance'][arm]
        manifest = _j(sub / 'rows' / arm / 'manifest.json')['provenance']
        assert prov['recorded'] == sorted(manifest['recorded']) and prov['inferred'] == sorted(manifest['inferred'])
        assert 'logits processors resolved (check f)' in prov['verified']


def test_the_selection_section_matches_the_freeze(run):
    for arm in ARMS:
        freeze = _j(run['sub'] / 'freeze' / arm / f'{arm}.json')
        for name, group in freeze['groups'].items():
            s = run['report']['selection'][arm][name]
            assert s['size'] == len(group['targets']) and s['shortfall'] == group['shortfall']
            assert s['prediction'] == group['prediction']
            control = group['control']
            assert s['control']['available'] == control['available']
            assert s['control']['match_ratio'] == control.get('match_ratio')
            assert s['control']['deficient_targets'] == len(control.get('deficient_targets') or [])
            assert s['distribution'] == group.get('layer_distribution', group.get('module_distribution'))
        assert run['report']['selection'][arm]['hot64-reads']['hot_layers'] == freeze['hot_layers']


def test_the_comparison_reports_cross_arm_read_overlaps_and_the_caveat(run):
    comp = run['report']['comparison']
    f = {arm: _j(run['sub'] / 'freeze' / arm / f'{arm}.json')['groups'] for arm in ARMS}
    glob = [set(f[a]['global64-reads']['targets']) for a in ARMS]
    assert comp['reads']['global_top64_overlap'] == len(glob[0] & glob[1])
    assert comp['adapters']['top10_overlap'] == len(set(f[ARMS[0]]['top10-adapters']['targets']) &
                                                     set(f[ARMS[1]]['top10-adapters']['targets']))
    assert set(comp['reads']['per_layer_top64_overlap']) == {str(l) for l in range(28)}
    assert comp['distributions']['online-grpo']['top10-adapters'] == f['online-grpo']['top10-adapters']['module_distribution']
    assert 'one seed per arm' in comp['caveat']
    assert len(comp['adapters']['paired_rows']) == 196


def test_the_ablation_table_keeps_every_condition_with_the_untuned_row(run):
    for arm in ARMS:
        conds = run['report']['arms'][arm]['conditions']
        assert {'intact', 'top10-adapters', 'global64-reads', 'untuned'} <= set(conds)


def test_the_markdown_puts_the_labels_on_every_results_table(run):
    md = run['markdown']
    assert md.startswith('# REHEARSAL')
    for heading in ('## How to read the results', '## Sign convention', '## Attribution', '## Stability',
                    '## Qualification and numerics', '## Selection', '## Ablation', '## Selected reads and matched controls',
                    '## Online against offline'):
        assert heading in md, heading
    sections = md.split('\n## ')
    for section in sections:
        title = section.splitlines()[0]
        if '|---' in section and title not in ('Sign convention',):
            assert cr.LABEL_LINE in section, title


def test_figures_are_built_from_report_json_only(run, tmp_path):
    report_path = run['local'] / cs.REPORT_DIR / 'report.json'
    copy = tmp_path / 'report.json'
    shutil.copy(report_path, copy)                 # nothing else from the run is reachable
    index = cr.figures(copy, tmp_path / 'figures')
    names = {f['file'] for f in index['figures']}
    assert names == {'adapters-heatmap-online-grpo.png', 'adapters-heatmap-offline-grpo.png',
                     'adapters-online-vs-offline.png', 'reads-per-layer.png', 'ablation-online-grpo.png',
                     'ablation-offline-grpo.png'}
    for f in index['figures']:
        assert (tmp_path / 'figures' / f['file']).stat().st_size > 1000 and f['sources']
    assert _j(tmp_path / 'figures' / 'figures.json') == index
    assert json.loads(copy.read_text())['arm_order'] == list(ARMS)   # the scatter's x axis is the record's first arm


def test_registration_writes_catalog_entries_that_verify(run, tmp_path):
    catalog_dir = tmp_path / 'catalog'
    shutil.copytree(run['w']['code'] / 'catalog', catalog_dir)
    arms = yaml.safe_load((catalog_dir / 'arms.yaml').read_text())   # the real catalog has an untuned arm
    real = yaml.safe_load((cr.REPO / 'catalog/arms.yaml').read_text())['arms']['untuned']
    arms['arms']['untuned'] = {k: v for k, v in real.items() if k not in ('checkpoints', 'training_data')}
    (catalog_dir / 'arms.yaml').write_text(yaml.safe_dump(arms))
    before = yaml.safe_load((catalog_dir / 'cohorts.yaml').read_text())
    written = cr.register(run['local'], catalog_dir, location='local', label='rehearsal-pod-1')
    run_ids = [w for w in written['responses']]
    # one response set per arm and condition that ran, plus untuned once
    assert 'online-grpo.s64.greedy.test150.c-attribution-rehearsal-pod-1-intact' in run_ids
    assert 'untuned.greedy.test150.c-attribution-rehearsal-pod-1' in run_ids
    for rid in run_ids:
        entry = yaml.safe_load((catalog_dir / 'responses' / f'{rid}.yaml').read_text())
        assert entry['c_suitability'] == 'greedy-only' and entry['kind'] == 'evaluation'
        assert entry['policy']['mode'] == 'greedy'
    after = yaml.safe_load((catalog_dir / 'cohorts.yaml').read_text())
    uses = after['cohorts']['gsm8k-test-cohort150']['prior_uses']
    new = uses[len(before['cohorts']['gsm8k-test-cohort150']['prior_uses']):]
    assert any('freeze' in u and 'rehearsal-pod-1' in u for u in new)
    assert any('gsm8k:test:817' in u for u in new)              # the exact rehearsal IDs
    cohort_file = run['local'] / world.COHORT          # the cohort source, so question IDs are checked too
    cohort_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(run['w']['artifacts'] / world.COHORT, cohort_file)
    result = cat.verify(cat.load(catalog_dir), roots={'local': str(run['local'])})
    assert any(c['entry'] == 'cohort:gsm8k-test-cohort150' and c['status'] == 'pass' for c in result['checks'])
    mine = [c for c in result['checks'] if c['entry'].split(':', 1)[1] in run_ids]
    assert mine and all(c['status'] == 'pass' for c in mine), [c for c in mine if c['status'] != 'pass'][:3]
    with pytest.raises(cr.ReportError, match='already registered'):
        cr.register(run['local'], catalog_dir, location='local', label='rehearsal-pod-1')


def test_the_cached_vs_forced_row_table_is_in_the_report_and_its_markdown(run):
    qual = _j(run['sub'] / 'qualification/qualification.json')['details']
    for arm in ARMS:
        table = run['report']['qualification']['arms'][arm]['cached_vs_forced']['per_row']
        assert table == qual[arm]['cached_vs_forced']['per_row'] and len(table) == len(qual[arm]['rows'])
        assert run['report']['qualification']['numerics']['per_arm'][arm]['cached_vs_forced']['per_row'] == table
    section = run['markdown'].split('## Qualification and numerics')[1].split('\n## ')[0]
    assert section.count('cached vs forced by row') == len(ARMS)
    first = run['report']['qualification']['arms'][ARMS[0]]['cached_vs_forced']['per_row'][0]
    assert first['id'] in section
