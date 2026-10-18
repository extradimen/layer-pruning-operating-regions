"""Frozen Stage B Holm-96 and complete discrete quality-cost table."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np

from contrasts import METHODS, contrast, holm, cvar90
from cost_evidence import read_cost_events

p = argparse.ArgumentParser()
p.add_argument('--project-root', required=True)
root = Path(p.parse_args().project_root)
here = Path(__file__).resolve().parent
s = json.loads((here / 'spec.json').read_text())
start = time.monotonic()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for item in json.loads((here / 'snapshot.json').read_text())['files']:
    assert sha(here / item['path']) == item['sha256']
for item in s['inputs']:
    assert sha(root / item['path']) == item['sha256']
assert s['seed'] == 20261027 and s['replicates'] == 50000

pool = json.loads((root / s['artifacts']).read_text())
paired = json.loads((root / s['paired']).read_text())
phase_a = json.loads((root / s['quality_phase_A']).read_text())
phase_b = [json.loads((root / path).read_text()) for path in s['quality_phase_B']]
assert pool['status'] == 'PASS_STAGE_B_AND_A_64_PHYSICAL_AGGREGATE'
assert paired['status'] == 'PASS_STAGE_B_ALL_K1_K2_K3_K4_PAIRED_SERVING'
assert phase_a['status'] == 'PASS_K3_ALL_24_LOGICAL_QUALITY_WITH_AUDITED_REUSE'
assert len(phase_b) == 4 and all(x['status'] == 'PASS_STAGE_B_ONE_SHARD_COMPLETE_QUALITY'
                                 for x in phase_b)

lookup = {(v['model'], v['method'], v['seed'], v['k']): v['topology_id']
          for v in pool['logical_points']}
assert len(lookup) == 96
meta = {v['topology_id']: v for v in pool['artifacts']}
assert len(meta) == 64 and len(paired['artifacts']) == 64
quality = {name: {'dense': phase_a['models'][name]['dense'],
                  'artifacts': dict(phase_a['models'][name]['artifacts'])}
           for name in ('qwen', 'mistral')}
for report in phase_b:
    for name, model in report['models'].items():
        for key, value in model['artifacts'].items():
            assert key not in quality[name]['artifacts']
            quality[name]['artifacts'][key] = value
assert sum(len(v['artifacts']) for v in quality.values()) == 64
seeds = (11, 23, 37)
tasks = ('piqa', 'hellaswag', 'gsm8k')
sizes = {'piqa': 128, 'hellaswag': 128, 'gsm8k': 32, 'c4': 64}
serving = {}
dense_baseline = {name: {} for name in ('qwen', 'mistral')}
for key, item in paired['artifacts'].items():
    serving[key] = {}
    for workload, observations in item['measurements'].items():
        dense = {x['trial']: x for x in observations if not x['warmup'] and x['model'] == 'dense'}
        candidate = {x['trial']: x for x in observations if not x['warmup'] and x['model'] == 'candidate'}
        assert set(dense) == set(candidate) == set(range(5, 25))
        for trial in dense:
            assert dense[trial]['input_ids_sha256'] == candidate[trial]['input_ids_sha256']
            assert dense[trial]['pair_order'] == candidate[trial]['pair_order']
            assert dense[trial]['generated_tokens'] == candidate[trial]['generated_tokens'] == 128
        dv = np.array([dense[t]['wall_seconds'] for t in sorted(dense)])
        cv = np.array([candidate[t]['wall_seconds'] for t in sorted(candidate)])
        ix = np.random.default_rng(s['seed']).integers(0, 20, size=(50000, 20))
        ratio = dv[ix].mean(1) / cv[ix].mean(1)
        serving[key][workload] = {
            'dense_mean_s': float(dv.mean()), 'candidate_mean_s': float(cv.mean()),
            'speedup': float(dv.mean() / cv.mean()),
            'speedup_CI_nominal95': np.quantile(ratio, [.025, .975]).tolist(),
            'peak_allocated_bytes': max(x['peak_allocated_bytes'] for x in candidate.values()),
            'pairs': 20,
        }
assert set(serving) == set(meta)
for name in ('qwen', 'mistral'):
    keys = sorted(key for key, item in paired['artifacts'].items() if item['model'] == name)
    assert len(keys) == sum(item['model'] == name for item in pool['artifacts'])
    for workload in ('B1S512T128', 'B4S2048T128'):
        dense_trials = [observation for key in keys
                        for observation in paired['artifacts'][key]['measurements'][workload]
                        if observation['model'] == 'dense' and not observation['warmup']]
        assert len(dense_trials) == len(keys) * 20
        dense_mean = float(np.mean([row['wall_seconds'] for row in dense_trials]))
        dense_baseline[name][workload] = {
            'dense_mean_s': dense_mean, 'candidate_mean_s': dense_mean,
            'speedup': 1.0, 'speedup_CI_nominal95': [1.0, 1.0],
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in dense_trials),
            'pairs': len(dense_trials), 'baseline_blocks': len(keys),
            'baseline_scope': 'pooled dense side of all contemporaneous physical pairs',
        }

contrasts = []
rows = []
for name in ('qwen', 'mistral'):
    source = quality[name]
    ids = {task: sorted(x['id'] for x in values)
           for task, values in source['dense'].items()}
    assert {task: len(values) for task, values in ids.items()} == sizes
    for key, item in source['artifacts'].items():
        for task, values in item['quality'].items():
            assert len(values) == sizes[task]
            assert sorted(x['id'] for x in values) == ids[task]
            assert len({x['id'] for x in values}) == sizes[task]
    for k in (1, 2, 3, 4):
        for method in METHODS:
            for task in tasks:
                dense_by_id = {x['id']: float(x['correct']) for x in source['dense'][task]}
                dense_vector = np.array([dense_by_id[i] for i in ids[task]])
                candidate = []
                for seed in seeds:
                    key = lookup[name, method, seed, k]
                    by_id = {x['id']: float(x['correct'])
                             for x in source['artifacts'][key]['quality'][task]}
                    candidate.append([by_id[i] for i in ids[task]])
                diff = contrast(candidate, np.tile(dense_vector, (3, 1)))
                observed = float(diff.mean())
                ix = np.random.default_rng(s['seed']).integers(0, len(diff),
                                                               size=(50000, len(diff)))
                signs = np.random.default_rng(s['seed']).choice([-1., 1.],
                                                                  size=(50000, len(diff)))
                sim = (signs * diff).mean(axis=1)
                contrasts.append({
                    'model': name, 'k': k, 'method': method, 'task': task,
                    'comparison': method + '-minus-Dense',
                    'accuracy_difference': observed,
                    'nominal95CI': np.quantile(diff[ix].mean(1), [.025, .975]).tolist(),
                    'p_two_sided': (int((np.abs(sim) >= abs(observed) - 1e-12).sum()) + 1) / 50001,
                    'question_clusters': len(diff),
                })
    dense_nll = {x['id']: x['nll'] for x in source['dense']['c4']}
    for method, seed, k in [('Dense', None, 0)] + [
        (method, seed, k) for k in (1,2,3,4) for method in METHODS for seed in seeds
    ]:
        key = None if k == 0 else lookup[name, method, seed, k]
        values = source['dense'] if key is None else source['artifacts'][key]['quality']
        gsm = values['gsm8k']
        nll = {x['id']: x['nll'] for x in values['c4']}
        delta = np.array([nll[i] - dense_nll[i] for i in ids['c4']])
        parameters = 7615616512 if name == 'qwen' else 7248023552
        row = {
            'model': name, 'method': method, 'seed': seed, 'deleted_layers': k,
            'physical_key': key,
            'parameters': parameters if key is None else meta[key]['parameters'],
            'parameter_removed_ratio': 0 if key is None else meta[key]['removed_parameters'] / parameters,
            'C4_mean_NLL': float(np.mean(list(nll.values()))),
            'C4_delta_mean': float(delta.mean()),
            'C4_positive_tail_CVaR90': cvar90(np.maximum(delta, 0)),
            'GSM_truncated': sum(bool(x['truncated']) for x in gsm),
            'GSM_parse_failures': sum(bool(x['parse_failure']) for x in gsm),
            'GSM_total_output_tokens': sum(x['output_tokens'] for x in gsm),
            'GSM_natural_wall_s_descriptive_only': sum(x['wall_seconds'] for x in gsm),
            'quality_reused': k in (0, 3),
        }
        for task in tasks:
            row[task + '_accuracy'] = sum(x['correct'] for x in values[task]) / sizes[task]
        for workload in ('B1S512T128', 'B4S2048T128'):
            cell = dict(row, workload=workload)
            cell.update(dense_baseline[name][workload] if key is None
                        else serving[key][workload])
            rows.append(cell)

assert len(contrasts) == 96 and len(rows) == 196
adjusted = holm([x['p_two_sided'] for x in contrasts])
for item, p_value in zip(contrasts, adjusted):
    item.update(p_Holm96=p_value, reject_Holm96=p_value < .05)

cost_events = []
for source in s['cost_archives']:
    cost_events.extend(read_cost_events(root / source['path'], source['sha256'], source['stage']))
lease_costs = []
for source in s['cost_receipts']:
    receipt = json.loads((root / source['path']).read_text())
    amount = receipt[source['amount_field']]
    assert isinstance(amount, (int, float)) and amount >= 0
    lease_costs.append({'source': source['path'], 'role': source['role'],
                        'GPUh': amount, 'shared_or_historical': source['shared_or_historical']})
assert len({x['source'] for x in lease_costs}) == len(lease_costs)

result = {
    'status': 'PASS_STAGE_B_COMPLETE_ALL_K_AND_METHODS',
    'contrasts': contrasts, 'long_table': rows, 'serving': serving,
    'dense_baseline': dense_baseline,
    'cost_lease_lineage': lease_costs, 'cost_event_count': len(cost_events),
    'family': {'size': 96, 'alpha': .05, 'old_families_unchanged': True,
               'not_joint_FWER_with_phase_A': True},
    'replicates': 50000, 'seed': 20261027,
    'scope': 'Previously observed E1 follow-up; fixed-three-seed mean/question-cluster inference; '
             'nominal nonsimultaneous CIs; no equivalence or universal optimal k',
    'cost_boundary': 'GPU lease occupancy includes setup, failures and retries; event wall/CPU is '
                     'descriptive and not added to lease GPUh. Shared priors counted once; '
                     'energy and price unavailable.',
    'shared_physical_not_independent': True, 'inputs': s['inputs'],
    'wall_seconds': time.monotonic() - start, 'GPUh': 0,
}
out = root / s['output']
out.mkdir(exist_ok=False)
(out / 'statistics.json').write_text(json.dumps(result, indent=2) + '\n')
fields = sorted(set().union(*(row.keys() for row in rows)))
with (out / 'all_configurations_long.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
if cost_events:
    fields = sorted(set().union(*(event.keys() for event in cost_events)))
    with (out / 'cost_events_long.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cost_events)
print(json.dumps({'status': result['status'], 'rows': len(rows),
                  'contrasts': len(contrasts), 'cost_events': len(cost_events),
                  'cost_leases': len(lease_costs)}))
