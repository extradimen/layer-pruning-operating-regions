"""Phase B governed CPU search from previously verified model-own priors.

The launcher pins spec.json, every input and this source in an immutable snapshot.
This worker never reads an evaluation set or any prior Final report.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time

from recipe_solver import solve_tabu
from selection_depth import tabu_starts


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qwen_prior(report, seed):
    assert report['status'] == 'PASS_STAGE_B6_CALIBRATION_PRIORS_FOR12_OBJECTIVE_CONFIGS_NO_FINAL'
    item = report['priors'][f'uniform-s{seed}']
    assert len(item['doc_ids']) == 32
    return item['a_by_objective']['NLL'], item['H'], item['doc_ids']


def mistral_prior(report, seed):
    assert report['status'] == 'PASS_96_DOCUMENTS_3_SEEDS_INDEPENDENT_BLOCK_BRANCH_PRIORS'
    assert report['documents_completed'] == 96 and report['actual_calls'] == 18816
    item = report['priors'][str(seed)]['block']
    assert item['documents'] == 32
    return item['a'], item['H'], item.get('doc_ids')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    root = Path(parser.parse_args().project_root).resolve()
    here = Path(__file__).resolve().parent
    spec = json.loads((here / 'spec.json').read_text())
    for record in json.loads((here / 'snapshot.json').read_text())['files']:
        assert sha256(here / record['path']) == record['sha256']
    for record in spec['inputs']:
        assert sha256(root / record['path']) == record['sha256']
    assert spec['k'] == [1, 2, 4]
    assert spec['seeds'] == [11, 23, 37]
    assert spec['evaluations_per_endpoint'] == 100000
    assert spec['protocol_sha256'] == sha256(root / spec['protocol'])
    calibration = json.loads((root / spec['calibration_ids']).read_text())['selections']
    output = root / spec['output']
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records = []
    for model in spec['models']:
        name, revision, layers = model['name'], model['revision'], model['layers']
        assert (name, layers) in (('qwen', 28), ('mistral', 32))
        source = root / model['prior']
        report = json.loads(source.read_text())
        getter = qwen_prior if name == 'qwen' else mistral_prior
        for seed in spec['seeds']:
            a, h, document_ids = getter(report, seed)
            frozen_ids = calibration[str(seed)]['uniform']
            assert len(frozen_ids) == 32
            if document_ids is not None:
                assert document_ids == frozen_ids
            document_ids = frozen_ids
            assert len(a) == len(h) == layers and all(len(row) == layers for row in h)
            for k in spec['k']:
                starts = tabu_starts(revision, seed, layers, k)
                solved = solve_tabu(a, h, k, starts,
                                    protected=(0, layers - 1),
                                    max_evaluations=100000,
                                    max_cpu_seconds=3600)
                assert solved['evaluations'] == 100000
                assert [row['evaluations'] for row in solved['restart_records']] == [12500] * 8
                assert solved['tabu_tenure'] == 5
                mask = solved['top_two'][0]['mask']
                assert len(mask) == k and all(0 < i < layers - 1 for i in mask)
                records.append({
                    'model': name, 'revision': revision, 'seed': seed, 'k': k,
                    'method': 'interaction_tabu', 'mask': mask,
                    'solver': solved, 'q': 0, 'R': 0, 'Final_read': False,
                    'prior_path': model['prior'], 'prior_SHA256': sha256(source),
                    'calibration_IDs': document_ids,
                })
                print(json.dumps({'done': len(records), 'model': name, 'seed': seed, 'k': k,
                                  'evaluations': solved['evaluations']}), flush=True)
    assert len(records) == 18
    output.write_text(json.dumps({
        'status': 'PASS_PHASE_B_FIXED_18_TABU_MASKS', 'Final_read': False,
        'records': records, 'logical_endpoints': 18, 'wall_seconds': time.monotonic() - start,
        'inputs': spec['inputs'], 'protocol_sha256': spec['protocol_sha256'],
    }, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
