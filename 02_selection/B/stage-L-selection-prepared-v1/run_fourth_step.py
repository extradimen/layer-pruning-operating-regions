"""One independent model/seed Phase B selection shard on a controller CUDA lease.

BI, random and SLEB steps 1-3 are verified reuse; only SLEB step 4
requires new model calls. All three k endpoints are emitted for each method.
"""

from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import importlib
import json
import math
import os
import tarfile
import time
import uuid

import torch

from controller_identity import controller_identity
from cost_recorder import Costs, measured
from journal import Journal
from resource_probe import ResourceProbe
from selection_reuse import bi_masks, random_masks, stage_a_trajectory, validated_prefix


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, here, spec):
    for record in json.loads((here / 'snapshot.json').read_text())['files']:
        assert sha256(here / record['path']) == record['sha256']
    for record in spec['inputs']:
        assert sha256(root / record['path']) == record['sha256']
    assert spec['model']['name'] in ('qwen', 'mistral') and spec['seed'] in (11, 23, 37)
    assert spec['protocol_sha256'] == sha256(root / spec['protocol'])
    source = json.loads((root / spec['stage_a_report']).read_text())
    assert source['status'] == 'PASS_18_BASELINE_K3_SELECTION_ENDPOINTS'
    name, seed = spec['model']['name'], spec['seed']
    old = {row['method']: row for row in source['endpoints'] if row['model'] == name and row['seed'] == seed}
    assert set(old) == {'shortgpt_bi_adapted', 'sleb_iterative_adapted', 'random'}
    layers = spec['model']['layers']
    revision = spec['model']['revision']
    bi = bi_masks(old['shortgpt_bi_adapted']['BI_mean'], layers, old['shortgpt_bi_adapted']['mask'])
    random = random_masks(revision, seed, layers, old['random']['mask'])
    trajectory = stage_a_trajectory(root / spec['stage_a_archive'], name, seed)
    prefix = validated_prefix(trajectory, layers, old['sleb_iterative_adapted']['mask'])
    calibration = [json.loads(line) for line in (root / spec['calibration']).read_text().splitlines()]
    selected = [row for row in calibration if row['calibration_seed'] == seed]
    frozen_ids = json.loads((root / spec['calibration_ids']).read_text())['selections'][str(seed)]['uniform']
    if len(selected) != 32 or [row['id'] for row in selected] != frozen_ids:
        raise ValueError('calibration ID set or order mismatch')
    original = {row['id']: row for row in map(json.loads, (root / spec['original_texts']).read_text().splitlines())}
    if any(row['text'] != original[row['id']]['text'] for row in selected):
        raise ValueError('calibration text mismatch')
    controller = controller_identity(spec['project_id'])
    assert len(controller['gpu_uuids']) == 1
    output = root / spec['output']
    archive_path = root / spec['archive']
    journal_root = root / spec['journal']
    assert not output.exists() and not archive_path.exists() and not journal_root.exists()
    started = time.monotonic()
    journal = Journal(journal_root, {'controller': controller, 'seed': seed, 'model': name,
                                     'protocol_sha256': spec['protocol_sha256'], 'Final_read': False})
    costs = Costs(journal, controller['run_id'], gpu_uuids=controller['gpu_uuids'], instrument_factory=ResourceProbe)
    session = importlib.import_module(name + '_model_session')
    precision = importlib.import_module(name + '_precision_policy')
    precision.configure('BF16-strict-fp32-attention')
    report = {'status': 'RUNNING', 'controller': controller, 'model': name, 'seed': seed,
              'Final_read': False, 'stage_a_report_sha256': sha256(root / spec['stage_a_report']),
              'stage_a_archive_sha256': sha256(root / spec['stage_a_archive']),
              'endpoints': [], 'sleb_prefix_3': prefix}

    def check():
        if time.monotonic() - started > spec['soft_timeout_seconds']:
            raise TimeoutError('Phase B fourth-step soft timeout')

    try:
        check()
        model, tokenizer, identity = session.load_ready(
            root / spec['model']['model_directory'], spec['model']['model_files'], journal, costs)
        assert len(model.model.layers) == layers and session.MODEL_REVISION == revision
        barrier = root / spec['barrier']['path']
        barrier.mkdir(parents=True, exist_ok=True)
        marker = barrier / (spec['barrier']['slot'] + '.json')
        assert not marker.exists()
        ready = {'slot': spec['barrier']['slot'], 'run_id': controller['run_id'],
                 'gpu_uuid': controller['gpu_uuids'][0], 'model_identity': identity,
                 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        temporary = barrier / ('.' + spec['barrier']['slot'] + '-' + uuid.uuid4().hex)
        with temporary.open('x') as handle:
            json.dump(ready, handle, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, marker)
        temporary.unlink()
        journal.complete_work('barrier-ready', ready)
        names = spec['barrier']['slots']
        while True:
            check()
            if all((barrier / (slot + '.json')).exists() for slot in names):
                peers = {slot: json.loads((barrier / (slot + '.json')).read_text()) for slot in names}
                break
            time.sleep(1)
        assert set(peers) == set(names)
        assert all(peers[slot]['slot'] == slot for slot in names)
        assert len({peer['gpu_uuid'] for peer in peers.values()}) == len(names)
        assert len({peer['run_id'] for peer in peers.values()}) == len(names)
        report['barrier_peers'] = peers
        journal.complete_work('barrier-release', {'peers': peers})
        documents = []
        for row in selected:
            token_ids = tokenizer.encode(row['text'], add_special_tokens=False)[:512]
            if not 2 <= len(token_ids) <= 512:
                raise ValueError('invalid calibration token length')
            if name == 'mistral' and token_ids != row['tokens']:
                raise ValueError('Mistral canonical token mismatch')
            key = hashlib.sha256(row['id'].encode()).hexdigest()[:16]
            documents.append((row['id'], token_ids, key))
            journal.complete_work('tokens-' + key, {'id': row['id'], 'tokens': token_ids,
                                                     'seed': seed})
        scores = {}
        for candidate in range(1, layers - 1):
            if candidate in prefix:
                continue
            check()
            mask = sorted(prefix + [candidate])
            values = []
            with costs.phase(f'SLEB4-{seed}-{candidate}', 'prior', 'method_production') as metrics:
                with session.physical_mask(model, mask):
                    for document_id, token_ids, key in documents:
                        check()
                        value = session.document_nll(model, token_ids)
                        values.append(value['nll'])
                        journal.complete_work(f'SLEB4-{candidate}-{key}',
                                              {'id': document_id, 'seed': seed, 'deleted': mask, **value})
                metrics['forward_calls'] = measured(32, 'one forward for each calibration document', 'calls')
            score = sum(values) / 32
            if not math.isfinite(score):
                raise FloatingPointError('nonfinite fourth-step NLL')
            scores[candidate] = score
            print(json.dumps({'model': name, 'seed': seed, 'fourth_step_candidates': len(scores),
                              'expected': layers - 5}), flush=True)
        if len(scores) != layers - 5:
            raise ValueError('incomplete fourth-step candidate pool')
        fourth = min(scores, key=lambda layer: (scores[layer], layer))
        sleb = {1: sorted(prefix[:1]), 2: sorted(prefix[:2]), 4: sorted(prefix + [fourth])}
        journal.complete_work('SLEB4-decision', {'seed': seed, 'prefix_3': prefix,
                                                 'candidate_scores': scores, 'chosen': fourth})
        for method, masks in [('shortgpt_bi_adapted', bi),
                              ('sleb_iterative_adapted', sleb), ('random', random)]:
            for k in (1, 2, 4):
                mask = masks[k]
                if len(mask) != k or not all(0 < layer < layers - 1 for layer in mask):
                    raise ValueError('invalid Phase B endpoint')
                report['endpoints'].append({'model': name, 'revision': revision, 'seed': seed,
                                            'method': method, 'k': k, 'mask': mask,
                                            'selection_provenance': 'Stage A verified reuse + new SLEB step4'
                                            if method == 'sleb_iterative_adapted' else 'Stage A verified reuse'})
        assert len(report['endpoints']) == 9
        report['model_identity'] = identity
        report['sleb_fourth_choice'] = fourth
        report['sleb_fourth_scores'] = scores
        report['status'] = 'PASS_PHASE_B_9_BASELINE_ENDPOINTS_ONE_MODEL_SEED'
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with tarfile.open(archive_path, 'w:gz') as archive:
            archive.add(output, arcname=str(output.relative_to(root)))
            archive.add(journal_root, arcname=str(journal_root.relative_to(root)))
        print(json.dumps({'status': report['status'], 'archive_sha256': sha256(archive_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--shard', required=True, choices=[f'{model}-s{seed}'
                                                for model in ('qwen', 'mistral')
                                                for seed in (11, 23, 37)])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec = json.loads((here / f'spec-{args.shard}.json').read_text())
    assert args.shard == f"{spec['model']['name']}-s{spec['seed']}"
    run(Path(args.project_root).resolve(), here, spec)


if __name__ == '__main__':
    main()
