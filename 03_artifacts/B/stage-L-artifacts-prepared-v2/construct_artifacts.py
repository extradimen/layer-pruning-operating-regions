"""One independent Stage B native-artifact shard; all masks are frozen upstream."""

from pathlib import Path
import argparse
import gc
import hashlib
import importlib
import json
import shutil
import tarfile
import time

import torch

from controller_identity import controller_identity
from cost_recorder import Costs, measured
from journal import Journal
from native_loader import load_native
from resource_probe import ResourceProbe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probes(model):
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8] * 2], device='cuda:0')
        output = model(input_ids=ids, use_cache=True)
        logits = [output.logits.detach().cpu()]
        cache = output.past_key_values
        for token in (9, 10, 11):
            output = model(input_ids=torch.tensor([[token]], device='cuda:0'),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits.append(output.logits.detach().cpu())
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--shard', required=True,
                        choices=('qwen-a', 'qwen-b', 'mistral-a', 'mistral-b'))
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    here = Path(__file__).resolve().parent
    spec = json.loads((here / f'spec-{args.shard}.json').read_text())
    assert spec['shard'] == args.shard
    start = time.monotonic()
    for item in json.loads((here / 'snapshot.json').read_text())['files']:
        assert sha(here / item['path']) == item['sha256']
    for item in spec['inputs']:
        assert sha(root / item['path']) == item['sha256']
    plan = json.loads((root / spec['point_plan']).read_text())
    assert plan['status'] == 'FROZEN_PHASE_B_72_LOGICAL_ALL_POINTS'
    assert len(plan['logical_points']) == 72 and plan['physical_count'] == 47
    groups = [item for item in plan['physical_groups'] if item['shard'] == args.shard]
    assert len(groups) == spec['expected_physical_groups']
    assert sorted(item['topology_id'] for item in groups) == sorted(plan['four_shards'][args.shard])
    partial_inventory = json.loads((root / spec['partial_inventory']).read_text())
    assert sorted(partial_inventory) == ['mistral-a', 'qwen-a', 'qwen-b']
    model_spec = spec['model']
    assert model_spec['name'] == args.shard.split('-')[0]
    controller = controller_identity(spec['project_id'])
    assert len(controller['gpu_uuids']) == 1
    output = root / spec['output']
    archive = root / spec['archive']
    journal_root = root / spec['journal']
    assert not output.exists() and not archive.exists() and not journal_root.exists()
    report = {'status': 'RUNNING', 'controller': controller, 'shard': args.shard,
              'Final_read': False, 'point_plan_sha256': sha(root / spec['point_plan']),
              'artifacts': [], 'logical_points': []}

    def check():
        if time.monotonic() - start > spec['soft_timeout_seconds']:
            raise TimeoutError('Stage B artifact shard soft timeout')

    journal = Journal(journal_root, {'controller': controller, 'shard': args.shard,
                                     'point_plan_sha256': report['point_plan_sha256'],
                                     'Final_read': False})
    costs = Costs(journal, controller['run_id'], gpu_uuids=controller['gpu_uuids'],
                  instrument_factory=ResourceProbe)
    try:
        if shutil.disk_usage(root).free < spec['initial_disk_required_bytes']:
            raise OSError('insufficient project disk before artifact shard')
        name = model_spec['name']
        session = importlib.import_module(name + '_model_session')
        importlib.import_module(name + '_precision_policy').configure('BF16-strict-fp32-attention')
        base, tokenizer, identity = session.load_ready(
            root / model_spec['model_directory'], model_spec['model_files'], journal, costs)
        layers = len(base.model.layers)
        total = sum(param.numel() for param in base.parameters())
        report['base_model_identity'] = identity
        report['base_parameters'] = total
        for group in sorted(groups, key=lambda item: item['topology_id']):
            check()
            mask = group['mask']
            assert len(mask) == len(set(mask)) and len(mask) in (1, 2, 4)
            assert mask == sorted(mask) and all(0 < index < layers - 1 for index in mask)
            assert hashlib.sha256(json.dumps({'model': name, 'mask': mask},
                                             sort_keys=True).encode()).hexdigest() == group['topology_id']
            if shutil.disk_usage(root).free < spec['disk_reserve_bytes'] + 20 * 1024**3:
                raise OSError('project disk reserve would be crossed')
            directory = root / spec['artifact_root'] / group['topology_id']
            directory.mkdir(parents=True, exist_ok=False)
            recovered = (args.shard in partial_inventory and
                         partial_inventory[args.shard]['topology_id'] == group['topology_id'])
            removed_expected = sum(param.numel() for index in mask
                                   for param in base.model.layers[index].parameters())
            with costs.phase('export-' + group['topology_id'], 'export_hash', 'method_production'):
                with session.physical_mask(base, mask):
                    expected = {key: value for key, value in base.state_dict().items()}
                    parameters = sum(param.numel() for param in base.parameters())
                    reference = probes(base)
                    if recovered:
                        original = root / spec['partial_artifact_root'] / group['topology_id']
                        inventory = partial_inventory[args.shard]
                        assert inventory['mask'] == mask and original.is_dir()
                        assert sorted(str(p.relative_to(original)) for p in original.rglob('*')
                                      if p.is_file()) == sorted(x['path'] for x in inventory['files'])
                        for member in inventory['files']:
                            source = original / member['path']
                            assert source.stat().st_size == member['bytes']
                            assert sha(source) == member['sha256']
                            destination = directory / member['path']
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            if source.suffix == '.safetensors':
                                destination.hardlink_to(source)
                            else:
                                shutil.copy2(source, destination)
                    else:
                        base.save_pretrained(directory, safe_serialization=True,
                                             max_shard_size='2GB')
                token_names = (['tokenizer.json', 'tokenizer_config.json',
                                'special_tokens_map.json', 'tokenizer.model.v3']
                               if name == 'mistral' else
                               ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'])
                for filename in token_names:
                    shutil.copyfile(root / model_spec['model_directory'] / filename,
                                    directory / filename)
                for filename in ('native_loader.py', name + '_precision_policy.py',
                                 'tokenizer_loader.py'):
                    shutil.copyfile(here / filename, directory / filename)
            with costs.phase('strict-reload-' + group['topology_id'], 'reload',
                             'research_evaluation') as metrics:
                native = load_native(directory, 'cuda:0', layers - len(mask))
                actual = native.state_dict()
                assert set(actual) == set(expected)
                for key in expected:
                    assert torch.equal(actual[key], expected[key]), key
                assert sum(param.numel() for param in native.parameters()) == parameters
                assert total - parameters == removed_expected
                result = probes(native)
                errors = [float((left.float() - right.float()).abs().max())
                          for left, right in zip(reference, result)]
                assert errors == [0.0] * 4
                metrics['forward_calls'] = measured(4, 'native prefill and three cached decode', 'calls')
                del actual, native, expected, reference, result
                gc.collect()
                torch.cuda.empty_cache()
            members = [{'path': str(path.relative_to(directory)), 'sha256': sha(path),
                        'bytes': path.stat().st_size}
                       for path in sorted(directory.rglob('*')) if path.is_file()]
            bundle = hashlib.sha256(json.dumps(members, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
            artifact = {'model': name, 'topology_id': group['topology_id'],
                        'mask': mask, 'path': str(directory.relative_to(root)),
                        'bundle_SHA256': bundle, 'members': members,
                        'native_bytes': sum(item['bytes'] for item in members),
                        'parameters': parameters, 'removed_parameters': total - parameters,
                        'retained_tensor_exact': True, 'strict_reload': True,
                        'prefill_cached_max_abs_errors': errors,
                        'logical_consumers': group['logical_consumers']}
            artifact['recovered_from_failed_v1'] = recovered
            if recovered:
                artifact['partial_inventory_sha256'] = sha(root / spec['partial_inventory'])
            (directory / 'bundle_manifest.json').write_text(json.dumps(artifact, indent=2) + '\n')
            journal.complete_work('artifact-' + group['topology_id'], artifact)
            report['artifacts'].append(artifact)
            report['logical_points'].extend(dict(point, bundle_SHA256=bundle,
                                                 path=str(directory.relative_to(root)))
                                            for point in plan['logical_points']
                                            if point['topology_id'] == group['topology_id'])
            print(json.dumps({'shard': args.shard,
                              'artifacts_done': len(report['artifacts']),
                              'expected': len(groups)}), flush=True)
        assert len(report['artifacts']) == spec['expected_physical_groups']
        report['status'] = 'PASS_STAGE_B_ONE_SHARD_STRICT_NATIVE_ARTIFACTS'
        del base, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        with tarfile.open(archive, 'w:gz') as bundle:
            bundle.add(output, arcname=str(output.relative_to(root)))
            if journal_root.exists():
                bundle.add(journal_root, arcname=str(journal_root.relative_to(root)))


if __name__ == '__main__':
    main()
