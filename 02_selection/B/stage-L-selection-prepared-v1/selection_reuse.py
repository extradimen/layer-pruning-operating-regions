"""Strictly derive Phase B prefixes from checksum-pinned Phase A selection evidence."""

import hashlib
import json
import math
import tarfile


def stage_a_trajectory(archive_path, model, seed):
    member = f'results/stage-K-selection-v1-{model}/work-SLEB-trajectory-{seed}.json'
    with tarfile.open(archive_path, 'r:gz') as archive:
        source = archive.extractfile(member)
        if source is None:
            raise ValueError('missing frozen SLEB trajectory')
        row = json.load(source)
    if row['work_id'] != f'SLEB-trajectory-{seed}' or row['payload']['seed'] != seed:
        raise ValueError('trajectory identity mismatch')
    if row['identity']['Final_read'] is not False:
        raise ValueError('trajectory used Final')
    return row['payload']['evaluations']


def validated_prefix(evaluations, layers, stage_a_mask):
    deleted = []
    for step in range(3):
        candidates = [row for row in evaluations if row['step'] == step and 'candidate' in row]
        decisions = [row for row in evaluations if row['step'] == step and 'chosen' in row]
        eligible = set(range(1, layers - 1)) - set(deleted)
        if len(decisions) != 1 or {row['candidate'] for row in candidates} != eligible:
            raise ValueError('incomplete SLEB candidate pool')
        if len(candidates) != len(eligible) or any(
            row['deleted'] != sorted(deleted + [row['candidate']]) or
            not math.isfinite(row['score']) for row in candidates
        ):
            raise ValueError('invalid SLEB score or mask')
        selected = min(candidates, key=lambda row: (row['score'], row['candidate']))['candidate']
        if decisions[0]['chosen'] != selected or decisions[0]['deleted'] != sorted(deleted + [selected]):
            raise ValueError('SLEB decision not minimum score')
        deleted.append(selected)
    if sorted(deleted) != stage_a_mask:
        raise ValueError('Stage A endpoint and trajectory disagree')
    return deleted


def bi_masks(bi_mean, layers, stage_a_mask):
    scores = {int(key): value for key, value in bi_mean.items()}
    if set(scores) != set(range(1, layers - 1)) or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError('incomplete BI means')
    ranking = sorted(scores, key=lambda layer: (scores[layer], layer))
    if sorted(ranking[:3]) != stage_a_mask:
        raise ValueError('Stage A BI endpoint mismatch')
    return {k: sorted(ranking[:k]) for k in (1, 2, 4)}


def random_masks(revision, seed, layers, stage_a_mask):
    ranking = sorted(range(1, layers - 1), key=lambda layer: hashlib.sha256(
        f'selection-supplement-v1|{revision}|{seed}|{layer}'.encode()).digest())
    if sorted(ranking[:3]) != stage_a_mask:
        raise ValueError('Stage A random endpoint mismatch')
    return {k: sorted(ranking[:k]) for k in (1, 2, 4)}
