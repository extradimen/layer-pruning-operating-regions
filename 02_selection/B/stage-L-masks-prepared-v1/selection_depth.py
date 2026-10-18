"""Frozen Phase B starting masks; scientific scores are never read here."""

import hashlib


def depth_strata(layers, count):
    if count <= 0 or len(layers) < count or layers != sorted(set(layers)):
        raise ValueError("invalid ordered legal layer set or stratum count")
    quotient, remainder = divmod(len(layers), count)
    sizes = [quotient + (index < remainder) for index in range(count)]
    result = []
    cursor = 0
    for size in sizes:
        result.append(layers[cursor:cursor + size])
        cursor += size
    assert cursor == len(layers) and all(result)
    return result


def tabu_starts(revision, seed, layer_count, k):
    if not isinstance(revision, str) or not revision or seed not in (11, 23, 37):
        raise ValueError("unfrozen model revision or seed")
    if layer_count not in (28, 32) or k not in (1, 2, 4):
        raise ValueError("Phase B permits only fixed model depths and k=1,2,4")
    legal = list(range(1, layer_count - 1))
    strata = depth_strata(legal, 8 if k == 1 else k)
    starts = []
    for restart in range(8):
        picks = []
        buckets = [(restart, strata[restart])] if k == 1 else list(enumerate(strata))
        for partition, bucket in buckets:
            prefix = f"phaseB-start-v1|{revision}|{seed}|{k}|{restart}|{partition}|"
            picks.append(min(bucket, key=lambda layer: hashlib.sha256((prefix + str(layer)).encode()).digest()))
        starts.append(sorted(picks))
    assert len(starts) == 8 and all(len(mask) == k and len(set(mask)) == k for mask in starts)
    return starts
