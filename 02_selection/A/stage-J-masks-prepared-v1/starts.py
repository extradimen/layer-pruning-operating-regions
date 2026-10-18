"""Predeclared independent starts, never selected from Final results."""
import random
BINS=[list(range(1,11)),list(range(11,21)),list(range(21,31))]
def block_starts(k,seed):
 if k not in [3,6,9]:raise ValueError('fixed deletion level')
 rng=random.Random(seed);return tuple(tuple(sorted(i for group in BINS for i in rng.sample(group,k//3))) for _ in range(8))
def branch_starts(seed):
 return tuple(tuple(list(m)+[i+32 for i in m]) for m in block_starts(6,seed))
