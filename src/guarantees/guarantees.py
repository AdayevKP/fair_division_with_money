import numpy as np
import typing as tp
import itertools as it


def _partitions_utilities(utility_profile: tp.List[np.ndarray]):
    m = utility_profile[0].size

    partitions = [
        comb for comb in it.combinations(range(m), len(utility_profile))
        if sum(comb) == m - 1
    ]
    return [
        sum(u[p] for p, u in zip(prt, utility_profile))
        for prt in partitions
    ]


def max_min(utility_profile: tp.List[np.ndarray]):
    return max(_partitions_utilities(utility_profile)) / len(utility_profile)


def min_max(utility_profile: tp.List[np.ndarray]):
    return min(_partitions_utilities(utility_profile)) / len(utility_profile)

