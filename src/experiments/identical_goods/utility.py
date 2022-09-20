import numpy as np

from src import utility


class IdenticalGoodsUtility(utility.Utility):
    def __init__(self, utilities):
        self._utilities = utilities

    def __call__(self, goods_num):
        assert goods_num < len(self._utilities)
        return self._utilities[goods_num]


def cum_sum(array: np.ndarray, initial_val=0):
    return np.cumsum(np.insert(array, initial_val, 0))


def gen_utility_profiles(samples: np.ndarray):
    sorted_inc = np.sort(samples)
    sorted_dec = sorted_inc[::-1]
    return (
        IdenticalGoodsUtility(cum_sum(s))
        for s in (samples, sorted_inc, sorted_dec)
    )


def gen_composite_utility_profiles(samples_g, samples_d, epsilon, lmbd):
    composite_profile = np.array([
        lmbd*((1-epsilon)*d + epsilon*g)
        for g, d in zip(samples_g, samples_d)
    ])
    return gen_utility_profiles(composite_profile)
