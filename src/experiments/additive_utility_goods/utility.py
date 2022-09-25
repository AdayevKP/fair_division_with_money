import typing as tp

import numpy as np

from src import utility


class AdditiveGoodsUtility(utility.Utility):
    def __init__(self, utilities):
        self._utilities = utilities

    # todo: make *goods instead of list
    def __call__(self, goods_indexes: tp.Iterable[int]):
        if not goods_indexes:
            return 0
        assert max(goods_indexes) < len(self._utilities)
        return sum(self._utilities[gi] for gi in goods_indexes)


def gen_utility_profile(samples: np.ndarray):
    return AdditiveGoodsUtility(samples)


def gen_composite_utility_profile(samples_g, samples_d, mix_param, scale_param):
    composite_profile = np.array([
        scale_param*((1-mix_param)*d + mix_param*g)
        for g, d in zip(samples_g, samples_d)
    ])
    return gen_utility_profile(composite_profile)
