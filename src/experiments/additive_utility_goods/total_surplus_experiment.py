import enum
import typing as tp

import numpy as np

from src.experiments import experiment
from src.experiments.additive_utility_goods import utility
from src.experiments.additive_utility_goods import guarantees as gr
from src.experiments.additive_utility_goods import (
    auctions_safe_play as auctions
)


class TotalSurplusExperiment(experiment.Experiment):
    result_path = 'data/total_surplus.csv'
    columns = [
        'utility_type',

        'goods_num',
        'scale_param',
        'mix_param',

        'min',
        'max',
        'dc1_total_surplus',
        'sb_total_surplus',
        'ba_total_surplus'
    ]

    calculation_params = {
        'goods_num': [3, 5, 10, 20],  # m
        'scale_param': [1, 1.5, 2, 5],  # lambda
        'mix_param': [0, 0.1, 0.5, 0.8],  # epsilon
    }

    class UtilityNames(enum.Enum):
        additive = 'additive'

    def _make_utilities(self, goods_num, scale_param, mix_param):
        # noqa
        # noinspection Duplicates
        samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)
        # noqa
        samples_delta = np.random.uniform(low=0.0, high=1.0, size=goods_num)
        u = utility.gen_utility_profile(samples_gamma)
        v = utility.gen_composite_utility_profile(
            samples_gamma,
            samples_delta,
            mix_param,
            scale_param
        )

        return [
            {'name': self.UtilityNames.additive.value, 'utility': [u, v]},
        ]

    def calculate(self, goods_num, scale_param, mix_param) -> tp.List[tp.Dict]:
        def auction_surplus(auction):
            def inner(utilities, goods):
                return auction(utilities, goods).total_surplus()
            return inner

        def max_utility(utilities, goods):
            return max(u(goods) for u in utilities)

        guarantees = {
            'dc1_total_surplus': auction_surplus(auctions.DivideAndChoose1),
            'sb_total_surplus': auction_surplus(auctions.SellAndBuy2Agents),

            'min': gr.min_utility,
            'max': gr.max_utility,

            'ba_total_surplus': max_utility,
        }

        goods_set = list(range(goods_num))
        results = []
        for ut in self._make_utilities(goods_num, scale_param, mix_param):
            results.append(
                {
                    **self._calc_normalized(
                        guarantees,
                        guarantees['max'],
                        ut['utility'],
                        goods_set
                    ),
                    'utility_type': ut['name'],
                    'goods_num': goods_num,
                    'scale_param': scale_param,
                    'mix_param': mix_param,
                }
            )

        return results
