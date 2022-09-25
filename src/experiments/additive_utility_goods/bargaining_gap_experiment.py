import enum
import typing as tp

import numpy as np

from src.experiments import experiment
from src.experiments.additive_utility_goods import utility
from src.experiments.additive_utility_goods import guarantees as gr


class BargainingGapExperiment(experiment.Experiment):
    result_path = 'data/additive_utilities/bargaining_gap.csv'
    columns = [
        'utility_type',

        'goods_num',
        'scale_param',
        'mix_param',

        'min',
        'max',
        'familiar_share_gr_sum',
        'sb_gr_sum',
        'dc_gr_sum'
    ]

    calculation_params = {
        'goods_num': list(range(3, 21)),  # [3, 5, 10, 20],  # m
        'scale_param': [1, 1.5, 2, 5],  # lambda
        'mix_param': [0, 0.1, 0.5, 0.8],  # epsilon
    }

    class UtilityNames(enum.Enum):
        additive = 'additive_utility'

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
            {'name': self.UtilityNames.additive.value, 'utility': [u, v]}
        ]

    def calculate(self, goods_num, scale_param, mix_param) -> tp.List[tp.Dict]:
        def sum_gr(func):
            def inner(ut_profile, goods):
                return sum(func(u, goods) for u in ut_profile)
            return inner

        guarantees = {
            'min': gr.min_utility,
            'max': gr.max_utility,
            'familiar_share_gr_sum': sum_gr(gr.familiar_share),
            'sb_gr_sum': sum_gr(gr.guarantee_sb),
            'dc_gr_sum': sum_gr(gr.guarantee_dc),
        }

        results = []
        goods_set = list(range(goods_num))
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
