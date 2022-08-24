import enum
import typing as tp

import numpy as np
import toolz as tls

from src.experiments import experiment
from src.experiments.identical_goods import utility
from src.experiments.identical_goods import guarantees as gr
from src.experiments.identical_goods import auctions_safe_play as auctions


class TotalSurplusExperiment(experiment.Experiment):
    result_path = 'data/total_surplus.csv'
    columns = {
        'utility_type',

        'goods_num',
        'scale_param',
        'mix_param',

        'min',
        'max',
        'dc1_total_surplus',
        'dc2_total_surplus',
        'sb_total_surplus',
        'max_utility'
    }

    calculation_params = {
        'goods_num': [3, 5, 10, 20],  # m
        'scale_param': [1, 1.5, 2, 5],  # lambda
        'mix_param': [0, 0.1, 0.5, 0.8],  # epsilon
    }

    class UtilityNames(enum.Enum):
        random = 'random'
        convex = 'vex'
        concave = 'cav'

    def _make_utilities(self, goods_num, scale_param, mix_param):
        # noqa
        # noinspection Duplicates
        samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)
        # noqa
        samples_delta = np.random.uniform(low=0.0, high=1.0, size=goods_num)
        u, u_vex, u_cav = utility.gen_utility_profiles(samples_gamma)
        v, v_vex, v_cav = utility.gen_composite_utility_profiles(
            samples_gamma,
            samples_delta,
            mix_param,
            scale_param
        )

        return [
            {'name': self.UtilityNames.random.value, 'utility': [u, v]},
            {'name': self.UtilityNames.convex.value, 'utility': [u_vex, v_vex]},
            {'name': self.UtilityNames.concave.value, 'utility': [u_cav, v_cav]},
        ]

    def calculate(self, goods_num, scale_param, mix_param) -> tp.List[tp.Dict]:
        def auction_surplus(auction):
            def inner(utilities, goods_n):
                return auction(utilities, goods_n).total_surplus()
            return inner

        def max_utility(utilities, goods_n):
            return max(u(goods_n) for u in utilities)

        guarantees = {
            'dc1_total_surplus': auction_surplus(auctions.DivideAndChoose1),
            'dc2_total_surplus': auction_surplus(auctions.DivideAndChoose2),
            'sb_total_surplus': auction_surplus(auctions.SellAndBuy2Agents),

            'min': tls.compose(min, gr.partitions_utilities),
            'max': tls.compose(max, gr.partitions_utilities),

            'max_utility': max_utility,
        }

        results = []
        for ut in self._make_utilities(goods_num, scale_param, mix_param):
            results.append(
                {
                    **self._calc_normalized(
                        guarantees,
                        guarantees['max'],
                        ut['utility'],
                        goods_num
                    ),
                    'utility_type': ut['name'],
                    'goods_num': goods_num,
                    'scale_param': scale_param,
                    'mix_param': mix_param,
                }
            )

        return results


def run_experiment():
    TotalSurplusExperiment().run_and_save(10)


if __name__ == '__main__':
    run_experiment()

