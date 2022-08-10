import enum
import typing as tp

import numpy as np
import toolz as tls

from src.experiments import experiment
from src.experiments.identical_goods import utility
from src.experiments.identical_goods import guarantees as gr


class BargainingGapExperiment(experiment.Experiment):
    result_path = 'results/bargaining_gap.csv'
    columns = {
        'utility_type',

        'goods_num',
        'scale_param',
        'mix_param',

        'min',
        'max',
        'familiar_share_gr_sum',
        'sb_gr_sum',
        'dc_gr_sum'
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
        def sum_gr(func):
            def inner(ut_profile, n):
                return sum(func(u, n) for u in ut_profile)
            return inner

        guarantees = {
            'min': tls.compose(min, gr.partitions_utilities),
            'max': tls.compose(max, gr.partitions_utilities),
            'familiar_share_gr_sum': sum_gr(gr.familiar_share),
            'sb_gr_sum': sum_gr(gr.guarantee_sb),
            'dc_gr_sum': sum_gr(gr.guarantee_dc),
        }

        results = []
        for ut in self._make_utilities(goods_num, scale_param, mix_param):
            results.append(
                {
                    **{
                        name: g(ut['utility'], goods_num)
                        for name, g in guarantees.items()
                    },
                    'utility_type': ut['name'],
                    'goods_num': goods_num,
                    'scale_param': scale_param,
                    'mix_param': mix_param,
                }
            )

        return results


class BargainingGapMixedUtilitiesExperiment(BargainingGapExperiment):
    result_path = 'results/bargaining_gap_mixed_utilities.csv'

    class UtilityNames(enum.Enum):
        random_concave = 'random-cav'
        concave_convex = 'cav-vex'

    def _make_utilities(self, goods_num, scale_param, mix_param):
        # noinspection Duplicates
        # noqa
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
            {
                'name': self.UtilityNames.random_concave.value,
                'utility': [u, v_cav]
            },
            {
                'name': self.UtilityNames.concave_convex.value,
                'utility': [u_cav, v_vex]
            },
        ]


def run_experiment():
    BargainingGapExperiment().run_and_save(10)
    BargainingGapMixedUtilitiesExperiment().run_and_save(10)


if __name__ == '__main__':
    run_experiment()

