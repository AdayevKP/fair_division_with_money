import enum
import typing as tp

import numpy as np

from src.experiments import experiment
from src.experiments.identical_goods import utility
from src.experiments.identical_goods import guarantees as gr


class CompareGuaranteeExperiment(experiment.Experiment):
    result_path = 'results/guarantees.csv'
    columns = {
        'utility_type',
        'goods_num',

        'min_max_gr',
        'max_min_gr',
        'familiar_share_gr',
        'sb_gr',
        'dc_gr'
    }

    calculation_params = {
        'goods_num': [3, 5, 10, 20]
    }

    class UtilityNames(enum.Enum):
        random = 'random'
        convex = 'vex'
        concave = 'cav'

    def _make_utilities(self, goods_num):
        # noqa
        samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)
        u, u_vex, u_cav = utility.gen_utility_profiles(samples_gamma)

        return [
            {'name': self.UtilityNames.random.value, 'utility': u},
            {'name': self.UtilityNames.convex.value, 'utility': u_vex},
            {'name': self.UtilityNames.concave.value, 'utility': u_cav},
        ]

    def calculate(self, goods_num) -> tp.List[tp.Dict]:
        guarantees = {
            'min_max_gr': gr.min_max_single,
            'max_min_gr': gr.max_min_single,
            'familiar_share_gr': gr.familiar_share,
            'sb_gr': gr.guarantee_sb,
            'dc_gr': gr.guarantee_dc,
        }

        results = []
        for ut in self._make_utilities(goods_num):
            results.append(
                {
                    **{
                        name: g(ut['utility'], goods_num)
                        for name, g in guarantees.items()
                    },
                    'utility_type': ut['name'],
                    'goods_num': goods_num
                }
            )

        return results


def run_experiment():
    CompareGuaranteeExperiment().run_and_save(10)
    res = CompareGuaranteeExperiment.load_experiment()
    print(res)


if __name__ == '__main__':
    run_experiment()

