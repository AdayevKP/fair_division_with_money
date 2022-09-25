from src.experiments import (
    bargaining_gap_experiment as bg_ig,
    compare_guarantees_experiment as cg_ig,
    total_surplus_experiment as ts_ig
)
from src.experiments.additive_utility_goods import (
    bargaining_gap_experiment as gb_au,
    total_surplus_experiment as ts_au
)


EXP_NUMBER = 1000


def run():
    experiments = [
        cg_ig.CompareGuaranteeExperiment(),
        bg_ig.BargainingGapExperiment(),
        bg_ig.BargainingGapMixedUtilitiesExperiment(),
        ts_ig.TotalSurplusExperiment(),
        gb_au.BargainingGapExperiment(),
        ts_au.TotalSurplusExperiment()
    ]

    for exp in experiments:
        print(f'run {exp}')
        exp.run_and_save(EXP_NUMBER)
        print(f'{exp} done!')


if __name__ == '__main__':
    run()
