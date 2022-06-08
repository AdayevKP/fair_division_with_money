import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.guarantees import identical_goods_guarantees as gr
from src.experiments.identical_goods import utility


RESULTS_PATH = 'results/guarantees.csv'
STATS_RESULTS_PATH = 'results/guarantees_mean.csv'
GOODS_NUM = (3, 5, 10, 20)
guarantees = {
    'min_max': gr.min_max_single,
    'max_min': gr.max_min_single,
    'familiar_share': gr.familiar_share,
    'sb': gr.guarantee_sb,
    'dc': gr.guarantee_dc
}


def calc_guarantees_and_save_results():
    exp_results = pd.DataFrame(
        columns=[
            'goods_num',
            'exp_number',
            'utility_type',
            *guarantees.keys()
        ]
    )

    for exp_num in range(100):
        for goods_num in GOODS_NUM:
            samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa
            u, u_vex, u_cav = utility.gen_utility_profiles(samples_gamma)

            for u_name, ut in zip(('regular', 'vex', 'cav'), (u, u_vex, u_cav)):
                row = {
                    'exp_number': exp_num,
                    'goods_num': goods_num,
                    'utility_type': u_name,
                    **{name: g(ut, goods_num) for name, g in guarantees.items()}
                }
                exp_results = exp_results.append(
                    row,
                    ignore_index=True,
                )

    exp_results.to_csv(RESULTS_PATH)


def plot_results():
    data = pd.read_csv(STATS_RESULTS_PATH)

    cols = ['goods_num', *guarantees.keys()]
    for utility_type in ('regular', 'vex', 'cav'):
        data.loc[
            data['utility_type'] == utility_type
        ][cols].plot(x='goods_num', title=utility_type)

    plt.show()


def calc_stats():
    data = pd.read_csv(RESULTS_PATH)
    results = pd.DataFrame(
        columns=[
            'goods_num',
            'utility_type',
            *guarantees.keys()
        ]
    )

    for goods_num in GOODS_NUM:
        for utility_type in ('regular', 'vex', 'cav'):
            selection = data.loc[
                (data['utility_type'] == utility_type) &
                (data['goods_num'] == goods_num),
            ][['goods_num', *guarantees.keys()]].mean()

            selection['utility_type'] = utility_type
            results = results.append(selection, ignore_index=True)

    results.to_csv(STATS_RESULTS_PATH)


def run():
    calc_guarantees_and_save_results()
    calc_stats()
    plot_results()


if __name__ == '__main__':
    run()
