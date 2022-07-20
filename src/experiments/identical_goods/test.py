import numpy as np
import matplotlib.pyplot as plt

from src.experiments.identical_goods import guarantees as gr
from src.experiments.identical_goods import utility as ig_utility


def run():
    samples_delta = np.random.uniform(low=0.0, high=1.0, size=10)  # noqa
    ut, uvex, ucav = ig_utility.gen_utility_profiles(samples_delta)

    plt.plot(ut._utilities)
    plt.show()

    res = gr.guarantee_sb(ut, goods_num=10)

    print(f'{ut(10) / 2} == {res}')


if __name__ == '__main__':
    run()
