import numpy as np

from src.experiments.identical_goods import auctions_safe_play as auctions
from src.experiments.identical_goods import utility
from src.experiments.identical_goods import guarantees as gr

GOODS_NUM = (3, 5, 10, 20)


def run():
    goods_num = 20
    samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa
    samples_delta = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa

    u1, *_ = utility.gen_utility_profiles(samples_gamma)
    u2, *_ = utility.gen_utility_profiles(samples_delta)

    utility_profile = [u1, u2]

    dc2_surplus = auctions.DivideAndChoose2(utility_profile, goods_num).run()
    dc1_surplus = auctions.DivideAndChoose1(utility_profile, goods_num).run()
    u_p = gr.partitions_utilities(utility_profile, goods_num)

    print(min(u_p), dc1_surplus, dc2_surplus, max(u_p))


if __name__ == "__main__":
    run()
