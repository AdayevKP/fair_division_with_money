import numpy as np
import typing as tp
import operator as op

from src.guarantees import identical_goods_guarantees as gr
from src.experiments.identical_goods import utility

fst = op.itemgetter(0)
snd = op.itemgetter(1)


def sort_guarantees(utility_shape, goods_num):
    guarantees: tp.List[tp.Callable] = [
        gr.min_max_single,
        gr.max_min_single,
        gr.familiar_share,
        gr.guarantee_sb,
        gr.guarantee_dc
    ]

    gr_values = [(g.__name__, g(utility_shape, goods_num)) for g in guarantees]
    return sorted(gr_values, key=snd)


def compare_guarantees():
    for goods_num in (3, 5, 10, 20):
        samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa
        u, u_vex, u_cav = utility.gen_utility_profiles(samples_gamma)

        print(f'goods_num: {goods_num}')
        ut = "\n".join(map(str, sort_guarantees(u, goods_num)))
        print(f'random utilities:\n{ut}')
        vex_ut = "\n".join(map(str, sort_guarantees(u_vex, goods_num)))
        print(f'vex utilities:\n{vex_ut}')
        cav_ut = "\n".join(map(str, sort_guarantees(u_cav, goods_num)))
        print(f'cav utilities:\n{cav_ut}\n')
