import operator as op
import numpy as np

from src.experiments.identical_goods import guarantees as gr
from src.experiments.identical_goods import utility as ut

fst = op.itemgetter(0)
snd = op.itemgetter(1)


def sort_guarantees(utility_profile, goods_num):
    u_p = gr.partitions_utilities(utility_profile, goods_num)
    metrics = [
        ('min_u', min(u_p)),
        ('gr_dc_sum', sum(gr.guarantee_dc(u, goods_num) for u in utility_profile)),
        ('gr_sb_sum', sum(gr.guarantee_sb(u, goods_num) for u in utility_profile)),
        ('gr_fs_sum', sum(gr.familiar_share(u, goods_num) for u in utility_profile)),
        ('max_u', max(u_p)),
    ]

    return '\n'.join(f'{name}: {val}'for name, val in sorted(metrics, key=snd))


def run(goods_num, lmbd, epsilon):
    samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa
    samples_delta = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa
    u = ut.gen_utility_profiles(samples_gamma)
    v = ut.gen_composite_utility_profiles(
        samples_gamma,
        samples_delta,
        epsilon,
        lmbd
    )

    for u, v, t in zip(u, v, ['', 'vex', 'cav']):
        print(
            f"(u{t}, v{t}), goods: {goods_num}, lambda: {lmbd}, epsilon: {epsilon}\n"
            f"{sort_guarantees([u, v], goods_num)}\n"
        )


def check_bargaining_gap():
    for goods_num in (3, 5, 10, 20):
        for l in (1, 1.5, 2, 5):
            for e in (0, 0.1, 0.5, 0.8):
                run(goods_num, l, e)
