import numpy as np
import typing as tp
import operator as op

from src.guarantees import identical_goods_guarantees as gr

fst = op.itemgetter(0)
snd = op.itemgetter(1)


def gen_utility_profiles(samples: np.ndarray):
    samples_with_zero = np.insert(samples, 0, 0)
    sorted_inc = np.sort(samples_with_zero)
    sorted_dec = sorted_inc[::-1]
    return np.cumsum(samples), np.cumsum(sorted_inc), np.cumsum(sorted_dec)


def gen_composite_utility_profiles(samples_g, samples_d, epsilon, lmbd):
    composite_profile = np.array([
        lmbd*((1-epsilon)*d + epsilon*g)
        for g, d in zip(samples_g, samples_d)
    ])
    return gen_utility_profiles(composite_profile)


def sort_guarantees(utility_shape):
    guarantees: tp.List[tp.Callable] = [
        gr.min_max,
        gr.max_min,
        gr.familiar_share,
        gr.guarantee_sb,
        gr.guarantee_dc
    ]

    gr_values = [(g.__name__, g(utility_shape)) for g in guarantees]
    return sorted(gr_values, key=snd)


def compare_guarantees():
    for goods_num in (3, 5, 10, 20):
        samples_gamma = np.random.uniform(low=0.0, high=1.0, size=goods_num)  # noqa
        u, u_vex, u_cav = gen_utility_profiles(samples_gamma)

        print(f'goods_num: {goods_num}')
        ut = "\n".join(map(str, sort_guarantees(u)))
        print(f'random utilities:\n{ut}')
        vex_ut = "\n".join(map(str, sort_guarantees(u_vex)))
        print(f'vex utilities:\n{vex_ut}')
        cav_ut = "\n".join(map(str, sort_guarantees(u_cav)))
        print(f'cav utilities:\n{cav_ut}\n')


def run():
    compare_guarantees()


if __name__ == '__main__':
    run()
