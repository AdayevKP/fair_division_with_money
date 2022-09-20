import numpy as np
import typing as tp

from src import utility
from src.experiments.identical_goods import helpers


def guarantee_sb_2(u: utility.Utility, goods_num):
    m = goods_num

    x = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for l in range(m + 1):
            if l + k > m:
                continue
            x[k][l] = m * (u(l + k) - u(l)) / (m + k)

    k_max, l_max = np.unravel_index(np.argmax(x, axis=None), x.shape)
    optimal_bid = x[k_max][l_max]
    return (
            (l_max + k_max) * u(l_max) / (m + k_max) +
            (m - l_max) * u(l_max + k_max) / (m + k_max),
            optimal_bid
    )


def guarantee_sb(u: utility.Utility, goods_num, agents_number=2):
    if agents_number == 2:
        gr, _ = guarantee_sb_2(u, goods_num)
        return gr

    raise ValueError(
        'sell and buy rule guarantee not implemented for more than 2 agents'
    )


def guarantee_dc(u: utility.Utility, goods_num, agents_number=2):
    n = agents_number
    return max_min([u] * n, goods_num) / n + (n - 1) * min_max([u] * n, goods_num) / n


def familiar_share(u: utility.Utility, goods_num, agents_number=2):
    return u(goods_num) / agents_number


def partitions_utilities(utility_profile: tp.List[utility.Utility], goods_num):
    partitions = helpers.goods_partitions(len(utility_profile), goods_num)
    return [
        sum(u(p) for p, u in zip(prt, utility_profile))
        for prt in partitions
    ]


def max_min_single(u: utility.Utility, goods_num, agents_num=2):
    return max_min([u] * agents_num, goods_num)


def min_max_single(u: utility.Utility, goods_num, agents_num=2):
    return min_max([u] * agents_num, goods_num)


def max_min(utility_profile: tp.List[utility.Utility], goods_num):
    return max(partitions_utilities(utility_profile, goods_num)) / len(utility_profile)


def min_max(utility_profile: tp.List[utility.Utility], goods_num):
    return min(partitions_utilities(utility_profile, goods_num)) / len(utility_profile)
