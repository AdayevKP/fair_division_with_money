import itertools as it
import typing as tp


def goods_partitions(agents_num: int, goods_num: int) -> tp.List[tp.List[int]]:
    m = goods_num
    n = agents_num

    return [
        comb for comb in it.product(range(m + 1), repeat=n)
        if sum(comb) == m
    ]
