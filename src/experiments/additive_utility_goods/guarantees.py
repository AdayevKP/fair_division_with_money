import typing as tp

from src import utility


def familiar_share(u: utility.Utility, goods_set, agents_number=2):
    return u(goods_set) / agents_number


def guarantee_sb(u: utility.Utility, goods_set, agents_number=2):
    return familiar_share(u, goods_set, agents_number)


def guarantee_dc(u: utility.Utility, goods_set, agents_number=2):
    return familiar_share(u, goods_set, agents_number)


def min_utility(
        utility_profile: tp.List[utility.Utility],
        goods_set: tp.Iterable[int],
):
    return sum(
        min([u([g]) for u in utility_profile])
        for g in goods_set
    )


def max_utility(
        utility_profile: tp.List[utility.Utility],
        goods_set: tp.Iterable[int],
):
    return sum(
        max([u([g]) for u in utility_profile])
        for g in goods_set
    )
