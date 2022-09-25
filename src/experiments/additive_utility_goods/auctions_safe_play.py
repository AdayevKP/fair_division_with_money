import typing as tp
import numpy as np
import random

from src import auctions
from src.experiments.additive_utility_goods import guarantees as gr
from src.experiments.additive_utility_goods import utility


class Auction:
    def __init__(
            self,
            agents_utilities: tp.List[utility.AdditiveGoodsUtility],
            goods: tp.List[int]
    ):
        self.agents_num = len(agents_utilities)
        self.goods = goods
        self.utilities = agents_utilities

    def allocations(self) -> tp.List[auctions.Allocation]:
        raise NotImplementedError()

    def total_surplus(self) -> auctions.TotalSurplus:
        raise NotImplementedError()


def _divide_almost_equally(goods, ut, parts=2):
    if parts == 1:
        return [goods]
    if not goods:
        return [[]]

    sorted_goods = sorted(goods, key=lambda g: ut([g]))
    part1 = []
    part2 = []
    for g in sorted_goods:
        if ut(part1) < ut(part2):
            part1.append(g)
        else:
            part2.append(g)

    if parts == 2:
        return [part1, part2]

    result = []

    new_parts = parts//2

    result.extend(_divide_almost_equally(part1, ut, new_parts))
    result.extend(_divide_almost_equally(part2, ut, parts-new_parts))
    return result


class DivideAndChoose1(Auction):
    def _choose_partition(
            self
    ) -> tp.Tuple[tp.List[tp.List[int]], tp.List[float]]:
        divider_index = random.randint(0, self.agents_num - 1)
        divider_utility = self.utilities[divider_index]

        partition = _divide_almost_equally(self.goods, divider_utility, 2)
        transfers = [0]*self.agents_num

        return partition, transfers

    def allocations(self):
        partition, transfers = self._choose_partition()
        pi_allocations = auctions.PiAuction(
            self.utilities, partition
        ).allocations()

        return [
            auctions.Allocation(
                goods=al.goods,
                transfer=al.transfer + tr
            )
            for tr, al in zip(transfers, pi_allocations)
        ]

    def total_surplus(self) -> auctions.TotalSurplus:
        partition, _ = self._choose_partition()
        return auctions.PiAuction(self.utilities, partition).total_surplus()


class SellAndBuy2Agents(Auction):
    def _make_bids(self):
        return [gr.guarantee_sb(u, self.goods) for u in self.utilities]

    def allocations(self) -> tp.List[auctions.Allocation]:
        bids = self._make_bids()
        seller_index = int(np.argmin(bids))
        buyer_index = seller_index - 1

        buyer_utility = self.utilities[buyer_index]
        prices = [
            self.utilities[seller_index]([g])/self.agents_num
            for g in self.goods
        ]

        bought_goods = []
        goods_price = 0
        for p, g in zip(prices, self.goods):
            if buyer_utility([g]) - p > 0:
                bought_goods.append(g)
                goods_price += p

        allocations = [
            auctions.Allocation(
                goods=list(set(self.goods) - set(bought_goods)),
                transfer=goods_price
            ),
            auctions.Allocation(
                goods=bought_goods, transfer=-goods_price
            )
        ]

        if seller_index == 1:
            allocations[0], allocations[1] = allocations[1], allocations[0]

        return allocations

    def total_surplus(self) -> auctions.TotalSurplus:
        return sum(
            ut(al.goods) + al.transfer
            for al, ut in zip(self.allocations(), self.utilities)
        )
