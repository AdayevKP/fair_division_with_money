import typing as tp
import numpy as np

from src import auctions
from src.experiments.identical_goods import guarantees as gr
from src.experiments.identical_goods import utility
from src.experiments.identical_goods import helpers


class Auction:
    def __init__(
            self,
            agents_utilities: tp.List[utility.IdenticalGoodsUtility],
            goods_num: int
    ):
        self.agents_num = len(agents_utilities)
        self.goods_num = goods_num
        self.utilities = agents_utilities

    def total_surplus(self) -> auctions.TotalSurplus:
        raise NotImplementedError()

    def allocations(self) -> tp.List[auctions.Allocation]:
        raise NotImplementedError()


class DivideAndChoose1(Auction):
    def _make_bids(self) -> tp.List[float]:
        n = self.agents_num

        bids = []
        for u in self.utilities:
            bid = (
                    gr.max_min_single(u, self.goods_num, n) -
                    gr.min_max_single(u, self.goods_num, n)
            )
            bids.append(bid)

        return bids

    def _choose_partition(self) -> tp.Tuple[tp.List[int], tp.List[float]]:
        bids: tp.List[float] = self._make_bids()
        divider_index = np.argmax(bids)
        divider_utility = self.utilities[divider_index]

        partitions = helpers.goods_partitions(self.agents_num, self.goods_num)
        partitions_utilities = [
            sum([divider_utility(goods_num) for goods_num in partition])
            for partition in partitions
        ]

        best_partition_idx = np.argmax(partitions_utilities)

        divider_transfer = bids[divider_index]/self.agents_num
        transfers = [divider_transfer]*self.agents_num
        transfers[divider_index] *= -divider_transfer*(self.agents_num-1)

        return partitions[best_partition_idx], transfers

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


class DivideAndChoose2(Auction):
    def _propose_partitions(self):
        all_partitions: tp.List[tp.List[int]] = helpers.goods_partitions(
            self.agents_num, self.goods_num
        )

        selected_partitions = []

        for u in self.utilities:
            partitions_utilities = [
                sum(u(p) for p in partition)
                for partition in all_partitions
            ]
            partition_idx = np.argmax(partitions_utilities)
            selected_partitions.append(all_partitions[partition_idx])

        return selected_partitions

    def total_surplus(self) -> auctions.TotalSurplus:
        partitions: tp.List[tp.List[int]] = self._propose_partitions()
        avg_auction = auctions.AveragingAuction(self.utilities, partitions)
        return avg_auction.total_surplus()


class SellAndBuy2Agents(Auction):
    def _make_bids(self):
        bids = []
        for u in self.utilities:
            _, bid = gr.guarantee_sb_2(u, self.goods_num)
            bids.append(bid)

        return bids

    def total_surplus(self) -> auctions.TotalSurplus:
        bids = self._make_bids()
        seller_index = np.argmin(bids)

        seller_utility = self.utilities[seller_index]
        buyer_utility = self.utilities[seller_index - 1]

        price = seller_utility(self.goods_num) / self.goods_num

        bought_goods = np.argmax(
            [buyer_utility(k) - price * k for k in range(self.goods_num)]
        )
        goods_price = price * bought_goods

        return (
            buyer_utility(bought_goods) - goods_price +
            seller_utility(self.goods_num - bought_goods) + goods_price
        )

