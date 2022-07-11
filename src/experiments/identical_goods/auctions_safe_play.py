import typing as tp

from src import auctions
from src.experiments.identical_goods import guarantees as gr
from src.experiments.identical_goods import utility
from src.experiments.identical_goods import helpers


class DivideAndChoose1:
    def __init__(
            self,
            agents_utilities: tp.List[utility.IdenticalGoodsUtility],
            goods_num: int
    ):
        self._agents_num = len(agents_utilities)
        self._goods_num = goods_num
        self._utilities = agents_utilities

    def _make_bids(self) -> tp.List[float]:
        n = self._agents_num

        bids = []
        for u in self._utilities:
            bid = (
                    gr.max_min_single(u, self._goods_num, n) -
                    gr.min_max_single(u, self._goods_num, n)
            )
            bids.append(bid)

        return bids

    def _choose_partition(
            self, divider_utility: utility.IdenticalGoodsUtility
    ) -> tp.List[int]:
        partitions = helpers.goods_partitions(self._agents_num, self._goods_num)
        partitions_utilities = [
            sum([divider_utility(goods_num) for goods_num in partition])
            for partition in partitions
        ]

        max_utility = max(partitions_utilities)
        best_partition_idx = partitions_utilities.index(max_utility)

        return partitions[best_partition_idx]

    def run(self) -> auctions.TotalSurplus:
        bids: tp.List[float] = self._make_bids()

        max_bid = max(bids)
        divider_index = bids.index(max_bid)

        partition = self._choose_partition(self._utilities[divider_index])

        return auctions.PiAuction(self._utilities, partition).run()

