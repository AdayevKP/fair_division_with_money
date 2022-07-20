import dataclasses as dc
import numpy as np
import typing as tp
import itertools as it

from src import guarantees as gr
from src import utility

from src.experiments.identical_goods import utility as ig_utility
from src.experiments.identical_goods import helpers as ig_helpers


TotalSurplus = float


def _iter_bijections(s1, s2):
    for perm in it.permutations(s2):
        yield list(zip(s1, perm))


class Allocation(tp.NamedTuple):
    goods: tp.Any
    transfer: float


class PiAuction:
    def __init__(
            self,
            agents_utilities: tp.List[utility.Utility],
            partition: tp.List
    ):
        assert len(agents_utilities) == len(partition)

        self._partition: tp.List = partition
        self._utilities: tp.List[utility.Utility] = agents_utilities
        self._agents_num = len(self._utilities)

    def _make_transfers(self) -> tp.List[tp.List[float]]:
        all_agents_transfers = []

        for u in self._utilities:
            transfers = [-u(p) for p in self._partition]
            mean = sum(transfers) / len(transfers)
            transfer = [t - mean for t in transfers]
            all_agents_transfers.append(transfer)

        return all_agents_transfers

    def allocations(self) -> tp.List[Allocation]:
        transfers = self._make_transfers()

        best_bij = []
        best_sum = float('inf')

        indexes = range(self._agents_num)
        for bij in _iter_bijections(indexes, indexes):
            tr_sum = sum(
                transfers[agent_idx][part_idx]
                for agent_idx, part_idx in bij
            )
            if tr_sum < best_sum:
                best_sum = tr_sum
                best_bij = bij

        slack_per_agent = best_sum / self._agents_num

        return [
            Allocation(
                goods=self._partition[part_idx],
                transfer=transfers[agent_idx][part_idx] - slack_per_agent
            )
            for agent_idx, part_idx in best_bij
        ]

    def total_surplus(self) -> TotalSurplus:
        surplus = 0.0

        for agent_idx, allocation in enumerate(self.allocations()):
            surplus += self._utilities[agent_idx](allocation.goods)
            surplus += allocation.transfer

        return surplus


class AveragingAuction:
    def __init__(
            self,
            agents_utilities: tp.List[utility.Utility],
            agents_partitions: tp.List[tp.List],
    ):
        self._partitions = agents_partitions
        self._utilities = agents_utilities
        self._agents_num = len(self._utilities)

    def _single_agent_transfers(
            self, agent_utility: utility.Utility
    ) -> tp.List[float]:
        mean_gr = sum(
            gr.fixed_partition_guarantee(p, agent_utility)
            for p in self._partitions
        ) / len(self._partitions)

        return [
            mean_gr - gr.fixed_partition_guarantee(p, agent_utility)
            for p in self._partitions
        ]

    def _make_transfers(self) -> tp.List[tp.List[float]]:
        return [
            self._single_agent_transfers(u)
            for u in self._utilities
        ]

    def _choose_partition(self):
        transfers = self._make_transfers()
        gr_transfers_sums = [sum(gr_tr) for gr_tr in zip(*transfers)]

        min_sum_index = np.argmin(gr_transfers_sums)

        return self._partitions[min_sum_index]

    def allocations(self) -> tp.List[Allocation]:
        return PiAuction(
            self._utilities, self._choose_partition()
        ).allocations()

    def total_surplus(self) -> TotalSurplus:
        return PiAuction(
            self._utilities, self._choose_partition()
        ).total_surplus()


def test_pi_auction():
    utilities = []
    for _ in range(2):
        samples_delta = np.random.uniform(low=0.0, high=1.0, size=10)  # noqa
        ut, *_ = ig_utility.gen_utility_profiles(samples_delta)
        utilities.append(ut)

    res = PiAuction(utilities, [8, 2]).total_surplus()
    print(res)


def test_avg_auction():
    utilities = []
    for _ in range(2):
        samples_delta = np.random.uniform(low=0.0, high=1.0, size=10)  # noqa
        ut, *_ = ig_utility.gen_utility_profiles(samples_delta)
        utilities.append(ut)

    res = AveragingAuction(
        utilities,
        ig_helpers.goods_partitions(2, 10)
    ).total_surplus()

    print(res)


if __name__ == '__main__':
    test_avg_auction()
    test_pi_auction()
