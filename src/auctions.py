import numpy as np
import typing as tp
import itertools as it

from src import utility

from src.experiments.identical_goods import utility as ut

TotalSurplus = float


def _make_transfers(
        utility_fun: utility.Utility, partition: tp.List
) -> tp.List[float]:
    transfers = [-utility_fun(p) for p in partition]
    mean = sum(transfers) / len(transfers)
    return [t - mean for t in transfers]


def iter_bijections(s1, s2):
    for perm in it.permutations(s1):
        yield list(zip(perm, s2))


def pi_auction(
        agents_utilities: tp.List[utility.Utility],
        partition: tp.List
) -> TotalSurplus:
    assert len(agents_utilities) == len(partition)

    transfers = [
        _make_transfers(u, partition)
        for u in agents_utilities
    ]

    best_bij = []
    best_sum = float('inf')

    agents_num = len(agents_utilities)
    indexes = range(agents_num)
    for bij in iter_bijections(indexes, indexes):
        tr_sum = sum(
            transfers[agent_idx][part_idx]
            for agent_idx, part_idx in bij
        )
        if tr_sum < best_sum:
            best_sum = tr_sum
            best_bij = bij

    slack_per_agent = best_sum/agents_num

    total_surplus = sum(
        agents_utilities[agent_idx](partition[part_idx]) +
        transfers[agent_idx][part_idx] - slack_per_agent
        for agent_idx, part_idx in best_bij
    )

    return total_surplus


if __name__ == '__main__':
    utilities = []
    for _ in range(2):
        samples_delta = np.random.uniform(low=0.0, high=1.0, size=10)  # noqa
        ut, *_ = ut.gen_utility_profiles(samples_delta)
        utilities.append(ut)

    res = pi_auction(utilities, [8, 2])
    print(res)
