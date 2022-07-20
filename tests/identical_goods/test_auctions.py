import pytest

from src import auctions
from src.experiments.identical_goods import utility as ut


@pytest.fixture(scope='function')
def utility():
    def make(utilities):
        return ut.IdenticalGoodsUtility(utilities)

    yield make


def test_pi_auction(utility):
    agent1 = [0, 1, 4, 9]
    agent2 = [0, 2, 7, 10]

    # 1 good for and 2 goods
    partition = [1, 2]

    pi_auction = auctions.PiAuction(
        agents_utilities=[utility(agent1), utility(agent2)],
        partition=partition
    )

    """
    agent1: u(1) = 1, u(2) = 4
            t = 1.5   t = -1.5
            
    agent2: u(1) = 2, u(2) = 7
            t = 2.5   t = -2.5
            
    [t12, t21] = [-1.5, 2.5] => sum = 1
    [t11, t22] = [1.5, -2.5] => sum = -1 -> choose this allocation
    slack = 0.5
    """

    assert pi_auction.allocations() == [
        auctions.Allocation(goods=1, transfer=1.5 + 0.5),
        auctions.Allocation(goods=2, transfer=-2.5 + 0.5),
    ]
    "sum only utilities because money returned back"
    assert pi_auction.total_surplus() == 1 + 7



