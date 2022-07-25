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

    # 1 good and 2 goods
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
    "sum only utilities because money sum to 0"
    assert pi_auction.total_surplus() == 1 + 7


def test_averaging_auction(utility):
    agent1 = [0, 1, 4, 9]
    agent2 = [0, 2, 7, 10]

    partition1 = [1, 2]
    partition2 = [0, 3]

    avg_auction = auctions.AveragingAuction(
        agents_utilities=[utility(agent1), utility(agent2)],
        agents_partitions=[partition1, partition2]
    )

    """
    agent1:
        meanG = ((1 + 4)/2 + 9/2)/2 = 3.5
        G11 = (1 + 4)/2 = 2.5; t11 = 3.5 - 2.5 = 1
        G12 = 9/2 = 4.5;       t12 = 3.5 - 4.5 = -1
        
    agent2:
        meanG = ((2 + 7)/2 + 10/2)/2 = 4.75
        G21 = (2 + 7)/2 = 4.5; t21 = 4.75 - 4.5 = 0.25
        G22 = 10/2 = 5;        t22 = 4.75 - 5   = -0.25
        
    max = G12 + G22 -> choose partition2 
    slack = (t22 + t12)/2 = (-0.25 - 1)/2 = -0.625
    
    pi auction
    agent1: u(0) = 0, u(3) = 9
            t11 = 4.5   t12 = -4.5
            
    agent2: u(0) = 0, u(3) = 10
            t12 = 5    t22 = -5
            
    sum[t12, t21] = 0.5
    sum[t11, t22] = -0.5 -> choose this allocation
    slack = -0.25
    """

    assert avg_auction.allocations() == [
        auctions.Allocation(goods=0, transfer=4.5-1+0.25+0.625),
        auctions.Allocation(goods=3, transfer=-5-0.25+0.25+0.625),
    ]
    assert avg_auction.total_surplus() == 0 + 10

