import pytest

from src import auctions
from src.experiments.identical_goods import auctions_safe_play as ig_auctions
from src.experiments.identical_goods import utility as ut


@pytest.fixture(scope='function')
def utility():
    def make(utilities):
        return ut.IdenticalGoodsUtility(utilities)

    yield make


class TestDC1:
    def test1(self, utility):
        agent1 = [0, 1, 4, 9]
        agent2 = [0, 2, 7, 10]

        dc1_auction = ig_auctions.DivideAndChoose1(
            agents_utilities=[utility(agent1), utility(agent2)],
            goods_num=3
        )

        """
        step 1: 
            bids: maxmin - minmax
            1: (0+9)/2  - 5/2 = 2 -> win
            2: 10/2 - 9/2 = 0.5
            
        step 2: 
            agent1 pays 1 to agent2
            choose partition maximizing utility -> [0, 3]
        
        step 3: pi-auction
            agent1: u(0) = 0, u(3) = 9
                    t11 = 4.5   t12 = -4.5
                    
            agent2: u(0) = 0, u(3) = 10
                    t12 = 5   t22 = -5
                    
            sum[t12, t21] = 0.5
            sum[t11, t22] = -0.5 -> choose this allocation
            slack = -0.25
        """

        assert dc1_auction.allocations() == [
            auctions.Allocation(goods=0, transfer=4.5 + 0.25 - 1),
            auctions.Allocation(goods=3, transfer=-5 + 0.25 + 1),
        ]
        assert dc1_auction.total_surplus() == 0 + 10

    def test2(self, utility):
        agent1 = [0, 5, 7, 8]
        agent2 = [0, 2, 3, 9]

        dc1_auction = ig_auctions.DivideAndChoose1(
            agents_utilities=[utility(agent1), utility(agent2)],
            goods_num=3
        )

        assert dc1_auction.allocations() == [
            auctions.Allocation(goods=2, transfer=-1-1+0.25),
            auctions.Allocation(goods=1, transfer=1+0.5+0.25),
        ]
        assert dc1_auction.total_surplus() == 2 + 7


class TestDc2:
    def test1(self, utility):
        agent1 = [0, 1, 4, 9]
        agent2 = [0, 2, 9, 10]

        """
        step 1: 
            choose partitions:
            1: (0,3)
            2: (1,2)

        step 2 averaging auction: 
            agent1:
                meanG = ((1 + 4)/2 + 9/2)/2 = 3.5
                G11 = (1 + 4)/2 = 1.5; t11 = 3 - 1.5 = 1
                G12 = 9/2 = 4.5;       t12 = 3 - 4.5 = -1
                
            agent2:
                meanG = ((2 + 9)/2 + 10/2)/2 = 5.25
                G21 = (2 + 9)/2 = 5.5; t21 = 5.25 - 5.5 = - 0.25
                G22 = 10/2 = 5;        t22 = 5.25 - 5   = 0.25
                
            max = G12 + G22 -> choose (0, 3) 
            slack = (t22 + t12)/2 = (0.25 - 1)/2 = -0.375
            
            pi auction
            agent1: u(0) = 0, u(3) = 9
                    t11 = 4.5   t12 = -4.5
                    
            agent2: u(0) = 0, u(3) = 10
                    t12 = 5    t22 = -5
                    
            sum[t12, t21] = 0.5
            sum[t11, t22] = -0.5 -> choose this allocation
            slack = -0.25
        """

        dc2_auction = ig_auctions.DivideAndChoose2(
            agents_utilities=[utility(agent1), utility(agent2)],
            goods_num=3
        )

        assert dc2_auction.allocations() == [
            auctions.Allocation(goods=0, transfer=4.5+0.25-1+0.375),
            auctions.Allocation(goods=3, transfer=-5+0.25+0.25+0.375),
        ]
        assert dc2_auction.total_surplus() == 0 + 10

    def test2(self, utility):
        agent1 = [0, 5, 7, 8]
        agent2 = [0, 2, 3, 9]

        dc2_auction = ig_auctions.DivideAndChoose2(
            agents_utilities=[utility(agent1), utility(agent2)],
            goods_num=3
        )

        assert dc2_auction.allocations() == [
            auctions.Allocation(goods=2, transfer=-1+0-1+0.25),
            auctions.Allocation(goods=1, transfer=1+0+0.5+0.25),
        ]
        assert dc2_auction.total_surplus() == 7 + 2
