from src.experiments.identical_goods.compare_guarantees import compare_guarantees
from src.experiments.identical_goods.bargaining_gap import check_bargaining_gap


def run():
    compare_guarantees()
    check_bargaining_gap()


if __name__ == '__main__':
    run()
