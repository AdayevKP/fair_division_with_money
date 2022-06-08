from src.experiments.identical_goods.compare_guarantees import calc_guarantees_and_save_results
from src.experiments.identical_goods.bargaining_gap import check_bargaining_gap


def run():
    calc_guarantees_and_save_results()
    check_bargaining_gap()


if __name__ == '__main__':
    run()
