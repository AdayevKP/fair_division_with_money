import itertools as it
import typing as tp

import pandas as pd


class Experiment:
    result_path = ''
    columns: tp.Set[str] = set()
    calculation_params: tp.Dict[str, tp.List[tp.Any]] = {}

    def __init__(self):
        self._data = []

    def calculate(self, **kwargs) -> tp.List[tp.Dict]:
        raise NotImplementedError

    def run_and_save(self, experiments_number):
        self._data.clear()

        for _ in range(experiments_number):
            self._run_experiment()

        result_data = pd.DataFrame(columns=self.columns, data=self._data)
        result_data.to_csv(self.result_path)
        self._data.clear()

    def _run_experiment(self):
        keys = list(self.calculation_params.keys())
        for row in it.product(*self.calculation_params.values()):
            calc_params = dict(zip(keys, row))
            new_rows = self.calculate(**calc_params)
            for r in new_rows:
                assert self.columns == set(r)
            self._data.extend(new_rows)

    @classmethod
    def load_experiment(cls, experiment_id=0) -> pd.DataFrame:
        data = pd.read_csv(cls.result_path)
        return data
