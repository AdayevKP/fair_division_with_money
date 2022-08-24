import itertools as it
import os
import typing as tp

import pandas as pd


class Experiment:
    result_path = ''
    columns: tp.Set[str] = set()
    calculation_params: tp.Dict[str, tp.List[tp.Any]] = {}

    def __init__(self):
        self._data = []

    def set_params(self, new_prams: tp.Dict[str, tp.List[tp.Any]]):
        assert set(new_prams).issubset(set(self.calculation_params))
        self.calculation_params.update(new_prams)

    @staticmethod
    def _calc_normalized(
            funcs: tp.Dict[str, tp.Callable], norm_func, *args, **kwargs
    ) -> tp.Dict[str, tp.Any]:
        return {
               name: f(*args, **kwargs)/norm_func(*args, **kwargs)
               for name, f in funcs.items()
           }

    def calculate(self, **kwargs) -> tp.List[tp.Dict]:
        raise NotImplementedError

    def run_and_save(self, experiments_number):
        if os.path.exists(self.result_path):
            os.remove(self.result_path)

        self._data.clear()

        for _ in range(experiments_number):
            self._run_experiment()

        result_data = pd.DataFrame(columns=self.columns, data=self._data)

        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        result_data.to_csv(self.result_path, index=False)
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
