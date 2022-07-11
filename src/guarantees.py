import typing as tp

from src import utility


def fixed_partition_guarantee(partition: tp.List, ut: utility.Utility):
    n = len(partition)
    return sum([ut(p) for p in partition]) / n
