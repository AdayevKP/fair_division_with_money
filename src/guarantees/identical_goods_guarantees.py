import numpy as np
import typing as tp
import itertools as it


from src.guarantees import guarantees


def guarantee_sb_2(utility: np.ndarray):
    u_size = utility.size
    m = utility.size - 1
    u = utility

    x = np.zeros((u_size, u_size))
    for k in range(u_size):
        for l in range(u_size):
            if l + k >= u_size:
                continue
            x[k][l] = m * (u[l + k] - u[l]) / (m + k)

    k_max, l_max = np.unravel_index(np.argmax(x, axis=None), x.shape)
    return (l_max + k_max) * u[l_max] / (m + k_max) + (m - l_max) * u[l_max + k_max] / (m + k_max)


def guarantee_sb(utility: np.ndarray, n=2):
    if n == 2:
        return guarantee_sb_2(utility)

    raise ValueError(
        'sell and buy rule guarantee not implemented for more than 2 goods'
    )


def guarantee_dc(utility: np.ndarray, n=2):
    if n == 2:
        return max_min(utility) / n + (n - 1) * min_max(utility) / n

    raise ValueError(
        'divide and choose rule guarantee not implemented for more than 2 goods'
    )


def max_min(utility: np.ndarray, n=2):
    return guarantees.max_min([utility] * n)


def min_max(utility: np.ndarray, n=2):
    return guarantees.min_max([utility] * n)
