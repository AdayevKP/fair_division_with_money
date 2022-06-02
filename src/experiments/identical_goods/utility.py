import numpy as np
import operator as op


fst = op.itemgetter(0)
snd = op.itemgetter(1)


def gen_utility_profiles(samples: np.ndarray):
    samples_with_zero = np.insert(samples, 0, 0)
    sorted_inc = np.sort(samples_with_zero)
    sorted_dec = sorted_inc[::-1]
    return np.cumsum(samples), np.cumsum(sorted_inc), np.cumsum(sorted_dec)


def gen_composite_utility_profiles(samples_g, samples_d, epsilon, lmbd):
    composite_profile = np.array([
        lmbd*((1-epsilon)*d + epsilon*g)
        for g, d in zip(samples_g, samples_d)
    ])
    return gen_utility_profiles(composite_profile)
