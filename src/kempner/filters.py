"""
  Various Fourier filters.
"""
from typing import Callable
from functools import partial
import numpy as np
from scipy.special import betainc
import mpmath as mp

FLOAT = np.float64

def fejer_filter(arg: np.ndarray) -> np.ndarray:

    return 1 - arg

def lanczos_filter(arg: np.ndarray) -> np.ndarray:

    mpsinc = np.vectorize(lambda _: mp.sinc(mp.pi * _))
    return mpsinc(arg)

def raised_cosine_filter(arg: np.ndarray) -> np.ndarray:

    mpcos = np.vectorize(lambda _: mp.cos(mp.pi * _))
    return (1 + mpcos(arg)) / 2

def sharpened_raised_cosine_filter(arg: np.ndarray) -> np.ndarray:

    rcos = raised_cosine_filter(arg)

    return rcos ** 4 * (35 - 84 * rcos + 70 * rcos ** 2
                        - 20 * rcos ** 3)

def exp_filter(alpha: mp.mpf, order: int, arg: np.ndarray) -> np.ndarray:

    mpexp = np.vectorize(lambda _: mp.exp(alpha * _ ** order))
    return mpexp(arg)

def daubechies_filter(order: int, arg: np.ndarray) -> np.ndarray:
    mpbetainc = np.vectorize(lambda _: mp.betainc(order, order, x1 = 0, x2 = _, regularized=True))
    return 1 - mpbetainc(arg)

FILTERS = {'fejer': (fejer_filter, ()),
           'lanczos': (lanczos_filter, ()),
           'raised_cosine': (raised_cosine_filter, ()),
           'sharpened_raised_cosine': (sharpened_raised_cosine_filter,
                                       ()),
           'exp': (exp_filter,(FLOAT, int)),
           'daubechies': (daubechies_filter,(int,))}

def get_filter(name: str, *params) -> Callable[[np.ndarray], np.ndarray]:
    if name not in FILTERS:
        raise ValueError(f"'{name}' not a known filter.")

    func, types = FILTERS[name]

    if not (len(params) == len(types)
            and all((isinstance(_[0], _[1])
                     for _ in zip(params, types)))):
        raise ValueError("mismatch in parameters")

    return partial(func, *params)
