"""
  Summing of a Fourier series.
"""

from typing import Callable
import numpy as np
import mpmath as mp

FLOAT = np.float64 | mp.mpf
COMPLEX = mp.mpc | np.complex64

def fourier_series(coeffs: np.ndarray) -> Callable[[FLOAT], FLOAT]:

    def fcn(arg: FLOAT) -> FLOAT:

        return mp.fsum([coeffs[0].real] + [2 * (mp.exp(2j * mp.pi * ind * arg) * val).real
                                           for ind, val in enumerate(coeffs[1:], start=1)])
    return fcn

def muval(kval: int) -> COMPLEX:
    return 1 + (2j * mp.pi * kval) / mp.log(10)

def chi(kval: int) -> COMPLEX:
    return muval(kval) + mp.log(11) / mp.log(10)

def exact_fourier(kval: int) -> COMPLEX:

    if kval == 0:
        return (
            (1 /(9 * (mp.log(11) + mp.log(10)))) *
            mp.fsum((1 / mp.mpf(_) for _ in range(1, 10))))

    muv = muval(kval)
    return ((1/mp.log(10)) * (1 / chi(kval)) *
            (mp.zeta(muv)/(muv - 1)) *
            mp.fsum((_ ** (- muv) for _ in range(1, 10)))/
            mp.fsum((_ ** (1 - muv) for _ in range(1, 10))))
