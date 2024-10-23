"""
  Generate Fourier coefficients for the modified Kempner problem.
"""
from typing import Callable
import numpy as np
import mpmath as mp

FLOAT = np.float64 | mp.mpf
COMPLEX = np.complex64 | mp.mpc

class Fourier:

    def __init__(self, use_numpy = True):

        self._base = np if use_numpy else mp
        self._pi = np.pi if use_numpy else mp.pi
        self._sinc = np.sinc if use_numpy else mp.sincpi
        self._exp = np.exp if use_numpy else mp.exp
        self._log = np.log if use_numpy else mp.log

    def scaled_sinc(self, start: FLOAT, end: FLOAT) -> Callable[[FLOAT],
                                                                COMPLEX]:
        """
        if start < end < start + 1, we produce the fourier transform
        of the periodization mod 1 of the characteristic function, I,  of the
        interval [start, end].
          
          Note that F(I)(y) = int_{-\infty}^\infty I(x) e(-ixy)
          where e(x) = exp(2 pi i x).
          
          Note that np.sinc = sin(pi*x)/(pi * x)
          but mp.sinc = sin(x) / x
          """
        if not ((start < end) and (end < start + 1)):
            raise ValueError(f"({start}, {end}) invalid interval")

        dif = (end - start)
        avg = 0.5 * (end + start)

        return lambda _: (dif *
                          self._exp(2j * self._pi * avg * _)
                          * self._sinc(dif * _))
    def interval_sinc(self, arg: int, eps: FLOAT) -> Callable[[FLOAT], COMPLEX]:

        start = self._log(arg) / self._log(10)
        end = self._log(arg + 1) / self._log(10)

        return self.scaled_sinc(start + eps, end - eps)
    
        
        
