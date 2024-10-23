"""
  Test of mollification by a bump function.
"""
from typing import Callable
from functools import partial
import numpy as np
import mpmath as mp

FLOAT = np.float64 | mp.mpf
COMPLEX = np.complex64 | mp.mpc

class Bump:

    def __init__(self, use_numpy = True):

        self._base = np if use_numpy else mp

    def bump(self, arg: FLOAT) -> FLOAT:
        " The function exp(-1/(1-x^2)) when |x| < 1, 0 otherwise."
        
        return (0.0 if self._base.abs(arg) >= 1
                else self._base.exp(-1/(1 - arg ** 2)))

    def bumpb(self, scale: FLOAT) -> Callable[[FLOAT], FLOAT]:

        return lambda _: (1 / scale) * self.bump(_ / scale)

    def bfourier(self, num: int, scale: FLOAT) -> Callable[[FLOAT], COMPLEX]:

        return lambda _: (1/ scale) * self.bump(_ / scale) * self._base.exp(-2j * self._base.pi * num * _)
