from typing import Callable, List
from functools import partial, cache
import numpy as np
import mpmath as mp
import scipy.integrate

FLOAT = np.float64 | mp.mpf
COMPLEX = np.complex64 | mp.mpc

FUN = Callable[[FLOAT], FLOAT]

class Bump:

    def __init__(self, use_numpy = True):

        self._use_numpy = use_numpy
        self._base = np if use_numpy else mp
        self._exp = np.exp if use_numpy else mp.exp
        self._pi = np.pi if use_numpy else mp.pi
        self._abs = np.abs if use_numpy else abs
        self._quad = (lambda a, b, c: scipy.integrate.quad(a,b,c, complex_func = True)[0]
            if use_numpy else mp.quad(a, [b, c]))
        self._total = self._quad(self.unbump, -np.inf, np.inf).real

    def unbump(self, arg: FLOAT) -> FLOAT:
        " The function exp(-1/(1-x^2)) when |x| < 1, 0 otherwise."
        
        return (0.0 if self._abs(arg) >= 1
                else self._exp(-1/(1 - arg ** 2)))

    def bump(self, arg: FLOAT) -> FLOAT:

        return self.unbump(arg) / self._total

    def bumpb(self, scale: FLOAT) -> Callable[[FLOAT], FLOAT]:

        return lambda _: (1 / scale) * self.bump(_ / scale)

    def afourier(self, scale: FLOAT, num: int) -> Callable[[FLOAT], COMPLEX]:
        
        return (lambda _: self.bumpb(scale)(_) *
                self._exp(-2j * self._pi * num * _))
    
    def fourier(self, fun: Callable[[FLOAT], COMPLEX], num: int) -> COMPLEX:

        return (mp.quad(
            lambda _: fun(_) * self._exp(-2j * self._pi * num * _),
            -1, 1) if not self._use_numpy
                else fcoeff(fun, num))
def prime_zeta_values(limit: int) -> List[FLOAT]:

    main = mp.log(11) / mp.log(10)
    return [mp.primezeta(main - (2j * mp.pi * _) / mp.log(10))
            for _ in range(limit)]
