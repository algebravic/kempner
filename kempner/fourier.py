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
        self._sinc = np.sinc if use_numpy else np.vectorize(mp.sincpi)
        self._exp = np.exp if use_numpy else np.vectorize(mp.exp)
        self._log = np.log if use_numpy else np.vectorize(mp.log)
        self._mpf = np.float64 if use_numpy else mp.mpf
        self._mpc = np.complex64 if use_numpy else mp.mpc
        self._real = np.vectorize(lambda _: _.real)

    def coefficients(self, start: FLOAT, end: FLOAT) -> Callable[[FLOAT],
                                                                COMPLEX]:
        r"""
        if start < end < start + 1, we produce the fourier transform
        of the periodization mod 1 of the characteristic function, I,  of the
        interval [start, end].
          
          Note that F(I)(y) = int_{-\infty}^\infty I(x) e(-xy)
          where e(x) = exp(2 pi i x).
          
          Note that np.sinc = sin(pi*x)/(pi * x)
          but mp.sinc = sin(x) / x
        """
        if not ((start < end) and (end < start + 1)):
            raise ValueError(f"({start}, {end}) invalid interval")

        dif = (end - start)
        pls = (end + start)

        return np.vectorize(lambda _:
                            (dif * self._exp(- 1j * self._pi * pls * _)
                             * self._sinc(dif * _)))

    def interval_coeffs(self, arg: int,
                        base: int = 10, eps: FLOAT = mp.mpf(0)) -> Callable[[FLOAT], COMPLEX]:
        r"""
           Calculate fourier coefficients for the interval
           [log(a)/log(b) + epsilon, log(a+1)/log(b) - epsilon]
        """

        start = self._log(arg) / self._log(base)
        end = self._log(arg + 1) / self._log(base)

        return self.coefficients(start + eps, end - eps)

    def approx(self, start: FLOAT, end: FLOAT, prec: int, scale: int) -> np.ndarray:
        r"""
           return the sum of prec fourier coefficients for the interval
           function, using scale points in [0,1].

           g[k] = k-th fourier coefficient
           value[l] =  sum[k](exp(2 * pi * 1j * k * (1/l)) * g[k])
        """
        # Calculate evaluation points: equal spaced on [0,1]
        points = np.arange(scale) / self._mpf(scale)
        # The Fourier coefficient function
        func = self.coefficients(start, end)
        return  self._real(2 * self._exp(2 * self._pi * 1j
                                         * np.outer(points, np.arange(1, prec)))
                           @ func(np.arange(1, prec)) + func(0))
