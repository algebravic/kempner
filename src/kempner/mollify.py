"""
  Test of mollification by a bump function.
"""
from typing import Callable, List
from functools import partial, cache
import numpy as np
import mpmath as mp
import scipy.integrate

def bump_function(arg: mp.mpf) -> mp.mpf:

    return mp.exp(-1/(1 - arg ** 2)) if mp.fabs(arg) < 1 else mp.mpf(0)

@cache
def normalization() -> mp.mpf:

    return 1 / mp.quadgl(bump_function, [-1, 1])

# Note that the bump_function is even, so all Fourier Coefficients
# Are real.
def moll_fourier_coeff(arg: mp.mpf) -> mp.mpf:
    """
      Fourier transform of the normalized bump function.
      Since it's even it's

      2 * integral[0, infty] f(y) cos(2 pi x y) dy

      Since f is normalized the fourier transform at 0 is 1.
    """
    if arg == 0:
        return mp.mpf(1)
    mult = normalization()
    cbump = lambda _: mult * bump_function(_) * mp.cos(2 * mp.pi * arg * _)
    # re_part = 2 * mp.quadosc(cbump, [0, mp.inf], period = 1 / arg)
    re_part = 2 * mp.quadgl(cbump, [0,1])
    # re_part =  mult * scipy.integrate.quad(bump_function,
    #     -1, 1, weight='cos',
    #     wvar = - 2 * np.pi * arg)[0]
    # im_part =  mult * scipy.integrate.quad(bump_function,
    #     -1, 1, weight='sin',
    #     wvar = - 2 * np.pi * arg)[0]
    return re_part

