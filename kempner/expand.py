from typing import Tuple, List
from functools import cache, partial
from joblib import Memory
import numpy as np
import mpmath as mp
from pyprimesieve import primes
from .mollify import moll_fourier_coeff
from .filters import get_filter

FLOAT = np.float64 | mp.mpf

memory = Memory('primecache.dat')

@cache
def expand_table(limit: int,
                 from_base: int, to_base: int) -> np.ndarray:
    """
      Calculate the expanded table.
    """
    top = from_base ** limit
    table = np.empty(top, dtype=np.int64)
    table[: from_base] = np.arange(from_base)
    fence = from_base
    ofence = to_base
    # We've filled in out[: from_base ** ind]
    for ind in range(limit - 1):
        for indx in range(1, from_base):
            bot = indx * fence
            addend = indx * ofence
            table[bot: bot + fence] = table[: fence] + addend
        fence *= from_base
        ofence *= to_base
    return table

def expand(from_base: int, to_base: int, limit: int, arg: int) -> int:
    "Rewrite base from_base digits in base to_base."

    res = 0
    narg = arg
    mult = 1
    table = expand_table(limit, from_base, to_base)
    tsize = from_base ** limit
    xsize = to_base ** limit
    while narg >= tsize:
        res += mult * table[narg % tsize]
        narg //= tsize
        mult *= xsize
    return res + mult * int(table[narg])

@memory.cache(verbose=0)
def get_chunk(from_base: int, to_base: int,  limit: int,
              start: int, end: int) -> Tuple[np.float64, np.float64]:
    """
      Get next partial sums of 1/p and 1/V(p)
      """
    vxpnd = np.vectorize(partial(expand, from_base, to_base, limit))
    expon = np.log(to_base) / np.log(from_base)
    my_primes = np.array(primes(start, end), dtype=np.int64)
    prime_sum = (my_primes ** (- expon)).sum()
    xpn_sum = (1 / vxpnd(my_primes)).sum()

    return prime_sum, xpn_sum

def prime_zeta_values(from_base: int, to_base: int,
                      lim: int) -> np.ndarray:

    expon = np.log(to_base) / np.log(from_base)
    return [mp.primezeta(expon - (2j * np.pi * _)
                         / np.log(from_base))
            for _ in range(lim)]

def truncated_zeta_values(lbound: int, lim: int,
                          expon: FLOAT,
                          iexpon: FLOAT) -> np.ndarray:

    mynums = np.arange(1, lbound)
    ipzeta = np.array([(mynums ** (- expon + iexpon * _)).sum()
        for _ in range(lim)], dtype=np.complex64)
    pzeta = np.array([mp.zeta(expon - iexpon * _)
        for _ in range(lim)])

    return pzeta - ipzeta

def truncated_primezeta_values(lbound: int, lim: int,
                               my_primes: np.ndarray,
                               expon: FLOAT,
                               iexpon: FLOAT) -> np.ndarray:

    # Calculate initial parts of prime zeta
    
    # Note mp.fp does not compute primezeta for complex arguments
    ipzeta = np.array([(my_primes ** (- expon + iexpon * _)).sum()
        for _ in range(lim)], dtype=np.complex64)
    pzeta = np.array([mp.primezeta(expon - iexpon * _)
        for _ in range(lim)])

    return pzeta - ipzeta

def digit_bounds(from_base: int, to_base: int, deg: int,
                 table: np.ndarray) -> np.ndarray:
    lbound = from_base ** deg
    ubound = lbound * from_base
    expon = np.log(to_base)/np.log(from_base)
    leading = np.arange(lbound, ubound)
    twiddle = (from_base - 1) / (to_base - 1)
    lower =  leading ** expon / (table[leading] + twiddle)
    upper =  (leading + 1) ** expon / table[leading]
    return np.vstack([lower, upper])

def approximation(from_base: int, to_base: int, deg: int,
                  lim: int,
                  use_primes: bool = True,
                  filter_name: str = '',
                  filter_params = (),
                  use_mollifier: bool = False,
                  use_fejer: bool = False,
                  verbose: int = 0,
                  scale: float = .01):

    """ Approximate the tail with deg digits in from_base.
      deg: digit size of leading digits
      lim: the number of fourier coefficients to use
      scale: the multiplier to find the epsilon to perturb by
      """

    # Find smallest interval size
    # log(b^(deg+1)) - log(b^(deg + 1) - 1) =
    # - log(1 - b^(-(deg+1)))

    table = expand_table(deg + 1, from_base, to_base)
    lbound = from_base ** deg
    ubound = from_base ** (deg + 1)

    expon = np.log(to_base)/np.log(from_base)
    iexpon = 2j * np.pi / np.log(from_base)
    leading = np.arange(lbound, ubound)
    if use_primes:
        my_primes = np.array(primes(1, lbound), dtype=np.int64)
        initial = (1/table[my_primes]).sum()
        dzeta = truncated_primezeta_values(lbound, lim,
            my_primes, expon, iexpon)
    else:
        initial = (1 / table[np.arange(1, lbound)]).sum()
        dzeta = truncated_zeta_values(lbound, lim,
            expon, iexpon)
    # Find upper and lower found coefficients

    # First to the unmollified
    # The fourier coefficients of the characteristic functions
    # of the intervals.
    filtered = False
    if use_mollifier:
        filtered = True
        smallest = - np.log(1 - from_base ** (-(deg + 1)))
        eps = scale * 0.5 * smallest
        if verbose > 0:
            print(f"smallest = {smallest}, eps = {eps}")
        mfour = np.vectorize(moll_fourier_coeff)
        # Get mollifier fourier coefficients
        filt_coeffs = mfour(np.arange(lim)/eps)
        if verbose > 1:
            print(f"log(mcoeffs) = {np.log(mcoeffs)}")
    else:
        eps = 0.0
        if filter_name != '':
            func = get_filter(filter_name, *filter_params)
            filt_coeffs = func(np.linspace(0,1,lim))
            filtered = True

    bounds = digit_bounds(from_base, to_base, deg, table)
    logs = np.log(np.arange(lbound, ubound + 1)) / np.log(from_base)

    coeffs = np.empty((2, lim), dtype=np.complex64)
    # 0: lower bound, 1: upper bound
    sums = logs[1:] + logs[:-1] - 2 * deg
    phase = np.exp(- 1j * np.pi * np.outer(sums, np.arange(lim)))
    for ind, delta in enumerate([eps, -eps]):
        diffs = logs[1:] - logs[:-1] - 2 * eps
    # np.sinc(x) = sin(pi * x) / (pi * x)
        main = np.sinc(np.outer(diffs, np.arange(lim)))
        fcoeffs = diffs.reshape(-1,1) * phase * main
        if filtered:
            fcoeffs *= filt_coeffs
        # multiply the correspoknding and sum up
        coeffs[ind] = (bounds[ind].reshape(-1,1) * fcoeffs).sum(axis=0)
    # Calculate the final summations
    lprod = np.array([_.real for _ in (coeffs[0] * dzeta)])
    lprod[1:] *= 2
    uprod = np.array([_.real for _ in (coeffs[1] * dzeta)])
    uprod[1:] *= 2
    if use_fejer:
        lprod *= np.arange(lim, 0, -1) / lim
        uprod *= np.arange(lim, 0, -1) / lim
    # initial coefficient is real
    # The others are complex, multiply by 2 for positive/negative
    ltail = lprod.sum()
    utail = uprod.sum()
    return (initial + ltail, initial + utail)
