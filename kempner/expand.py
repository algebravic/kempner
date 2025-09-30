from typing import Tuple, List
from functools import cache, partial
from joblib import Memory
import numpy as np
import mpmath as mp
#from pyprimesieve import primes
from sympy import primerange
from .mollify import moll_fourier_coeff
from .filters import get_filter
from .fourier import Fourier

FLOAT = np.float64 | mp.mpf

memory = Memory('primecache.dat')

@cache
def expand_table(limit: int,
                 from_base: int, to_base: int) -> np.ndarray:
    """
      Calculate the expanded table.
      V(n) = the integer obtained by using the digits of n
       in the base, from_base, as digits in to_base
      Tabulate all values of V(n) for 0 <=n < from_base ** limit

      Since integers in numpy are limited to 64 bits, this
      is only valid when to_base ** limit < 2 ** 64.
      Input:
       limit: int - the number of digits in the from_base
       from_base: int - the base of integers of the inputs
       to_base: int - the base of integers of the outputs
    """
    top = from_base ** limit
    table = np.empty(top, dtype=np.int64)
    table[: from_base] = np.arange(from_base, dtype=np.int64)
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
        res += mult * int(table[narg % tsize])
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
    expon = mp.log(to_base) / mp.log(from_base)
    my_primes = np.array(list(primerange(start, end)), dtype=np.uint64)
    prime_sum = (my_primes ** (- expon)).sum()
    xpn_sum = (1 / vxpnd(my_primes)).sum()

    return prime_sum, xpn_sum

def prime_zeta_values(from_base: int, to_base: int,
                      lim: int) -> List[mp.mpf]:

    expon = mp.log(to_base) / mp.log(from_base)
    return [mp.primezeta(expon - (2j * np.pi * _)
                         / mp.log(from_base))
            for _ in range(lim)]

def truncated_zeta_values(lbound: int, lim: int,
                          expon: FLOAT,
                          iexpon: FLOAT) -> np.ndarray:

    mynums = np.arange(1, lbound, dtype=int)
    ipzeta = np.array([(mynums ** (- expon + 1j * iexpon * _)).sum()
        for _ in range(lim)])
    pzeta = np.array([mp.zeta(expon - 1j * iexpon * _)
        for _ in range(lim)])

    return pzeta - ipzeta

def truncated_primezeta_values(lbound: int, lim: int,
                               expon: FLOAT,
                               iexpon: FLOAT,
                               verbose: int = 0) -> np.ndarray:

    # Calculate initial parts of prime zeta
    
    # Note mp.fp does not compute primezeta for complex arguments
    my_primes = np.array(list(primerange(1, lbound)), dtype=np.int64)
    ipzeta = np.array([(my_primes ** (- expon + 1j * iexpon * _)).sum()
        for _ in range(lim)])
    pzeta = np.array([mp.primezeta(expon - 1j * iexpon * _)
        for _ in range(lim)])

    return pzeta - ipzeta

def digit_bounds(from_base: int, to_base: int, deg: int,
                 table: np.ndarray) -> np.ndarray:
    mpf = np.vectorize(lambda _: mp.mpf(int(_)))
    lbound = from_base ** deg
    ubound = lbound * from_base
    expon = mp.log(to_base)/mp.log(from_base)
    leading = np.arange(lbound, ubound)
    twiddle = mp.mpf(from_base - 1) / (to_base - 1)
    lower =  leading ** expon / (mpf(table[leading]) + twiddle)
    upper =  (leading + 1) ** expon / mpf(table[leading])
    return np.vstack([lower, upper])

def approximation(from_base: int, to_base: int,
                  deg: int, # number of leading digits
                  lim: int, # number of terms in the fourier series
                  use_primes: bool = True, # Calculate the prime series
                  filter_name: str = '', # Fourier filter to use
                  filter_params = (), # Parameters for the filter
                  use_mollifier: bool = False, # use the mollifier instead
                  use_fejer: bool = False,
                  verbose: int = 0, # verbosity level
                  scale: mp.mpf = mp.mpf(1) / 100):

    r""" Approximate the tail with deg digits in from_base.
      deg: digit size of leading digits
      lim: the number of fourier coefficients to use
      scale: the multiplier to find the epsilon to perturb by
    """

    # Find smallest interval size
    # log(b^(deg+1)) - log(b^(deg + 1) - 1) =
    # - log(1 - b^(-(deg+1)))

    # Get the V(n) table
    table = expand_table(deg + 1, from_base, to_base)
    lbound = from_base ** deg
    ubound = from_base ** (deg + 1)

    expon = mp.log(to_base) / mp.log(from_base)
    iexpon = 2 * mp.pi / mp.log(from_base)
    # leading = np.arange(lbound, ubound)
    mpf = np.vectorize(lambda _: mp.mpf(int(_)))
    if use_primes:
        my_primes = np.array(list(primerange(1, lbound)), dtype=np.int64)
        initial = (1 / mpf(table[my_primes])).sum()
        if verbose > 0:
            print(f"Primes: initial={initial}")
        dzeta = truncated_primezeta_values(lbound, lim,
            expon, iexpon, verbose=verbose)
    else:
        initial = (1 / mpf(table[np.arange(1, lbound)])).sum()
        if verbose > 0:
            print(f"All integers: initial={initial}")
        dzeta = truncated_zeta_values(lbound, lim,
            expon, iexpon, verbose=verbose)
    # Find upper and lower found coefficients

    # First to the unmollified
    # The fourier coefficients of the characteristic functions
    # of the intervals.
    # Calculate any filters/mollifiers to use
    filtered = False
    if use_mollifier:
        filtered = True
        smallest = - mp.log(1 - mp.mpf(from_base) ** (-(deg + 1)))
        eps = scale * 0.5 * smallest
        if verbose > 0:
            print(f"smallest = {smallest}, eps = {eps}")
        mfour = np.vectorize(moll_fourier_coeff)
        # Get mollifier fourier coefficients
        filt_coeffs = mfour(np.arange(lim, dtype=int) * eps)
        if verbose > 1:
            print(f"filt_coeffs = {filt_coeffs}")
        filtered = True
    else:
        eps = 0.0
        if filter_name != '':
            func = np.vectorize(get_filter(filter_name, *filter_params))
            filt_coeffs = func(np.arange(lim, dtype=int) / mp.mpf(lim))
            filtered = True

    # Get the upper lower bounds from the selected leading digits
    # We may need to modify this if we use variable digit lengths
    bounds = digit_bounds(from_base, to_base, deg, table)
    mplog = np.vectorize(mp.log)
    logs = mplog(np.arange(lbound, ubound + 1)) / mp.log(from_base)

    # The arrays hold mpmath data
    coeffs = np.empty((2, lim), dtype=np.dtype('O'))
    # 0: lower bound, 1: upper bound
    sums = logs[1:] + logs[:-1] - 2 * deg
    mpexp = np.vectorize(mp.exp)
    # This is where we may run into storage difficulties.
    phase = mpexp(- 1j * mp.pi * np.outer(sums, np.arange(lim, dtype=int)))
    if verbose > 1:
        print(f"phase.shape = {phase.shape}")
    for ind, delta in enumerate([eps, 0]):
        diffs = logs[1:] - logs[:-1] - 2 * eps
        # np.sinc(x) = sin(pi * x) / (pi * x)
        mpsinc = np.vectorize(lambda _: mp.sinc(mp.pi * _))
        main = mpsinc(np.outer(diffs, np.arange(lim)))
        fcoeffs = diffs.reshape(-1,1) * phase * main
        if filtered:
            fcoeffs *= filt_coeffs
        # multiply the corresponding and sum up
        coeffs[ind] = (bounds[ind].reshape(-1,1) * fcoeffs).sum(axis=0)
    # Calculate the final summations
    lprod = np.array([_.real for _ in (coeffs[0] * dzeta)])
    lprod[1:] *= 2
    uprod = np.array([_.real for _ in (coeffs[1] * dzeta)])
    uprod[1:] *= 2
    if use_fejer:
        lprod *= np.arange(lim, 0, -1) / mp.mpf(lim)
        uprod *= np.arange(lim, 0, -1) / mp.mpf(lim)
    # initial coefficient is real
    # The others are complex, multiply by 2 for positive/negative
    ltail = lprod.sum()
    utail = uprod.sum()
    return (initial + ltail, initial + utail)
