from typing import Tuple
import resource
import numpy as np
import mpmath as mp
#from pyprimesieve import primes
from sympy import primerange
from .mollify import moll_fourier_coeff
from .filters import get_filter

FLOAT = np.float64 | mp.mpf

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

def truncated_zeta_values(lbound: int, nterms: int,
                          expon: FLOAT,
                          iexpon: FLOAT,
                          verbose: int = 0) -> np.ndarray:

    mynums = np.arange(1, lbound, dtype=int)
    # Can use the Euler-MacLaurin formula for more efficient calculation
    ipzeta = np.array([(mynums ** (- expon + 1j * iexpon * _)).sum()
        for _ in range(nterms)])
    pzeta = np.array([mp.zeta(expon - 1j * iexpon * _)
        for _ in range(nterms)])

    return pzeta - ipzeta

def truncated_primezeta_values(lbound: int, nterms: int,
                               expon: FLOAT,
                               iexpon: FLOAT,
                               verbose: int = 0) -> np.ndarray:

    # Calculate initial parts of prime zeta
    
    # Note mp.fp does not compute primezeta for complex arguments
    my_primes = np.array(list(primerange(1, lbound)), dtype=np.int64)
    # Can use the LMO algorithm for more efficient calculations
    ipzeta = np.array([(my_primes ** (- expon + 1j * iexpon * _)).sum()
        for _ in range(nterms)])
    pzeta = np.array([mp.primezeta(expon - 1j * iexpon * _)
        for _ in range(nterms)])

    return pzeta - ipzeta

def initial_truncated(from_base: int, to_base: int, deg: int, nterms: int,
                      extra: int = 0,
                      use_primes: bool = True) -> Tuple[mp.mpf, np.ndarray]:
    table = expand_table(deg + max(1,extra), from_base, to_base)
    bbound = from_base ** (deg + extra)
    expon = mp.log(to_base) / mp.log(from_base)
    iexpon = 2 * mp.pi / mp.log(from_base)
    # leading = np.arange(lbound, ubound)
    mpf = np.vectorize(lambda _: mp.mpf(int(_)))
    if use_primes:
        my_primes = np.array(list(primerange(1, bbound)), dtype=np.int64)
        initial = (1 / mpf(table[my_primes])).sum()
        dzeta = truncated_primezeta_values(bbound, nterms,
            expon, iexpon)
    else:
        initial = (1 / mpf(table[np.arange(1, bbound)])).sum()
        dzeta = truncated_zeta_values(bbound, nterms,
            expon, iexpon)
    return initial, dzeta
    

def digit_bounds(from_base: int, to_base: int, deg: int) -> np.ndarray:

    mpf = np.vectorize(lambda _: mp.mpf(int(_)))
    lbound = from_base ** deg
    ubound = lbound * from_base
    expon = mp.log(to_base)/mp.log(from_base)
    table = expand_table(deg + 1, from_base, to_base)
    leading = np.arange(lbound, ubound)
    twiddle = mp.mpf(from_base - 1) / (to_base - 1)
    lower =  leading ** expon / (mpf(table[leading]) + twiddle)
    upper =  (leading + 1) ** expon / mpf(table[leading])
    return np.vstack([lower, upper])

def get_filter_coeffs(filter_name: str, filter_params: Tuple,
                      scale: FLOAT,
                      small: FLOAT,
                      nterms: int) -> Tuple[FLOAT, np.ndarray] | None:
    """
      Get the filter coefficients for the filter_name or None if there's an error
    """
    eps = 0.0
    if filter_name == 'mollifier':
        smallest = - mp.log(1 - small)
        eps = scale * 0.5 * smallest
        mfour = np.vectorize(moll_fourier_coeff)
        # Get mollifier fourier coefficients
        return eps, mfour(np.arange(nterms, dtype=int) * eps)
    elif filter_name != '':
        try:
            func = np.vectorize(get_filter(filter_name, *filter_params))
            return mp.mpf(0), func(np.arange(nterms, dtype=int) / mp.mpf(nterms))
        except ValueError as msg:
            print(msg)
            print("Not using a filter")
        
    return mp.mpf(0), None

def mem_usage(msg: str):
    print(f"{msg}: memory = {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss:_}")

def approximation(from_base: int, to_base: int,
                  deg: int, # number of leading digits
                  nterms: int, # number of terms in the fourier series
                  extra: int = 0, # extra number of places for the tail estimate
                  use_primes: bool = True, # Calculate the prime series
                  filter_name: str = '', # Fourier filter to use
                  filter_params = (), # Parameters for the filter
                  verbose: int = 0, # verbosity level
                  scale: mp.mpf = mp.mpf(1) / 100):

    r"""
    Approximate the tail with deg digits in from_base.
    Inputs:
       Required:
        from_base: base of input integers.
        to_base: base of output integers.
        deg: digit size of leading digits
        nterms: the number of Fourier coefficients to use
       Optional:
        use_primes: bool -- calculated the prime series, default: True
        filter_name: str - name of filter to use, default: ''
        filter_params: Tuple - parameters of the filters, default: ()
        verbose: int - level of verbosity, default = 0
        scale: mp.mpf scale multiplier, default 0.01
    """
    # Find smallest interval size
    # log(b^(deg+1)) - log(b^(deg + 1) - 1) =
    # - log(1 - b^(-(deg+1)))

    # Get the V(n) table
    initial, dzeta = initial_truncated(from_base, to_base, deg, nterms,
        extra = extra, use_primes = use_primes)

    if verbose > 0:
        print(f"{'Primes' if use_primes else 'All integers'}: initial={initial}.")

    if verbose > 0:
        mem_usage("After initial")

    # Find upper and lower found coefficients
    # Calculate filter/mollification coefficients
    # First to the unmollified
    # The fourier coefficients of the characteristic functions
    # of the intervals.
    # Calculate any filters/mollifiers to use
    # Get the upper lower bounds from the selected leading digits
    # We may need to modify this if we use variable digit lengths

    eps, filt_coeffs = get_filter_coeffs(filter_name, filter_params,
        scale,
        mp.mpf(from_base) ** (- (deg + 1)), nterms)

    if verbose > 0:
        mem_usage("After filter")
    if verbose > 1 and filt_coeffs is not None:
        print(f"filt_coeffs = {filt_coeffs}")
    
    lbound = from_base ** deg
    ubound = from_base ** (deg + 1)

    bounds = digit_bounds(from_base, to_base, deg)
    mplog = np.vectorize(mp.log)
    logs = mplog(np.arange(lbound, ubound + 1)) / mp.log(from_base)

    if verbose > 0:
        mem_usage("After bounds")
    # The arrays hold mpmath data
    coeffs = np.empty((2, nterms), dtype=np.dtype('O'))
    # 0: lower bound, 1: upper bound
    sums = logs[1:] + logs[:-1] - 2 * deg
    if verbose > 0:
        print(f"sums.shape = {sums.shape}")
    mpexp = np.vectorize(mp.exp)
    # This is where we may run into storage difficulties.
    phase = mpexp(- 1j * mp.pi * np.outer(sums, np.arange(nterms, dtype=int)))
    if verbose > 0:
        mem_usage("After phase")
    if verbose > 1:
        print(f"phase.shape = {phase.shape}")
    for ind, delta in enumerate([eps, -eps]):
        diffs = logs[1:] - logs[:-1] - 2 * eps
        # np.sinc(x) = sin(pi * x) / (pi * x)
        mpsinc = np.vectorize(lambda _: mp.sinc(mp.pi * _))
        main = mpsinc(np.outer(diffs, np.arange(nterms)))
        fourier_coeffs = diffs.reshape(-1,1) * phase * main
        if filt_coeffs is not None:
            fourier_coeffs *= filt_coeffs
        # multiply the corresponding and sum up
        coeffs[ind] = (bounds[ind].reshape(-1,1) * fourier_coeffs).sum(axis=0)
    # Calculate the final summations
    lprod = np.array([_.real for _ in (coeffs[0] * dzeta)])
    lprod[1:] *= 2
    uprod = np.array([_.real for _ in (coeffs[1] * dzeta)])
    uprod[1:] *= 2
    # initial coefficient is real
    # The others are complex, multiply by 2 for positive/negative
    ltail = lprod.sum()
    utail = uprod.sum()
    return (initial + ltail, initial + utail)
