"""
  General Utility functions
"""
from typing import List, Tuple
import numpy as np
import mpmath as mp

def leading_digits(limit: int, deg: int, from_base: int = 10, to_base: int = 11) -> np.ndarray:
    """
      The leading digits of width deg, or all numbers whose length is in range(deg, limit)
    """
    start = from_base ** (deg - 1)
    end = from_base ** limit
    # Apparently there's a bug in np.log
    # I found that np.log(1000)/np.log(10) is 2.9999 and not 3.0000
    eps = 1e-9
    divisor = np.floor(np.log(np.arange(start, end, dtype=np.float64) + eps)
        / np.log(np.float64(from_base))).astype(int) - deg + 1
    return  np.arange(start, end) // from_base ** divisor

def simple_expand(nval: int, from_base: int = 10, to_base: int = 11) -> int:
    """
      V(10*n + c) = 11 * V(n) + c, where 0 <= c < 10
    """
    csum = 0
    val = nval
    mult = 1
    while val > 0:
        csum = csum + mult * (val % from_base)
        mult *= to_base
        val //= from_base
    return csum

def power_table(limit: int,
                from_base: int, to_base: int) -> np.ndarray:
    """
      Let nu(n) be the maximum power of from_base that divides n.
      We wish to calculated (c * to_base ** nu(n) + 1)/from_base
      which should be an integer.

      Let V(n) be the integer obtained by using the digits of n
      in from_base as an integer in to_base > from_base.

      V(n) - V(n-1) = to_base**(nu(n)) - (from_base - 1) * (to_base ** nu(n) - 1)/(to_base - 1)
      = ((to_based - 1) - (from_base - 1)) ** to_base ** nu(n) / (to_base - 1)
      + (from_base - 1)/(to_base - 1) =
      ((to_base - from_base) * to_base ** nu(n) + (from_base - 1))/(to_base - 1)

      We can do this by using powers of from_base

      Initialize the table to all 1's
      at pass k (starting at 1) for all multiples of from_base ** k
      we should subtract (to_base - from_base) * to_base ** (k-1)/(to_base - 1)
      and add (to_base - from_base) * to_base ** k / (to_base - 1).
      That is (to_base - from_base) * to_base ** (k-1)
    """
    lim = from_base ** limit
    table = np.ones(lim, dtype=np.int64)
    stride = from_base
    lstride = (to_base - from_base)
    while stride < lim:
        table[::stride] += lstride
        stride *= from_base
        lstride *= to_base
    return table

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

def digit_bounds(from_base: int, to_base: int, deg: int) -> np.ndarray:

    mpf = np.vectorize(lambda _: mp.mpf(int(_)))
    lbound = from_base ** deg
    ubound = lbound * from_base
    expon = mp.log(to_base)/mp.log(from_base)
    table = expand_table(deg + 1, from_base, to_base)
    leading = np.arange(lbound, ubound)
    twiddle = mp.mpf(from_base - 1) / (to_base - 1)
    lower =  leading ** expon / mpf(table[leading])
    # lower =  leading ** expon / (mpf(table[leading]) + twiddle)
    # middle = leading ** expon / mpf(table[leading])
    # upper =  (leading + 1) ** expon / mpf(table[leading])
    upper =  (leading + 1) ** expon / (mpf(table[leading]) + twiddle)
    return np.vstack([lower, upper, leading, table[leading]])

def make_ratio(limit: int, deg: int, from_base: int = 10, to_base: int = 11) -> Tuple[np.ndarray]:
    "Use deg leading digits to calculate upper and lower bounds"

    # Get a table of V(n) for 0 <= n < from_base ** limit
    table = expand_table(limit, from_base, to_base)
    # Find leading digits range
    start = from_base ** (deg - 1)
    lend = start * from_base
    end = from_base ** limit
    expon = np.log(to_base) / np.log(from_base)
    values = np.arange(start, end) ** expon / table[start: end]
    # Now do ratios
    lbound = (np.arange(start, lend) + 1) ** expon / table[start: lend]
    mbound = np.arange(start, lend) ** expon / table[start: lend]
    twiddle = (from_base - 1) / (to_base - 1)
    ubound = (table[start: lend] + twiddle) / np.arange(start, lend) ** expon
    leading = leading_digits(limit, deg, from_base = from_base, to_base = to_base) - start
    indices = np.arange(start, end)
    return indices, values, lbound[leading], ubound[leading], mbound[leading]
