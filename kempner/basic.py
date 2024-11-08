"""
"""
from typing import Tuple, List, Dict
from functools import reduce, partial
import numpy as np
from mpmath import mp
from .timeit import Timeit
from .expand import expand, get_chunk

FLOAT = mp.mpf | np.float64
DUAL = Tuple[FLOAT, FLOAT]

class Basic:
    def __init__(self, from_base: int = 10, to_base = 11,
                 precision = 50,
                 limit: int = 7,
                 use_numpy: bool = False):
        self._from_base = from_base
        self._to_base = to_base
        self._limit = limit
        if use_numpy:
            mp.prec = 53
            self._float = np.float64
            self._floor = np.floor
            self._exp = np.exp
            self._log = np.log
        else:
            mp.mps = precision
            self._float = mp.mpf
            self._floor = mp.floor
            self._exp = mp.exp
            self._log = mp.log
        self._expon = self._log(self._to_base) / self._log(self._from_base)
        self._incr = self._float(self._from_base - 1) / (self._to_base - 1)
        self.expand = partial(expand,
            self._cutoff, self._from_base, self._to_base)

    def log_bound(self, arg: int) -> Tuple[FLOAT, FLOAT]:

        xpn = self.expand(arg)
        lower = self._log(xpn) - self._expon * self._log(arg + 1)
        upper = (self._log(xpn + self._incr)
            - self._expon * self._log(arg))
        return lower, upper

    def deviation(self, start: int, arg: int) -> FLOAT:

        digs = int(self._floor(
            self._floor(self._log(arg) / self._log(from_base))))
        if digs <= start:
            raise ValueError(f"{arg} : {start} >= {digs}")

        reduced = arg // 10 ** (digs - start)
        lcoeff = (self._log(self.expand(reduced))
            - self._expon * self._log(reduced + 1))
        ucoeff = (self._log(self.expand(reduced) + self._incr)
            - self._expon * self._log(reduced))
        main_term = (self._log(self.expand(arg))
            - self._expon * self._log(arg))
        
        return (main_term - lcoeff, ucoeff - main_term)

    def coeff_bounds(self, digits: int,
                     strict: bool = True) -> Tuple[FLOAT, FLOAT]:

        def min_max(arg1: DUAL, arg2: DUAL) -> DUAL:

            return min(arg1[0], arg2[0]), max(arg1[1], arg2[1])
        
        lbound = self._from_base ** digits
        ubound = self._from_base * lbound
        if not strict:

            boundl = self.log_bound(ubound - 1)[0]
            boundu = self.log_bound(lbound)[1]

        else:
            with Timeit("bounds"):
                boundl, boundu = reduce(min_max,
                    map(self.log_bound, range(lbound, ubound)))
            
        return self._exp(- boundu), self._exp(- boundl)
    

    def simple_bounds(self, digits: int,
                      strict: bool = True) -> Tuple[FLOAT, FLOAT]:
        """
          Find simple lower and upper bounds by exhausting over digits.
          Integers in [lbound, ubound) can be prefixes of all
          integers >= lbound (including primes)
        """

        initial = self._float(0.0)
        first = self._float(0.0)
        lbound = self._from_base ** digits
        cutoff = self._from_base ** self._limit

        gchunk = partial(get_chunk,
            self._from_base, self._to_base, self._limit)

        # Do this in segments
        for start in range(1, lbound + 1, cutoff):
            end = min(lbound, start + cutoff)
            with Timeit(f"series chunks at ({start}, {end})"):

                prime_sum, xpn_sum = gchunk(start,end)
                initial += prime_sum
                first += xpn_sum

        print(f"initial = {initial}, first = {first}")

        tailvalue = mp.primezeta(self._expon) - initial
        print(f"tail = {tailvalue}")
        lmult, umult = self.coeff_bounds(digits, strict = strict)
        
        lower = first + lmult * tailvalue
        upper = first + umult * tailvalue

        return lower, upper

    def intervals(self, start: int, arg: int) -> FLOAT:

        digs = int(self._floor(self._log(arg) / self._log(from_base)))
        if digs < start:
            raise ValueError(f"{arg} : {start} > {digs}")

        lcoeff, ucoeff = self.log_bound(arg //
            (self._from_base ** (digs - start)))
        return ucoeff - lcoeff

    def rdeviation(arg: int) -> FLOAT:

        xpn0 = self.expand(arg)
        xpn1 = self._expand(arg // from_base)
        delta = self._log(xpn0) + self._log(self._to_base) * self._log(xpn1)

        return delta + self._expon * self._log(arg)
