"""
"""
from typing import Tuple, List, Dict
from functools import cache, reduce
from itertools import product
from math import log, floor
import numpy as np
from mpmath import mp
from sympy import sieve
from pyprimesieve import primes
from .timeit import Timeit

FLOAT = mp.mpf | np.float64
DUAL = Tuple[FLOAT, FLOAT]

class Expander:
    def __init__(self, from_base: int = 10, to_base = 11,
                 precision = 50,
                 cutoff: int = 7,
                 maxsieve: int = 10 ** 7,
                 use_numpy: bool = False):
        self._from_base = from_base
        self._to_base = to_base
        self._cutoff = cutoff
        self._limit = from_base ** cutoff
        self._maxsieve = maxsieve
        if use_numpy:
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
        self._table = np.empty(self._limit, dtype = np.int64)
        self.populate()

    def populate(self):

        self._table[: self._from_base] = np.arange(self._from_base)
        fence = self._from_base
        ofence = self._to_base
        # We've filled in out[: from_base ** ind]
        for ind in range(self._cutoff - 1):
            for indx in range(1, self._from_base):
                bot = indx * fence
                addend = indx * ofence

                self._table[bot: bot + fence] = (self._table[: fence]
                                             + addend)
            fence *= self._from_base
            ofence *= self._to_base

    def expand(self, arg: int) -> int:

        res = 0
        narg = arg
        mult = 1
        while narg >= self._limit:
            res += mult * (narg % self._from_base)
            narg //= self._from_base
            mult *= self._to_base
        return res + mult * int(self._table[narg])

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

        # Do this in segments
        with Timeit("series chunks"):
            for start in range(2, lbound + 1, self._maxsieve):

                my_primes = primes(start,
                    min(lbound, start + self._maxsieve) + 1)

                initial += sum((self._float(_) ** (- self._expon)
                    for _ in my_primes))
                first += sum((1 / self._float(self.expand(_))
                    for _ in my_primes))

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
