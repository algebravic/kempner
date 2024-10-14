"""
"""
from typing import Tuple, List, Dict
from functools import cache
from itertools import product
from math import log, floor
import numpy as np
from mpmath import mp
from sympy import sieve
from pyprimesieve import primes

class Expander:
    def __init__(self, from_base: int = 10, to_base = 11,
                 precision = 50,
                 cutoff: int = 7,
                 maxsieve: int = 10 ** 7):
        self._from_base = from_base
        self._to_base = to_base
        self._cutoff = cutoff
        self._limit = from_base ** cutoff
        self._maxsieve = maxsieve
        mp.mps = precision
        self._expon = mp.log(self._to_base) / mp.log(self._from_base)
        self._incr = mp.mpf(self._from_base - 1) / (self._to_base - 1)
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

    def log_bound(self, arg: int) -> Tuple[mp.mpf, mp.mpf]:

        xpn = self.expand(arg)
        lower = mp.log(xpn) - self._expon * mp.log(arg + 1)
        upper = (mp.log(xpn + self._incr)
            - self._expon * mp.log(arg))
        return lower, upper

    def deviation(self, start: int, arg: int) -> mp.mpf:

        digs = int(mp.floor(
            mp.floor(mp.log(arg) / mp.log(from_base))))
        if digs <= start:
            raise ValueError(f"{arg} : {start} >= {digs}")

        reduced = arg // 10 ** (digs - start)
        lcoeff = (mp.log(self.expand(reduced))
            - self._expon * mp.log(reduced + 1))
        ucoeff = (mp.log(self.expand(reduced) + self._incr)
            - self._expon * mp.log(reduced))
        main_term = (mp.log(self.expand(arg))
            - self._expon * mp.log(arg))
        
        return (main_term - lcoeff, ucoeff - main_term)


    def simple_bounds(self, digits: int) -> Tuple[mp.mpf, mp.mpf]:
        """
          Find simple lower and upper bounds by exhausting over digits.
          Integers in [lbound, ubound) can be prefixes of all
          integers >= lbound (including primes)
        """
        lbound = self._from_base ** digits
        ubound = self._from_base * lbound
        boundl, boundu = tuple(zip(*map(self.log_bound, 
            range(lbound, ubound))))

        initial = mp.mpf(0.0)
        first = mp.mpf(0.0)

        # Do this in segments

        for start in range(2, lbound + 1, self._maxsieve):

            my_primes = primes(start,
                min(lbound, start + self._maxsieve) + 1)

            initial += sum((mp.mpf(_) ** (- self._expon)
                for _ in my_primes))
            first += sum((1 / mp.mpf(self.expand(_))
                for _ in my_primes))

        print(f"inital = {initial}, first = {first}")

        tailvalue = mp.primezeta(self._expon) - initial
        print(f"tail = {tailvalue}")

        lower = first + mp.exp(- max(boundu)) * tailvalue
        upper = first + mp.exp(- min(boundl)) * tailvalue

        return lower, upper

    def intervals(self, start: int, arg: int) -> mp.mpf:

        digs = int(mp.floor(mp.log(arg) / mp.log(from_base)))
        if digs < start:
            raise ValueError(f"{arg} : {start} > {digs}")

        lcoeff, ucoeff = self.log_bound(arg //
            (self._from_base ** (digs - start)))
        return ucoeff - lcoeff

    def rdeviation(arg: int) -> mp.mpf:

        xpn0 = self.expand(arg)
        xpn1 = self._expand(arg // from_base)
        delta = mp.log(xpn0) + mp.log(self._to_base) * mp.log(xpn1)

        return delta + self._expon * mp.log(arg)
