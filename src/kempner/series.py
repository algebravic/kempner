r"""
   Class for calculating approximations to the values of the series

   sum_n 1/V(n) and sum_p 1/V(p)
   where V(n) is the integer formed by using the digits of n in base b
   and using them as the digits of a number in base b' > b.
"""
from typing import List, Tuple
import mpmath as mp
import numpy as np
from .expand import expanded, expand_table

class Series:

    def __init__(self, from_base: int, to_base: int):

        self._from_base = from_base
        self._to_base = to_base
        self._sinc = np.vectorize(mp.sincpi)
        self._exp = np.vectorize(mp.exp)
        self._log = np.vectorize(mp.log)
        self._expon = self._log(to_base) / self._log(from_base)
        self._iexpon = 2 * mp.pi / self._log(from_base)
        
class Approx(Series):

    def _init__(self, from_base: int, to_base: int,
                digits: int, # Number of leading digits
                limit: int): # Number of terms in the Fourier series

        parent().__init__(from_base, to_base)
        self._digits = digits
        self._limit = limit
        self._lbound = from_base ** self._digits
        self._ubound = from_base * self._lbound

    
