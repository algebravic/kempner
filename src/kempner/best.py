"""
  Find the smallest set of leading digit covers so that each
  yields error bounds that are indicated.
"""
from typing import List, Iterable
import mpmath as mp
from heapq import heapify, heappop, heappush

class Interval:
    pass

class Interval:

    def __init__(self, leading: int, expanded: int,
                 from_base: int = 10,
                 to_base: int = 11):

        self._leading = leading
        self._expanded = expanded
        self._from_base = from_base
        self._to_base = to_base
        expon = mp.log(self._to_base) / log(self._from_base)
        self._left = self._expanded / mp.mpf(self._leading + 1) ** expon
        self._right = ((self._expanded + ((self._from_base - 1) / (self._to_base - 1))
            / mp.mpf(self._leading) ** expon))
        # self._key = self._right / self._left
        self._key = 1 / self._left - 1 / self._right
    def __lt__(self, other: Interval | float) -> bool:

        return self._key < (other._key if isinstance(other, Interval) else other)

    def __str__(self) -> str:

        return f"I({self._leading}:{self._key})"

    def __repr__(self) -> str:
        return f"Interval({self._leading}, {self._expanded}, base={self._base})"

def covers(precision: mp.mpf,
           from_base: int = 10,
           to_base: int = 11,
           verbose: int = 0) -> Iterable[Interval]:
    """
      Find the minimal sized cover so that the the width of each
      element is < 10^(-prec).
    """
    # Initialize to a one digit cover
    heap = [Interval(_, _, from_base = from_base, to_base = to_base)
        for _ in range(1, from_base)]
    heapify(heap)

    prec = mp.mpf(10) ** ( - precision) 

    while heap:
        # emit all of the elements that are sufficiently small
        while heap and ((elt := heappop(heap)) < prec):
            yield elt
        # if there is an element that isn't small enough, split it.
        if not (elt < prec):
            # Now elt is too big. Split it.
            if verbose > 0:
                print(f"smallest violation = {elt}")

            for bottom in range(base):

                heappush(heap,
                         Interval(from_base * elt._leading + bottom,
                                  to_base * elt._expanded + bottom,
                                  from_base = from_base,
                                  to_base = to_base))

def check_cover(cover: List[int]) -> bool:
    """
      Given a list of positive integers, check the following:
      1) No member is prefix of any other
      2) All but a finite set of positive integers is dominated by some member.

      We will specify subsets of N as follows:

      C(a,j): the set of elements of the form a * b^k + c
      were 0 <= k <= j and 0 <= c < b^k

      C(a,None): the set of elements of the form a * b^k + c
      where 0 <= k and 0 <= c < b^k.

      In order to cover it means that.

      If a cover has an element of the form (a,None), its complement
      is {((a // b) + c , j) for 0 <= c < b} where j satisfies
      b^j <= a < b^{j+1}.
    """
    pass
