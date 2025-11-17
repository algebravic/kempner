from typing import Tuple, List
from functools import cache, partial
import numpy as np
import mpmath as mp
from sympy import primerange
from joblib import Memory
memory = Memory('primecache.dat')

def expanded(from_base: int, to_base: int, limit: int, arg: int) -> int:
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
    vxpnd = np.vectorize(partial(expanded, from_base, to_base, limit))
    expon = mp.log(to_base) / mp.log(from_base)
    my_primes = np.array(list(primerange(start, end)), dtype=np.uint64)
    prime_sum = (my_primes ** (- expon)).sum()
    xpn_sum = (1 / vxpnd(my_primes)).sum()

    return prime_sum, xpn_sum

def prime_zeta_values(from_base: int, to_base: int,
                      nterms: int) -> List[mp.mpf]:

    expon = mp.log(to_base) / mp.log(from_base)
    return [mp.primezeta(expon - (2j * np.pi * _)
                         / mp.log(from_base))
            for _ in range(nterms)]

