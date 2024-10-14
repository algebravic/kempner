"""
  The Shanks transofrm.

  Given a generator producing mp floats, become a generator producing the Shanks transformation

  S(A_n) = (A_{n+1} A_{n-1} - A_n^2)/(A_{n+1} - 2 A_n + A_{n-1}) if the denominator is not 0
  otherwise.

  An alternative is A_{n+1} - (A_{n+1} - A_n)^2/((A_{n+1} - A_n) - (A_n - A_{n-1}))
"""
from typing import Generator, Callable
from itertools import islice, accumulate, count, cycle
from mpmath import mp

GEN = Generator[mp.mpf, None, None]

def shanks_acceleration(arg: GEN) -> GEN:

    # Initialize
    values = list(islice(arg,3))

    while True:

        delta0 = values[2] - values[1]
        delta1 = values[1] - values[0]
        yield values[2] - delta0 ** 2/(delta0 - delta1)
        values = values[1:] + [next(arg)]

def test_series() -> GEN:
    """
      4 (-1)^n/(2n+1) for n=0, ...
    """

    return map(lambda _: _[0] / _[1],
               zip(cycle((4,-4)), count(1,2)))

def compose(mult: int, opr: Callable[[GEN], GEN]) -> Callable[[GEN], GEN]:

    def _inner(arg: GEN) -> GEN:
        if mult < 0:
            raise ValueError(f"Multiplicity {mult} < 0")
        val = arg
        for _ in range(1, mult):
            val = opr(val)
        return val
    return _inner

def test_shanks(mult: int = 1) -> GEN:
    """
      Test with partial sums of 1 - 1/3 + 1/5 - 1/7 + ...
    """
    return compose(mult, shanks_acceleration)(
        accumulate(test_series()))
