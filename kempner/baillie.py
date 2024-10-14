"""
  Baillie's algorithm for approximating the harmonic sum over
  integers in a base missing a digit.

  Let S denote the set of positive integers whose base b expansion
  is missing the digit d. If i is a positive integer let S[i]
  denote the subset of S whose base b expansion has exactly i
  digits.

  Let s(i,j) = sum[x in S[i]] 1/x^j
  Let D = {0 <= j < b : j != d}
  We have the recursion
  s(i+1,j) = sum[x in S[i]] sum[k in D] (b*x + k)^(-j)

  Set c(j,n) = (-1)^n * binom(j+n-1, n)
  b[n] = sum[x in D] x^n.
  a(j,n) = b[n]*c(j,n)/b^(j+n)

  We expand (b*x_k)^(-j) by the binomial theorem:
  (*) = (b*x+k)^(-j) = (b*x)^(-j) (1 + k/(b*x))^(-j)
    = (b*x)^(-j) * sum[n=0 to infinity] binom(-j,n) * (k/(b*x))^n
  But binom(-j,n) = c(j,n)

  So (*) = sum[x in S[i]] (b*x)^(-j) * sum[n] b[n] * c(j,n) (b*x)^(-n)

  Thus
  (**) s(i+1,j) = sum[n=0 to infinity] a(j,n) * s(i,j+n)

  The original sum is sum[i] s(i,1)

  Todo: find good upper bounds for a(j,n),

  From https://mathoverflow.net/questions/236508/are-there-good-bounds-on-binomial-coefficients

  binomial(n,k) <= sqrt(n/(2*pi*k*(n-k))) * exp(n*h(k/n))
  where h(x) = -x * log x - (1-x) * log (1-x)
  is the binary entropy function.

  Note that exp(n*h(k/n)) = n^n/(k^k* (n-k)^(n-k))

  So binomial(j+n-1,n) <= sqrt((j+n-1)/(2*pi*n*(j-1))) *
        (j+n-1)^(j+n-1)/(n^n * (j-1)^(j-1))

  Reference: Gallager, Information Theory and Reliable Communication

  A simple upper bound for s(i,j) is ((b-1)^i/b^(i*j))(b^(i+1) - b^i)
  = (b-1)^(i+1)/(b^(i*(j-1))) = (b-1) * ((b-1)/b^(j-1))^i

  From Baillie's paper:

  The needed s(i,j) for i <= 4 can be computed explicitly. The
  recurrence (**) is then used with at most 10 terms to get
  s(i,1) for 5 <= i <= 30. We then use the estimate

  sum[i=31 to infinity] s(i,1) ~ (b-1) * s(30,1)
  which comes from using the first terms of (**)

  Another approach:

  We have (1-x)^(-j) = sum[n=0 to infinity] binomial(-j,n) (-x)^n
  [The reason for doing this is that the sign of binomial(-j,n) is
  (-1)^n]. Is there a nice closed form for the tail:
  T[k](x) := sum[n=k to infinity] binomial(-j,n)(-x)^n?

  Note that T[k]'(x) = - sum[n=k to infinity] n binomial(-j,n) (-x)^(n-1)
  But binomial(-j,n) = (-1)^n binomial(j+n-1,n) = (-1)^n (j+n-1)!/(n! (j-1)!)
  so n binomial(-j,n) = - j binomial(-j,n-1)

  Thus T[k]'(x) = j sum[n=k-1 to infinity] binomial(-j,n) (-x)^n = j T[k-1](x)

  Thus, by integrating T[k](x) = j integral T[k-1](x) dx

  Duh, multiply by (1-x):

  (1-x)T[k](x) = binomial(-j,k) x^k + sum[n=k+1 to infinity] (binomial(-j,n) + binomial(-j,n-1)) (-x)^n

  Alternatively: integrating x (1-x)^(-j) = - 
  
"""
from functools import cache
from itertools import chain
from mpmath import mp
from sympy import binomial

def c_coeff(jval: int, nval: int) -> int:

    return (1 - 2 * (nval % 2)) * binomial(jval + nval - 1, nval)

class Baillie:

    def __init__(self, bval: int, dval: int):
        if not (isinstance(bval, int)
                and isinstance(dval, int)
                and bval > 1
                and 0 <= dval and dval < bval):
            raise ValueError(f"Improper base {bval} digit {dval}")

        self._bval = bval
        self._dval = dval
        self._ubound = None
        self._bound = None
        self._cutoff = None

    @cache
    def b_coeff(self, nval: int) -> int:

        return sum((_ ** nval for _ in range(self._bval)
                    if _ != self._dval))

    @cache
    def a_coeff(self, jval: int, nval: int) -> mp.mpf:

        return (self.b_coeff(nval)
                * c_coeff(jval, nval)
                * (mp.mpf(self._bval) ** (-jval - nval)))

    def omitted(self, expon: int):

        if expon == 1:
            yield from (_ for _ in range(1, self._bval)
                        if _ != self._dval)
        else:
            for prefix in self.omitted(expon - 1):
                yield from (prefix * self._bval + _
                            for _ in range(self._bval)
                            if _ != self._dval)

    @cache
    def s_sums(self, ind: int, jval: int) -> mp.mpf:
        # for values <= bound use exhaustion
        if ind <= self._bound:
            return sum(map(lambda _: 1/mp.mpf(_) ** jval,
                           self.omitted(ind)))
        # For very large values use the first term of
        # the asymptotic expression
        elif ind >= self._ubound:
            # return tail estimate
            return ((self._bval - 1)
                    * self.s_sums(self._ubound - 1, jval))
        else:
            return sum((self.a_coeff(jval, nval) * 
                    self.s_sums(ind - 1, jval + nval)
                        for nval in range(self._cutoff)))

    def set_bounds(self, ubound: int, bound: int, cutoff: int):

        self._ubound = ubound
        self._bound = bound
        self._cutoff = cutoff
        self.s_sums.cache_clear() # old values are no good

    def ksums(self) -> mp.mpf:

        if self._ubound is None:
            raise ValueError("Bounds must be set")

        return sum((self.s_sums( _, 1) for _ in range(1, self._ubound)))
