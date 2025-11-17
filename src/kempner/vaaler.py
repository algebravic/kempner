"""
  Compute Vaaler trigonometric polynomials.

  Reference: 'Ten Lectures on the Interface between Analytic Number Theory
    and Harmonic Analysis' by Hugh Montgomery, Chapter 1.

  
"""
from typing import Tuple, Callable
import mpmath as mp
import numpy as np

def fejer_coeffs(order: int) -> np.ndarray:

    return np.array([1 - mp.mpf(_)/mp.mpf(order) for _ in range(order)])

def vaaler_coeffs(order: int) -> np.ndarray:
    """
      Create the fourier coefficients for the vaaler polynomial of
      a given order. Since it repersents a real function we only
      need to give fourier coefficients for nonnegative indices.
    """
    start = mp.mpf(1)/(order + 1)
    pts = mp.linspace(start, 1 - start, order)
    return (mp.mpc(1j)/ (2 * (order + 1))) * np.array(
        [mp.mpf(0)] +  [(1 - _) * mp.cot(mp.pi * _ ) + 1 / mp.pi
                        for _ in pts])

def beurling_coeffs(order: int) -> np.ndarray:

    return vaaler_coeffs(order) + (1 / (2 * mp.mpf(order + 1))) * fejer_coeffs(order+1)

def twist(arg: mp.mpc, arr: np.ndarray) -> np.ndarray:
    """
      Transform a vector A into a vector B, where
      B[j] = omega^j A[j] for j=0,...,n-1

      If [a[0],...,a[n-1]] are the fourier coefficients of a real function
      f(x), then f(x + b) has fourier coefficient twist(e(b), a),
      where e(x) = exp(2pi i x).

      We also have f(x) = 2 * real(sum(twist(e(x), a))) - a[0].
    """
    return np.array([arg ** _ for _ in range(arr.shape[0])]) * arr

class Bounds:

    def __init__(self, order: int):

        self._beurling = beurling_coeffs(order)
        self._conj = np.vectorize(mp.conj)

    def sfuncs(self, alpha: mp.mpf, beta: mp.mpf) -> Tuple[np.ndarray, np.ndarray]:
        """

          Get the fourier coefficients of S+ and S-
          Since they represent real functions we only need
          to give those for nonnegative indices.
          
          s+ = beta - alpha + B_j(x - beta) + B_j(alpha - x)
          s- = beta - alpha - B_j(beta - x) - B_j(x - alpha)

          See: Montgomery, 'Ten Lectures ...' page 6.

          Note: if a[n] are the fourier coefficients of f(x)
          and f(x) is real then a[-n] = conjugate(a[n])

          Furthermore the Fourier coefficients of f(-x)
          and the conjugates of the fourier coefficients of f(x)
          since the coefficient of e(-n x) in f(x) is the coefficient
          of e(n x) in f(-x).
          
        """
        term1 = twist(mp.exp(-2j * mp.pi * alpha), self._beurling)
        term2 = twist(mp.exp(-2j * mp.pi * beta), self._beurling)
        term1n = self._conj(twist(mp.exp(2j * mp.pi * alpha), self._beurling))
        term2n = self._conj(twist(mp.exp(2j * mp.pi * beta), self._beurling))
        res_plus = term2 + term1n
        res_minus = - term2n - term1
        res_plus[0] += beta - alpha
        res_minus[0] += beta - alpha
        return res_plus, res_minus

def interval(leading: int) -> Tuple[mp.mpf, mp.mpf]:

    return (mp.log(leading) / mp.log(10),
            mp.log(leading + 1) / mp.log(10))

def four_eval(coeffs: np.ndarray) -> Callable[[mp.mpf], mp.mpf]:
    def myeval(arg: mp.mpf):
        return 2 * mp.fsum(twist(mp.exp(2j * mp.pi * arg), coeffs)).real - coeffs[0].real
    return myeval

def test_plot(leading: int, order: int):

    alpha, beta = interval(leading)
    bnds = Bounds(order)
    bot = mp.floor(alpha)
    spl, smn = bnds.sfuncs(alpha, beta)
    cfunc = lambda _: mp.mpf(1) if ((alpha - bot) <= _) and (_ <= beta - bot) else mp.mpf(0)
    hfunc = four_eval(spl)
    lfunc = four_eval(smn)
    mid = 0.5 * (alpha + beta) - bot
    delta = beta - alpha
    mpsinc = np.vectorize(lambda _: mp.sinc(mp.pi * _))
    plain = delta * twist(mp.exp(- 2j * mp.pi * mid), mpsinc(delta * np.arange(order)))
    pfunc = four_eval(plain)

    mp.plot([cfunc, hfunc, lfunc, pfunc], [mid - delta, mid + delta])
