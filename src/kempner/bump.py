"""
  Fourier transform of the bump function
  exp(1/(1-x^2)) when |x| < 1 and 0 otherwise.

  See 'Saddle point integration of C infinity bump functions'
  by Steven G. Johnson, arXiv:1508.04376v1

  The Asymptotic formula for

  2 Re int_0^1 exp(i k x - 1/(1-x^2)) dx is

  2 Re (sqrt((-i pi)/(sqrt(2 i k) k^{3/2})) exp(ik - 1/4 - sqrt(2ik)))

  Here we use the principal value for the square root. Thus

  sqrt(2i) = 1+i

  So sqrt(-i) = i (1+i)/sqrt(2) = (-1+i)/sqrt(2)

  and sqrt(2ik) = (1+i) sqrt(k) (here k>0 is real).

  and sqrt((-i)/sqrt(2i)) = sqrt(-i/(1+i)) = sqrt((1-i)/sqrt(2))2^(-3/4)
  = 2^(-3/4) sqrt(cos(-pi/4) + i sin(-pi/4)) = 2^(-3/4)*(cos(-pi/8) + i sin(-pi/8))
"""
import numpy as np

def fourier_bump(karg : np.float64) -> np.float64:

    coeff = np.float64(2) ** (- 0.75) * (np.cos(-np.pi/8) + 1j * np.sin(-np.pi/8)) * np.sqrt(np.pi)
    return 2 * (coeff * karg ** (-0.75) * np.exp(1j * karg
                                                 - 0.25
                                                 - (1 + 1j)
                                                 * np.sqrt(karg))).real
