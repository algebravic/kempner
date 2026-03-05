from scipy.interpolate import AAA
import numpy as np
from .expand import expand_table, digit_bounds

def interp(from_base: int, to_base: int, deg: int, max_terms: int = 100):
    # V(n) for n in range(from_base ** deg, from_base ** (deg + 1))
    table = expand_table(deg + 1, from_base, to_base)[from_base ** deg:]
    indices = np.arange(from_base ** deg, from_base ** (deg + 1))
    xvals = np.log(indices) / np.log(from_base) - deg
    expon = np.log(to_base) / np.log(from_base)
    yvals = indices ** expon / table
    return AAA(xvals, yvals, max_terms = max_terms), xvals, yvals
