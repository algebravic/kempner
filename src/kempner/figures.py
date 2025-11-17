"""
  Produce plots for the paper
"""
from functools import partial
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .expand import expand_table, digit_bounds

FLOAT = float | np.float32 | np.float64

def leading_digits(limit: int, deg: int, from_base: int = 10, to_base: int = 11) -> np.ndarray:
    start = from_base ** (deg - 1)
    end = from_base ** limit
    # Apparently there's a bug in np.log
    # I found that np.log(1000)/np.log(10) is 2.9999 and not 3.0000
    eps = 1e-9
    divisor = np.floor(np.log(np.arange(start, end, dtype=np.float64) + eps)
        / np.log(np.float64(from_base))).astype(int) - deg + 1
    return  np.arange(start, end) // from_base ** divisor

def make_ratio(limit: int, deg: int, from_base: int = 10, to_base: int = 11) -> List[np.ndarray]:
    "Use deg leading digits to calculate upper and lower bounds"

    # Get a table of V(n) for 0 <= n < from_base ** limit
    table = expand_table(limit, from_base, to_base)
    # Find leading digits range
    start = from_base ** (deg - 1)
    lend = start * from_base
    end = from_base ** limit
    expon = np.log(to_base) / np.log(from_base)
    bounds = digit_bounds(from_base, to_base, deg)
    values = table[start: end] / np.arange(start, end) ** expon
    # Now do ratios
    lbound = table[start: lend] / (np.arange(start, lend) + 1) ** expon
    mbound = table[start: lend] / np.arange(start, lend) ** expon
    twiddle = (from_base - 1) / (to_base - 1)
    ubound = (table[start: lend] + twiddle) / np.arange(start, lend) ** expon
    leading = leading_digits(limit, deg, from_base = from_base, to_base = to_base) - start
    indices = np.arange(start, end)
    return indices, values, lbound[leading], ubound[leading], mbound[leading]

def plot_ratios(limit: int, deg: int,
                from_base: int = 10, to_base: int = 11,
                truncated: int = 0):

    indices, values, lvalues, uvalues, mvalues = make_ratio(limit, deg,
        from_base = from_base,
        to_base = to_base)
    to_start = (from_base ** (deg + truncated - 1) - from_base ** (deg - 1))
    ratio = values / mvalues
    plt.figure('bounds')
    plt.plot(indices[to_start:], values[to_start:], color='red')
    plt.plot(indices[to_start:], lvalues[to_start:], color='green')
    plt.plot(indices[to_start:], uvalues[to_start:], color='blue')
    plt.plot(indices[to_start:], ratio[to_start:], color='purple')
    plt.title(r"$\frac{V(n)}{n^\alpha}$ and lower/upper bounds.")
    plt.xlabel(r"$n$")
    plt.ylabel("ratios")
    plt.figure('magnified')
    # Find the last bounds
    last = from_base ** deg
    plt.plot(indices[-last:], values[-last:], color='red')
    plt.plot(indices[-last:], lvalues[-last:], color='green')
    plt.plot(indices[-last:], uvalues[-last:], color='blue')
    plt.plot(indices[-last:], ratio[-last:], color='purple')
    plt.title(r"$Tail \frac{V(n)}{n^\alpha}$ and lower/upper bounds.")
    plt.xlabel(r"$n$")
    plt.ylabel("ratios")
    plt.figure("ratio")
    plt.plot(indices[-last:], ratio[-last:], color='purple')
    plt.title(r"$Tail \frac{V(n)}{n^\alpha}$ and lower/upper bounds.")
    plt.xlabel(r"$n$")
    plt.ylabel("ratios")
    print(f"ratio in [{ratio[-last:].min()},{ratio[-last:].min()}]")

def sawtooth(arg: FLOAT) -> FLOAT:

    delta = arg - np.floor(arg)
    return 0.0 if delta == 0.0 else delta - 0.5

def charfunc(alpha: FLOAT, beta: FLOAT, arg: FLOAT):

    return beta - alpha + sawtooth(arg - beta) + sawtooth(alpha - arg)

def fejer(order: int, arg: FLOAT) -> FLOAT:

    return (order if arg == 0.0
            else (1/order) * (np.sin(np.pi * order * arg)/np.sin(np.pi * arg)) ** 2)

def vaalerx(order: int, arg: FLOAT) -> FLOAT:

    return ((1/(order + 1)) * sum(((_/(order+1) -0.5)*fejer(order+1, arg - _/(order+1))
                                  for _ in range(1, order+1)))
        + np.sin(2 * np.pi * (order + 1) * arg)/(2 * np.pi * (order + 1))
            - ((np.sin(2 * np.pi * arg)/(2 * np.pi)) * fejer(order + 1, arg)))

def vaaler(order: int, arg: FLOAT) -> FLOAT:
    """
      According to Vaaler, Theorem 6: Jhat(t) = pi t (1-|t|) cot pi t + |t|
      when 0< |t| < 1. And the expansion psi * j_N (what Vaaler calls the
      Vaaler polynomial) is
      V_N(x) = sum_{n=-N to N, n != 0} (-2 pi i n)^{-1} Jhat(n/(N+1)) e(nx)

      Jhat(n/(N+1)) / (- pi * n) = - (1 / (N+1)) Jhat(n/(N+1)) / (pi * n/(N+1))
    """

    def coeff(argu: FLOAT):
        return -(1-argu) / np.tan(np.pi * argu) - 1 / np.pi

    return (1/(order + 1)) * sum((coeff(_/(order+1)) * np.sin(2 * np.pi * _ * arg)
                                  for _ in range(1, order + 1)))

def beurling(order: int, arg: FLOAT) -> FLOAT:

    return vaaler(order, arg) + (1/(2*(order + 1))) * fejer(order + 1, arg)

def majorant(alpha: FLOAT, beta: FLOAT, order: int, arg: FLOAT) -> FLOAT:

    return beta - alpha + beurling(order, arg - beta) + beurling(order, alpha - arg)
    
def minorant(alpha: FLOAT, beta: FLOAT, order: int, arg: FLOAT) -> FLOAT:

    return beta - alpha - beurling(order, beta - arg) - beurling(order, arg - alpha)

def characteristic(alpha: FLOAT, beta: FLOAT, order: int, points: int):


    adj = np.floor(alpha)
    intrvl = (alpha - adj, beta - adj)
    ilen = beta - alpha
    mid = 0.5 * (alpha + beta) - adj
    pts = np.linspace(mid - ilen, mid + ilen, points)
    func = partial(charfunc, alpha, beta)
    minf = partial(minorant, alpha, beta, order)
    majf = partial(majorant, alpha, beta, order)

    plt.figure("characteristic")
    plt.title(f"Characteristic function from sawtooth [{alpha},{beta}]")
    plt.plot(pts, list(map(func, pts)), color='red')
    plt.plot(pts, list(map(minf, pts)), color='blue')
    plt.plot(pts, list(map(majf, pts)), color='green')

def vdiff(order: int, points: int):
    pts = [_/points for _ in range(points)]
    vfunc = partial(vaaler, order)
    xfunc = partial(vaalerx, order)
    plt.figure("vaaler")
    plt.title("Comparison of true vaaler")
    plt.plot(pts, list(map(vfunc, pts)), color='red')
    plt.plot(pts, list(map(xfunc, pts)), color='blue')
