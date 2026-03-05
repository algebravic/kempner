"""
  Functions pertaining to Selberg et. al. Majorization/Minorization.
"""
import numpy as np
INT = int | np.int8 | np.int16 | np.int32 | np.int64
# A function that gives an estimate of V(n)
ESTIMATE = Callable[[INT], FLOAT]

FLOAT = float | np.float32 | np.float64

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

def extremal(val: List[float]) -> List[Tuple[int, float]]:
    """
      Find extremal points in a list a, of floats
      A point at position i is extremal if
      either
      1) a[i-1] < a[i] and a[i] > a[i+1]
      or
      2) a[i-1] > a[i] and a[i] < a[i+1]

      How to deal with equality?
      I would say equal values should be compressed
    """
    values = np.array(val)
    sdiffs = values[: -1] > values[1: ]
    seq = values[: -1] != values[1: ]
    diffs = sdiffs & seq
    xdiffs = diffs[1: ] ^ diffs[: -1] # 1 shorter
    places = map(int, np.arange(len(xdiffs))[xdiffs] + 1)
    return [(_, float(values[_])) for _ in places]
