"""
  Graphs of the prime zeta function at spaced values
"""
from typing import List
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

def prime_zeta_values(real_part: mp.mpf, imag_part: mp.mpf, terms: int,
                      start: int = 0) -> np.ndarray:

    return np.array([mp.primezeta(real_part + 1j * imag_part * _)
                     for _ in range(start, terms)])

def plot_prime_zeta(real_part: mp.mpf, imag_part: mp.mpf, terms: int,
                    start: int = 0):

    values = prime_zeta_values(real_part, imag_part, terms, start = start)
    magnitudes = np.vectorize(mp.fabs)(values)
    arguments = np.vectorize(mp.arg)(values)
    plt.title("Prime zeta values magnitude")
    plt.xlabel("index")
    plt.ylabel("Magnitude")
    plt.plot(magnitudes.astype(np.float64), color='red')
    plt.plot(arguments.astype(np.float64), color='blue')
