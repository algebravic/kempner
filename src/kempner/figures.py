"""
  Produce plots for the paper
"""
from functools import partial
from typing import List, Callable, Union, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .util import leading_digits, expand_table, make_ratio
from .majorization import FLOAT, vaaler, vaalerx

def ratio_plot(limit: int,
               from_base: int = 10, to_base: int = 11,
               deg: int = 0):
    end = from_base ** limit
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.set_xlabel(r"$n$")
    if deg > 0: # do some modulation
        ax.set_ylabel('modulated ratios')
        # We must have deg < limit
        if deg >= limit:
            print(f"deg must be < limit, setting it {limit-1}")
            deg = limit - 1
        x_formula = r"$\frac{n^\alpha}{V(n)} - \frac{a^\alpha}{V(a)}$" + f", degree={deg}"
        indices, values, lvalues, uvalues, mvalues = make_ratio(limit, deg,
            from_base = from_base,
            to_base = to_base)
        diffs = values - mvalues
        ax.plot(indices, diffs, color='blue')
    else:
        table = expand_table(limit, from_base, to_base)
        expon = np.log(to_base) / np.log(from_base)
        values = np.arange(1, end) ** expon / table[1: end]
        ax.set_ylabel("raw ratios")
        x_formula = r"$\frac{n^\alpha}{V(n)}$"
        ax.plot(np.arange(1, end), values, color='blue')
    ax.set_title(x_formula)
    return fig, ax

def plot_ratios(limit: int, deg: int,
                from_base: int = 10, to_base: int = 11,
                show_bounds: bool = True,
                show_values: bool = True,
                x_log: bool = False,
                y_log: bool = False,
                truncated: int = 0):

    indices, values, lvalues, uvalues, mvalues = make_ratio(limit, deg,
        from_base = from_base,
        to_base = to_base)
    to_start = (from_base ** (deg + truncated - 1) - from_base ** (deg - 1))
    ratio = values / mvalues
    fig, ax = plt.subplots(1, 3)
    if x_log:
        for _ in ax:
            _.set_xscale('log')
        additional = ': log scale'
    else:
        additional = ''
    if y_log:
        for _ in ax:
            _.set_yscale('log')
            
    x_formula = r"$\frac{V(n)}{n^\alpha}$"
    x_title = " and lower/upper bounds" + additional + '.' if show_bounds else ''
    # ax.figure('bounds')
    if show_values:
        ax[0].plot(indices[to_start:], values[to_start:], color='red')
    if show_bounds:
        ax[0].plot(indices[to_start:], lvalues[to_start:], color='green')
        ax[0].plot(indices[to_start:], uvalues[to_start:], color='blue')
    ax[0].plot(indices[to_start:], ratio[to_start:], color='purple')
    ax[0].set_title(x_formula + x_title)
    ax[0].set_xlabel(r"$n$")
    ax[0].set_ylabel("ratios")
    # ax.figure('magnified')
    # Find the last bounds
    last = from_base ** deg
    if show_values:
        ax[1].plot(indices[-last:], values[-last:], color='red')
    if show_bounds:
        ax[1].plot(indices[-last:], lvalues[-last:], color='green')
        ax[1].plot(indices[-last:], uvalues[-last:], color='blue')
    ax[1].plot(indices[-last:], ratio[-last:], color='purple')
    ax[1].set_title('Tail ' + x_formula + x_title)
    ax[1].set_xlabel(r"$n$")
    ax[1].set_ylabel("ratios")
    # ax.figure("ratio")
    ax[2].plot(indices[-last:], ratio[-last:], color='purple')
    ax[2].set_title("Tail " + x_formula + x_title)
    ax[2].set_xlabel(r"$n$")
    ax[2].set_ylabel("ratios")
    print(f"ratio in [{ratio[-last:].min()},{ratio[-last:].max()}]")
    return fig, ax


def characteristic(alpha: FLOAT, beta: FLOAT, order: int, points: int):

    adj = np.floor(alpha)
    intrvl = (alpha - adj, beta - adj)
    ilen = beta - alpha
    mid = 0.5 * (alpha + beta) - adj
    pts = np.linspace(mid - ilen, mid + ilen, points)
    func = partial(charfunc, alpha, beta)
    minf = partial(minorant, alpha, beta, order)
    majf = partial(majorant, alpha, beta, order)
    fig = plt.figure("characteristic")
    ax = fig.subplots()
    ax.set_title(f"Characteristic function from sawtooth [{alpha: .4f},{beta: .4f}]")
    ax.plot(pts, list(map(func, pts)), color='red')
    ax.plot(pts, list(map(minf, pts)), color='blue')
    ax.plot(pts, list(map(majf, pts)), color='green')

def vdiff(order: int, points: int):
    pts = [_/points for _ in range(points)]
    vfunc = partial(vaaler, order)
    xfunc = partial(vaalerx, order)
    fig = plt.figure("vaaler")
    ax = fig.subplots()
    ax.set_title("Comparison of true vaaler")
    ax.plot(pts, list(map(vfunc, pts)), color='red')
    ax.plot(pts, list(map(xfunc, pts)), color='blue')

def simple_stuff(from_base: int = 10, to_base: int = 11):
    expon = np.log(to_base) / np.log(from_base)

    return [(lower * to_base + _) / (lower * from_base + _) ** expon
            for _ in range(from_base) for lower in range(1, from_base)]
    
def simple(from_base: int = 10, to_base: int = 11):
    stuff = simple_stuff(from_base, to_base)
    fig, ax = plt.subplots(1, 1)
    ax.set_xscale('log')
    ax.plot(range(from_base, from_base ** 2), stuff)
    return fig, ax
