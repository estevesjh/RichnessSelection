"""Gauss-Legendre nodes and weights with lru_cache."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=64)
def _leggauss_N(N):
    return leggauss(N)


def gl_nodes(a, b, N):
    """Nodes and weights for int_a^b f(x) dx ~ sum_i w_i f(x_i)."""
    t, w = _leggauss_N(N)
    return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * (b - a) * w
