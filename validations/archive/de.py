"""Takahasi-Mori double-exponential (tanh-sinh) nodes and weights.

x(t) = 0.5(b-a) tanh(pi/2 sinh t) + 0.5(a+b)
w(t) = 0.5(b-a) h * (pi/2 cosh t) / cosh^2(pi/2 sinh t)

Nodes cluster doubly-exponentially near a and b -- the opposite of
Gauss-Legendre, which is densest mid-interval. Ideal when the
integrand peaks (or is singular) at the interval endpoints.
"""
import numpy as np


def de_nodes(a, b, N, t_max=5.0):
    """Nodes and weights for int_a^b f(x) dx ~ sum_i w_i f(x_i)."""
    if N < 2:
        raise ValueError("de_nodes needs N >= 2")
    t = np.linspace(-t_max, t_max, N)
    h = 2.0 * t_max / (N - 1)
    s = 0.5 * np.pi * np.sinh(t)
    phi = np.tanh(s)
    dphi = (0.5 * np.pi * np.cosh(t)) / np.cosh(s) ** 2
    x = 0.5 * (b - a) * phi + 0.5 * (a + b)
    w = 0.5 * (b - a) * h * dphi
    return x, w
