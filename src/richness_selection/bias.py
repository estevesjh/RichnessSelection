"""Tinker 2010 peak-height halo bias b(M, z).

Paper: Tinker+ 2010, ApJ 724, 878, Eq. 6 and Table 2 (Delta = 200m):
    b(nu) = 1 - A * nu^a / (nu^a + delta_c^a) + B * nu^b + C * nu^c

with
    A = 1 + 0.24 y exp(-(4/y)^4),     a = 0.44 y - 0.88
    B = 0.183,                         b = 1.5
    C = 0.019 + 0.107 y + 0.19 exp(-(4/y)^4), c = 2.4
    y = log10(Delta)

nu = delta_c / sigma(M, z), delta_c = 1.686.

This module is PURE halo bias.  Do NOT put selection-bias / projection
machinery here; that lives in sel_bias.py.
"""
from __future__ import annotations
import numpy as np

from .sigma_m import SigmaM

DELTA_C = 1.686


def _tinker10_params(delta=200.0):
    y = np.log10(delta)
    A = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4))
    c = 2.4
    return A, a, B, b, C, c


class Bias:
    """b(M, z) via Tinker 2010 peak-height formula."""

    def __init__(self, sigma_m: SigmaM, delta=200.0):
        self.sigma_m = sigma_m
        self.cosmo = sigma_m.cosmo
        self._pars = _tinker10_params(delta)

    def nu(self, M, z):
        return DELTA_C / self.sigma_m(M, z)

    def bias_at_nu(self, nu):
        A, a, B, b, C, c = self._pars
        return (1.0
                - A * nu ** a / (nu ** a + DELTA_C ** a)
                + B * nu ** b
                + C * nu ** c)

    def __call__(self, M, z):
        return self.bias_at_nu(self.nu(M, z))
