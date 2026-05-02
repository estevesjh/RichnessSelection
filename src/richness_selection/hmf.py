"""Tinker 2008 halo mass function for Delta = 200 (mean density).

Paper: Tinker+ 2008, ApJ 688, 709.  Table 2, Delta = 200:
    A = 0.186
    a = 1.47
    b = 2.57
    c = 1.19

HMF form:
    dn/dlnM = rho_m0 / M * f(sigma) * |d ln sigma / d ln M|
    f(sigma) = A * ((sigma/b)^(-a) + 1) * exp(-c / sigma^2)

v0.1 uses the z=0 Delta=200m constants at every redshift (no z-evolution of
A, a, b, c).  If y3_buzzard needs Tinker 2008 Table 4's (1+z)^{-0.14} etc.
scaling later, edit _params below.
"""
from __future__ import annotations
import numpy as np

from .sigma_m import SigmaM

TINKER08_DELTA200M = dict(A=0.186, a=1.47, b=2.57, c=1.19)


class HMF:
    """Tinker 2008 halo mass function n(M, z) [ (Msun/h)^-1 (cMpc/h)^-3 ]."""

    def __init__(self, sigma_m: SigmaM, delta=200):
        if delta != 200:
            raise NotImplementedError("v0.1 supports Delta=200m only")
        self.sigma_m = sigma_m
        self.cosmo = sigma_m.cosmo
        self.rho_m0 = sigma_m.rho_m0
        self._params = dict(TINKER08_DELTA200M)

    def f_sigma(self, sigma):
        A = self._params["A"]
        a = self._params["a"]
        b = self._params["b"]
        c = self._params["c"]
        return A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma ** 2)

    def dn_dlnM(self, M, z):
        """dn / d ln M  [ (cMpc/h)^-3 ]."""
        M = np.asarray(M, dtype=float)
        sig = self.sigma_m(M, z)
        dlnsig_dlnM = self.sigma_m.dln_sigma_dlnM(M, z)
        return (self.rho_m0 / M) * self.f_sigma(sig) * np.abs(dlnsig_dlnM)

    def __call__(self, M, z):
        """Number density dn / dM  [ (Msun/h)^-1 (cMpc/h)^-3 ]."""
        M = np.asarray(M, dtype=float)
        return self.dn_dlnM(M, z) / M
