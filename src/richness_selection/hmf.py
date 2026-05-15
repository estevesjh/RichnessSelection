"""Tinker 2008 halo mass function for Delta = 200 (mean density).

Paper: Tinker+ 2008, ApJ 688, 709.  Table 2 (z=0, Delta=200m):
    A = 0.186
    a = 1.47
    b = 2.57
    c = 1.19

HMF form:
    dn/dlnM = rho_m0 / M * f(sigma) * |d ln sigma / d ln M|
    f(sigma) = A * ((sigma/b)^(-a) + 1) * exp(-c / sigma^2)

Table 4 z-evolution (also Tinker+ 2008):
    A(z) = A_0 * (1+z)^-0.14
    a(z) = a_0 * (1+z)^-0.06
    b(z) = b_0 * (1+z)^-alpha(Delta)
    c(z) = c_0
with alpha(Delta) = 10^{-(0.75/log10(Delta/75))^{1.2}}; at Delta=200
alpha ~ 1.04.  Enabled by default (`z_evolution=True`) to stay apples-to-
apples with CosmoSIS mf_tinker (cosmosis-standard-library) and the paper;
pass z_evolution=False to recover the z=0-only v0.1 behaviour.
"""
from __future__ import annotations
import numpy as np

from .sigma_m import SigmaM

TINKER08_DELTA200M = dict(A=0.186, a=1.47, b=2.57, c=1.19)


def _alpha_tinker(delta):
    """Table 4 exponent for the b(z) scaling."""
    return 10.0 ** (-((0.75 / np.log10(delta / 75.0)) ** 1.2))


class HMF:
    """Tinker 2008 halo mass function n(M, z) [ (Msun/h)^-1 (cMpc/h)^-3 ]."""

    def __init__(self, sigma_m: SigmaM, delta=200, z_evolution: bool = True):
        if delta != 200:
            raise NotImplementedError("v0.1 supports Delta=200m only")
        self.sigma_m = sigma_m
        self.cosmo = sigma_m.cosmo
        self.rho_m0 = sigma_m.rho_m0
        self._params = dict(TINKER08_DELTA200M)
        self._z_evolution = bool(z_evolution)
        self._alpha = _alpha_tinker(delta)

    def _params_at_z(self, z):
        """Return (A, a, b, c) at redshift z per Tinker 2008 Table 4."""
        A0 = self._params["A"]; a0 = self._params["a"]
        b0 = self._params["b"]; c0 = self._params["c"]
        if not self._z_evolution:
            return A0, a0, b0, c0
        z = np.asarray(z, dtype=float)
        one_plus_z = 1.0 + z
        A = A0 * one_plus_z ** (-0.14)
        a = a0 * one_plus_z ** (-0.06)
        b = b0 * one_plus_z ** (-self._alpha)
        return A, a, b, c0

    def f_sigma(self, sigma, z=0.0):
        A, a, b, c = self._params_at_z(z)
        return A * ((sigma / b) ** (-a) + 1.0) * np.exp(-c / sigma ** 2)

    def dn_dlnM(self, M, z):
        """dn / d ln M  [ (cMpc/h)^-3 ]."""
        M = np.asarray(M, dtype=float)
        sig = self.sigma_m(M, z)
        dlnsig_dlnM = self.sigma_m.dln_sigma_dlnM(M, z)
        return (self.rho_m0 / M) * self.f_sigma(sig, z) * np.abs(dlnsig_dlnM)

    def __call__(self, M, z):
        """Number density dn / dM  [ (Msun/h)^-1 (cMpc/h)^-3 ]."""
        M = np.asarray(M, dtype=float)
        return self.dn_dlnM(M, z) / M
