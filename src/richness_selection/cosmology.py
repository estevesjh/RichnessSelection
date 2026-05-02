"""Cosmology wrapper on top of astropy.FlatLambdaCDM.

Caches distance grids once at construction so that chi(z) / D_A(z) /
dV/dOmega/dz calls are cheap interpolator lookups, mirroring the
`scaleShiftCosmo` idiom from haloModel.py.

Units throughout the package:
    distances  [cMpc/h]
    masses     [Msun/h]
    k          [h/Mpc]
    P(k)       [(Mpc/h)^3]
"""
from __future__ import annotations
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from .config import DEFAULT_COSMO_PARAMS

C_LIGHT = 2997.92458   # (c / 100 km/s) in cMpc/h


class Cosmology:
    """Frozen cosmology snapshot: use a new instance for each MCMC sample.

    Public attributes are *read-only* by convention; mutating them will
    silently invalidate the precomputed distance grids.
    """

    def __init__(self, Om0=None, Ob0=None, H0=None, ns=None, sigma8=None,
                 mnu=None, z_max=3.0, nz_grid=512):
        p = dict(DEFAULT_COSMO_PARAMS)
        for k, v in dict(Om0=Om0, Ob0=Ob0, H0=H0, ns=ns, sigma8=sigma8,
                         mnu=mnu).items():
            if v is not None:
                p[k] = v
        self.Om0 = p["Om0"]
        self.Ob0 = p["Ob0"]
        self.H0 = p["H0"]
        self.h = self.H0 / 100.0
        self.ns = p["ns"]
        self.sigma8 = p["sigma8"]
        self.mnu = p["mnu"]

        self._astropy = FlatLambdaCDM(H0=self.H0, Om0=self.Om0, Ob0=self.Ob0)

        self._z_grid = np.linspace(0.0, z_max, nz_grid)
        chi_mpc = self._astropy.comoving_distance(self._z_grid).to(u.Mpc).value
        self._chi_grid = chi_mpc * self.h   # cMpc/h

    @property
    def key(self):
        """Hashable tuple uniquely identifying this cosmology (for lru_cache)."""
        return (self.Om0, self.Ob0, self.H0, self.ns, self.sigma8, self.mnu)

    def E(self, z):
        z = np.asarray(z)
        a = 1.0 / (1.0 + z)
        return np.sqrt(self.Om0 / a**3 + 1.0 - self.Om0)

    def chi(self, z):
        """Comoving distance [cMpc/h]."""
        return np.interp(z, self._z_grid, self._chi_grid)

    def D_A(self, z):
        return self.chi(z) / (1.0 + np.asarray(z))

    def dV_dzdOm(self, z):
        """Comoving volume element dV/(dOmega dz) [ (cMpc/h)^3 ]."""
        return C_LIGHT * self.chi(z) ** 2 / self.E(z)

    def rho_m0(self):
        """Mean matter density today in Msun/h / (cMpc/h)^3."""
        rho_c0 = 2.77533742639e11
        return self.Om0 * rho_c0

    def __repr__(self):
        return (f"Cosmology(Om0={self.Om0:.4f}, Ob0={self.Ob0:.4f}, "
                f"H0={self.H0:.2f}, ns={self.ns:.3f}, "
                f"sigma8={self.sigma8:.3f}, mnu={self.mnu:.3f})")
