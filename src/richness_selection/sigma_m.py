"""sigma(M, z): r.m.s. linear density fluctuation in a top-hat window.

    sigma^2(R, z) = (1 / 2 pi^2) int_0^infty dk k^2 P_lin(k, z) W_th(k R)^2
    W_th(x) = 3 (sin(x) - x cos(x)) / x^3
    R(M) = (3 M / (4 pi rho_m0))^(1/3)    [cMpc/h]

Cached over the same PkGrid instance: one SigmaM per PkGrid.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import simpson

from .pk import PkGrid

RHO_CRIT_0 = 2.77533742639e11   # Msun/h / (cMpc/h)^3 at H0 = 100 km/s/Mpc


def _top_hat_window(x):
    """W_th(x) = 3 (sin x - x cos x) / x^3, with Taylor series near 0."""
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = 1.0 - xs * xs / 10.0 + xs ** 4 / 280.0
    xl = x[~small]
    out[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / xl ** 3
    return out


class SigmaM:
    """sigma(M, z) by direct Simpson integration over the PkGrid k-nodes."""

    def __init__(self, pk: PkGrid):
        self.pk = pk
        self.cosmo = pk.cosmo
        self.rho_m0 = self.cosmo.Om0 * RHO_CRIT_0   # Msun/h per (cMpc/h)^3

        # Precompute k, P on the PkGrid z-grid; evaluate sigma on an (M, z)
        # table so later calls are a 2-D interpolation, not a fresh quad.
        self._k = pk.k
        self._logk = np.log(self._k)
        self._k2 = self._k ** 2

        # Tabulate sigma(M, z) on the PkGrid z-nodes.
        lnM_grid = np.linspace(np.log(10 ** 10.0), np.log(10 ** 16.0), 200)
        self._lnM = lnM_grid
        M_arr = np.exp(lnM_grid)
        R_arr = self.radius_of_mass(M_arr)          # (NM,)

        z_grid = pk.z
        sigma = np.empty((z_grid.size, lnM_grid.size))
        # integrand in d ln k: k^3 P(k) W(kR)^2 / (2 pi^2)
        kR = self._k[None, :] * R_arr[:, None]   # (NM, nk)
        W_sq = _top_hat_window(kR) ** 2           # (NM, nk)
        for iz in range(z_grid.size):
            Pk = pk.P[iz]                                # (nk,)
            integrand = self._k[None, :] ** 3 * Pk[None, :] * W_sq
            sigma2 = simpson(integrand, x=self._logk, axis=1) / (2.0 * np.pi ** 2)
            sigma[iz] = np.sqrt(sigma2)
        self._sigma_tab = sigma

        from scipy.interpolate import RectBivariateSpline
        self._spl = RectBivariateSpline(z_grid, lnM_grid, sigma, kx=3, ky=3)

    def radius_of_mass(self, M):
        """Lagrangian radius R(M) = (3M / 4 pi rho_m0)^{1/3}  [cMpc/h]."""
        M = np.asarray(M, dtype=float)
        return (3.0 * M / (4.0 * np.pi * self.rho_m0)) ** (1.0 / 3.0)

    def __call__(self, M, z):
        """sigma(M, z).  Any broadcast-compatible shapes."""
        M, z = np.broadcast_arrays(np.asarray(M, dtype=float),
                                   np.asarray(z, dtype=float))
        shape = M.shape
        lnM_flat = np.log(M).ravel()
        z_flat = z.ravel()
        out = self._spl.ev(z_flat, lnM_flat)
        return out.reshape(shape) if shape else float(out[0])

    def dln_sigma_dlnM(self, M, z, h=1e-3):
        """d ln sigma / d ln M via centred finite difference in lnM."""
        M = np.asarray(M, dtype=float)
        s_up = self(M * np.exp(h), z)
        s_dn = self(M * np.exp(-h), z)
        return (np.log(s_up) - np.log(s_dn)) / (2.0 * h)
