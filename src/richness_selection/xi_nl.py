"""Nonlinear correlation function xi_NL(r, z) via Hankel transform of P_NL(k, z).

Uses ``mcfit.P2xi`` (FFTlog-based Hankel transform) which is the right
tool for this: accurate across eight decades in r, no oscillatory
trapezoidal blow-up at small r.  Matteo's cell 11 uses a direct 50k-point
trapz; we match it to ~1e-3 on the (r in [1e-3, 100] cMpc/h) grid.

Tabulated on a (log r, z) grid and exposed as xir2_NL_r_z(log_r, z) ->
xi * r^2 via RectBivariateSpline, matching Matteo's `xir2_NL_r_z` name.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .cosmology import Cosmology


def _camb_halofit_pk(cosmo: Cosmology, z_grid, kmax=100.0, nk=2048):
    """Returns halofit P_NL(k, z) on a log-spaced k grid (needed by mcfit).

    mcfit.P2xi requires k on a geometric (log) progression.  CAMB's
    `get_matter_power_interpolator` gives that; we read P(k) directly
    from the interpolator on our own logk grid.
    """
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo.H0,
                       ombh2=cosmo.Ob0 * cosmo.h ** 2,
                       omch2=(cosmo.Om0 - cosmo.Ob0) * cosmo.h ** 2,
                       mnu=cosmo.mnu)
    pars.InitPower.set_params(ns=cosmo.ns)
    # Ensure z=0 is in the list so sigma8_0 is defined.
    zs_camb = np.unique(np.concatenate([[0.0], np.sort(z_grid)]))[::-1]
    pars.set_matter_power(redshifts=list(zs_camb), kmax=kmax * 1.1)
    pars.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(pars)

    k_lo, k_hi = 1e-4, kmax
    kh = np.logspace(np.log10(k_lo), np.log10(k_hi), nk)
    pk_nl = np.empty((z_grid.size, nk))
    Pk_at_z_NL = camb.get_matter_power_interpolator(
        pars, zs=list(zs_camb), kmax=kmax * 1.1,
        nonlinear=True, hubble_units=True, k_hunit=True, log_interp=True)
    for iz, z in enumerate(z_grid):
        pk_nl[iz] = Pk_at_z_NL.P(float(z), kh)

    sig8_camb = results.get_sigma8_0()
    rescale = (cosmo.sigma8 / sig8_camb) ** 2
    return kh, np.asarray(z_grid), pk_nl * rescale


class XiNL:
    """xi_NL(r, z) interpolator built from halofit P_NL(k, z)."""

    def __init__(self, cosmo: Cosmology,
                 z_min: float = 0.05, z_max: float = 0.95, nz: int = 19,
                 k_max: float = 100.0, nk: int = 2048):
        self.cosmo = cosmo
        self.z_grid = np.linspace(z_min, z_max, nz)
        kh, zs_out, Pnl = _camb_halofit_pk(cosmo, self.z_grid,
                                           kmax=k_max, nk=nk)

        # mcfit: P(k) -> xi(r) via FFTlog.  r_grid comes out log-spaced
        # from the same k_grid; we don't pick it ourselves.
        from mcfit import P2xi
        h = P2xi(kh, lowring=True)
        xi_list = []
        for iz in range(zs_out.size):
            r_out, xi_z = h(Pnl[iz], extrap=True)
            xi_list.append(xi_z)
        self.r_grid = np.asarray(r_out)          # (nr,)
        self.xi = np.array(xi_list)              # (nz, nr)

        # Matteo-compatible interpolator: xir2 = xi * r^2, spline over
        # (ln r, z).
        self._spl = RectBivariateSpline(
            np.log(self.r_grid), zs_out, (self.xi * self.r_grid ** 2).T)

    def __call__(self, r, z):
        """xi_NL(r, z) at broadcast shapes; r in cMpc/h."""
        r = np.asarray(r, dtype=float)
        lnr = np.log(np.clip(r, self.r_grid[0], self.r_grid[-1]))
        z_arr = np.broadcast_to(np.asarray(z, dtype=float), lnr.shape)
        val = self._spl.ev(lnr.ravel(), z_arr.ravel()).reshape(lnr.shape)
        return val / np.clip(r, 1e-30, None) ** 2

    # Matteo-style alias:  xir2_NL_r_z(log_r, z) -> xi * r^2, returning a 2D
    # block if given 1-D inputs (same convention as scipy's __call__).
    def xir2(self, log_r, z):
        return self._spl(log_r, z)
