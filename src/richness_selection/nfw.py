"""Miscentered NFW surface density via precomputed "single" lookup table.

Tables (loaded once at construction):
    table_1000_1e-03_5e+03_single_logx.txt           999 pts   ln(R/R_s)
    table_1000_1e-03_5e+03_single_logxmis.txt        249 pts   ln(R_mis/R_s)
    table_1000_1e-03_5e+03_log_sigma_single.txt      250 x 1000  ln f(x, x_mis)

The shipped ``log_deltasigma_single`` table is deliberately NOT read:
its ``ln g`` storage cannot represent the negative branch
``DeltaSigma_mis(R < R_mis) < 0`` and floors it to ~0, which inflates
the two-halo cl piece by ~1.5x and breaks the uniform-field rnd
cancellation.  ``_dsig_spl`` is instead the *signed* excess
reconstructed at construction from the Sigma table (per x_mis row,
``bar-Sigma(<R) - Sigma(R)``), splined in linear space -- see
``_build_signed_dsigma_spline``.

C++ convention (authoritative: y3_cluster_cpp ``NFW_SIGMA_MIS`` and
``NFW_DSIGMA_MIS``)
----------------------------------------------------------------------

The stored tables ``f`` and ``g`` are the same ones used by
``y3_cluster_cpp``.  The C++ reconstruction recipe (mirrored here) is::

    r_200  = cbrt( 3 M / (800 pi rho_crit) )        [cMpc/h]
    r_s    = r_200 / c,                c = 4                    (default)
    delta_c = (200 c^3 / 3) / (ln(1+c) - c/(1+c))
    rho_eff = delta_c * rho_crit * Omega_m                      (cMpc/h units)

    Sigma_mis      = 2 * r_s * rho_eff * exp(ln f) * 1e-12      [Msun/h / pc^2]
    DeltaSigma_mis = 2 * r_s * rho_eff * exp(ln g) * 1e-12      [Msun/h / pc^2]

The ``1e-12`` factor converts ``Msun/h / (cMpc/h)^2`` (the natural
units of ``r_s * rho_eff``) into ``Msun/h / pc^2`` so downstream code
carries the C++ lensing-observable units directly.

Divergence from the previous (paper-Eq.-14) convention
------------------------------------------------------

- ``c`` default was 5; the C++ side uses 4.
- ``r_200`` was ``r_200m`` via ``rho_m``; C++ uses ``r_200c`` via ``rho_crit``.
- ``Sigma_mis`` had a ``2 * (2 pi r_s rho_s)`` prefactor (pair of factors
  chosen to land on Costanzi 2026 Eq. 14 / Wright & Brainerd 2000 convention);
  C++ uses ``2 * r_s * rho_eff * 1e-12`` — no ``2 pi``, natural C++ units.

The two conventions are related by a mass-dependent rescaling (different
``r_s`` *and* a per-call constant), so callers are *not* isomorphic up to
an overall constant. Regression goldens changed when this module was
switched over; see ``docs/sigma_prj_refactor.md`` Section 2.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .cosmology import Cosmology
from .config import NFW_TABLE_DIR


# rho_crit,0 at H0 = 100 km/s/Mpc, in Msun/h / (cMpc/h)^3.
RHO_CRIT_0 = 2.77533742639e11

# (cMpc/h)^-2 -> pc^-2  conversion.  Natural (r_s * rho_eff) units are
# Msun/h / (cMpc/h)^2; multiply by (Mpc/pc)^-2 = 1e-12 to get Msun/h/pc^2.
CMPCH2_TO_PC2 = 1.0e-12


def wright_brainerd_g(x, eps=1e-9):
    """Wright & Brainerd (2000) centered NFW excess kernel g(x):
    ``DeltaSigma_cen = r_s rho_s g(x)``.  Analytic reference for the
    signed-table reconstruction (ported from CLensPy
    ``clenspy.halo.nfw.NfwProfile._gNfw``)."""
    x = np.array(x, dtype=float)
    res = np.empty_like(x)
    mask_c = np.abs(x - 1.0) <= eps
    res[mask_c] = 10.0 / 3.0 + 4.0 * np.log(0.5)

    mask_l = x < 1.0 - eps
    xl = x[mask_l]
    s = np.sqrt(1.0 - xl ** 2)
    atanh = np.arctanh(s / (1.0 + xl))
    res[mask_l] = (8.0 * atanh / (xl ** 2 * s)
                   + 4.0 / xl ** 2 * np.log(xl / 2.0)
                   - 2.0 / (xl ** 2 - 1.0)
                   + 4.0 * atanh / ((xl ** 2 - 1.0) * s))

    mask_g = x > 1.0 + eps
    xg = x[mask_g]
    s = np.sqrt(xg ** 2 - 1.0)
    atan = np.arctan(s / (1.0 + xg))
    res[mask_g] = (8.0 * atan / (xg ** 2 * s)
                   + 4.0 / xg ** 2 * np.log(xg / 2.0)
                   - 2.0 / (xg ** 2 - 1.0)
                   + 4.0 * atan / ((xg ** 2 - 1.0) ** 1.5))
    return res


class NFWMiscentered:
    """Miscentered NFW Sigma(R | M, z, R_mis) from the Y3 lookup table.

    C++ convention (``y3_cluster_cpp::NFW_SIGMA_MIS``):
    ``c = 4`` default, ``r_200`` via ``rho_crit``, output in
    ``Msun/h / pc^2``.  See module docstring.
    """

    def __init__(self, cosmo: Cosmology, table_dir=NFW_TABLE_DIR,
                 c: float = 4.0, rho_crit: float = RHO_CRIT_0,
                 rho_mult: float | None = None):
        self.cosmo = cosmo
        self.c = float(c)
        self._rho_crit = float(rho_crit)
        # Default rho_mult = Omega_m (matches C++ ``rho_mult = omega_m``).
        self._rho_mult = (float(rho_mult) if rho_mult is not None
                          else float(cosmo.Om0))
        self._log_x = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_single_logx.txt"))
        self._log_xmis = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_single_logxmis.txt"))
        log_sigma = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_log_sigma_single.txt"))

        lnxmis = self._log_xmis[: log_sigma.shape[0]]
        lnx = self._log_x[: log_sigma.shape[1]]
        self._lnx_lo, self._lnx_hi = lnx[0], lnx[-1]
        self._lnxmis_lo, self._lnxmis_hi = lnxmis[0], lnxmis[-1]
        self._spl = RectBivariateSpline(lnxmis, lnx, log_sigma, kx=1, ky=1)
        self._dsig_spl = self._build_signed_dsigma_spline(lnxmis)

    def _build_signed_dsigma_spline(self, lnxmis, n_dense: int = 4000):
        """Signed DeltaSigma_mis lookup, reconstructed from the Sigma
        table (the shipped ``log_deltasigma_single`` table is NOT used:
        its ``ln g`` storage floors the negative branch
        ``DeltaSigma_mis(R < R_mis) < 0`` to ~0, which inflates the
        two-halo cl piece by ~1.5x and breaks the uniform-field rnd
        cancellation).

        Per x_mis row:  gs(x | x_mis) = bar-f(<x) - f(x), with
        ``bar-f(<x) = (2/x^2) int_0^x x' f(x') dx'`` accumulated by
        trapezoid on a dense ln-x grid (``x' f dx' = x'^2 f dlnx'``;
        flat inner-disk cap below the table floor, x^2-suppressed).
        Returned as a *linear*-space bilinear spline on
        ``(lnxmis, lnx_dense)`` -- one-time ~ms cost, per-eval cost
        identical to the old lookup.
        """
        lnx_g = np.linspace(self._lnx_lo, self._lnx_hi, int(n_dense))
        f = np.exp(self._spl(lnxmis, lnx_g))         # (Nxmis, Nx)
        x_g = np.exp(lnx_g)
        integ = f * x_g[None, :] ** 2
        K = np.concatenate([
            np.zeros((f.shape[0], 1)),
            np.cumsum(0.5 * (integ[:, 1:] + integ[:, :-1])
                      * np.diff(lnx_g)[None, :], axis=1)], axis=1)
        K += f[:, :1] * x_g[0] ** 2 / 2.0            # inner-disk cap
        gs = 2.0 * K / x_g[None, :] ** 2 - f         # signed excess
        return RectBivariateSpline(lnxmis, lnx_g, gs, kx=1, ky=1)

    def _rs_and_rhos(self, M, z):
        """``(r_s, rho_eff)`` in the C++ convention.

        ``r_s`` [cMpc/h], ``rho_eff = delta_c * rho_crit * rho_mult``
        [Msun/h / (cMpc/h)^3].  The ``_rho_s`` name used in older call
        sites now refers to ``rho_eff`` (same object, C++-recipe value).
        """
        c = self.c
        rhoc = self._rho_crit
        r_200 = np.cbrt(3.0 * M / (800.0 * np.pi * rhoc))
        rs = r_200 / c
        fc = np.log(1.0 + c) - c / (1.0 + c)
        delta_c = (200.0 * c ** 3 / 3.0) / fc
        return rs, delta_c * rhoc * self._rho_mult

    def sigma_grid(self, R_arr, R_mis_arr, M, z):
        """Sigma_mis(R, R_mis | M) in the C++ convention [Msun/h / pc^2].

        Returns a (N_Rmis, N_R) array.
        """
        rs, rho_eff = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        lnF = self._spl(lnxmis, lnx)
        norm = 2.0 * rs * rho_eff
        return norm * np.exp(lnF) * CMPCH2_TO_PC2

    def delta_sigma_grid(self, R_arr, R_mis_arr, M, z):
        """Signed DeltaSigma_mis(R, R_mis | M) [Msun/h / pc^2].

        Returns a (N_Rmis, N_R) array.  Same ``2 * r_s * rho_eff * 1e-12``
        prefactor as ``sigma_grid``.  Values come from the signed
        reconstruction (``_build_signed_dsigma_spline``): negative for
        R < R_mis, Wright & Brainerd centered excess at R_mis -> 0.
        """
        rs, rho_eff = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        gs = self._dsig_spl(lnxmis, lnx)
        norm = 2.0 * rs * rho_eff
        return norm * gs * CMPCH2_TO_PC2
