"""Miscentered NFW surface density via precomputed "single" lookup table.

Tables (loaded once at construction):
    table_1000_1e-03_5e+03_single_logx.txt           999 pts   ln(R/R_s)
    table_1000_1e-03_5e+03_single_logxmis.txt        249 pts   ln(R_mis/R_s)
    table_1000_1e-03_5e+03_log_sigma_single.txt        250 x 1000  ln f(x, x_mis)
    table_1000_1e-03_5e+03_deltasigma_signed_single.txt 250 x 1000  SIGNED g(x, x_mis)

The DeltaSigma table is SIGNED/linear (not log): DeltaSigma_mis(R|R_mis) is
negative for R_mis > R.  The legacy log_deltasigma table was exp()'d on read
(forced >= 0) and zeroed that negative lobe; see
data/nfw_off_center/make_signed_deltasigma_table.py.

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
        # SIGNED (linear) DeltaSigma table -- see
        # data/nfw_off_center/make_signed_deltasigma_table.py.  Replaces the
        # legacy log_deltasigma table, which was exp()'d on read and so could
        # only be >= 0; the true DeltaSigma_mis is NEGATIVE for R_mis > R
        # (halo center outside the aperture), and zeroing that lobe broke the
        # projection mean-field cancellation.  Stored signed, read WITHOUT exp.
        dsigma = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_deltasigma_signed_single.txt"))
        if dsigma.shape != log_sigma.shape:
            raise ValueError(
                "deltasigma_signed_single table shape "
                f"{dsigma.shape} != log_sigma_single shape "
                f"{log_sigma.shape}; axes must match")

        lnxmis = self._log_xmis[: log_sigma.shape[0]]
        lnx = self._log_x[: log_sigma.shape[1]]
        self._lnx_lo, self._lnx_hi = lnx[0], lnx[-1]
        self._lnxmis_lo, self._lnxmis_hi = lnxmis[0], lnxmis[-1]
        self._spl = RectBivariateSpline(lnxmis, lnx, log_sigma, kx=1, ky=1)
        self._dsig_spl = RectBivariateSpline(
            lnxmis, lnx, dsigma, kx=1, ky=1)   # signed values (no log)

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
        """DeltaSigma_mis(R, R_mis | M) in the C++ convention [Msun/h / pc^2].

        Returns a (N_Rmis, N_R) array.  Same ``2 * r_s * rho_eff * 1e-12``
        prefactor as ``sigma_grid`` — both tables are in the C++ kernel's
        natural units.
        """
        rs, rho_eff = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        G = self._dsig_spl(lnxmis, lnx)   # signed dimensionless DeltaSigma_mis
        norm = 2.0 * rs * rho_eff
        return norm * G * CMPCH2_TO_PC2   # NO exp: table is signed/linear
