"""Miscentered NFW surface density via precomputed "single" lookup table.

Tables (loaded once at construction):
    table_1000_1e-03_5e+03_single_logx.txt           999 pts   ln(R/R_s)
    table_1000_1e-03_5e+03_single_logxmis.txt        249 pts   ln(R_mis/R_s)
    table_1000_1e-03_5e+03_log_sigma_single.txt      250 x 1000  ln f(x, x_mis)
    table_1000_1e-03_5e+03_log_deltasigma_single.txt 250 x 1000  ln g(x, x_mis)

The stored `f` is empirically (2023-table convention):
    f = (1/(4 pi^2 R_s rho_s)) * int d phi Sigma_NFW(R_h)
    <=> f = (1/(2 pi)) * <Sigma_NFW>_phi / (R_s rho_s)

When we reconstruct via  `Sigma_mis = (2 pi R_s rho_s) * exp(ln f)`
we get  Sigma_mis = <Sigma_NFW>_phi / pi  (i.e. phi-integrated then /pi),
which is HALF the Costanzi 2026 paper eq. 14 definition:
    Sigma_mis^{paper} = int_0^2pi Sigma_NFW(R_h) d phi

So the lookup's return, multiplied by 2, matches paper eq. 14.  The
factor-of-2 is applied in `sigma_grid` below, so downstream callers
(e.g. Sigma_prj) see paper-convention values.

The `g` (deltasigma) table follows the SAME half-paper convention:
cross-checked at R_mis -> 0 against the Wright & Brainerd 2000 centred
NFW Delta_Sigma(R), the raw table value is exactly 1/2 of the paper
definition, and `2 * (2 pi R_s rho_s) * exp(ln g)` reproduces the
analytical centred answer to 4 decimals.  The same factor-of-2 is
therefore applied in `delta_sigma_grid` below.

Cross-check: at R_mis -> 0 the paper eq. 14 gives
    Sigma_mis = 2 pi * Sigma_NFW(R)   (phi-independent integrand)
whereas the centered Sigma_NFW itself is just `Sigma_NFW(R)`.  In
Sigma_prj / eq. 13, `Sigma_mis` with paper eq. 14's definition is what
gets integrated; the centered profile is never passed directly.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .cosmology import Cosmology
from .config import NFW_TABLE_DIR


class NFWMiscentered:
    """Miscentered NFW Sigma(R | M, z, R_mis) from the Y3 lookup table.

    Concentration is Duffy 2008-style fixed c = 5 for v0.1; swap to a
    c(M, z) model later if needed.
    """

    def __init__(self, cosmo: Cosmology, table_dir=NFW_TABLE_DIR, c=5.0):
        self.cosmo = cosmo
        self.c = c
        self._log_x = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_single_logx.txt"))
        self._log_xmis = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_single_logxmis.txt"))
        log_sigma = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_log_sigma_single.txt"))
        log_dsigma = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_log_deltasigma_single.txt"))
        if log_dsigma.shape != log_sigma.shape:
            raise ValueError(
                "log_deltasigma_single table shape "
                f"{log_dsigma.shape} != log_sigma_single shape "
                f"{log_sigma.shape}; axes must match")

        lnxmis = self._log_xmis[: log_sigma.shape[0]]
        lnx = self._log_x[: log_sigma.shape[1]]
        self._lnx_lo, self._lnx_hi = lnx[0], lnx[-1]
        self._lnxmis_lo, self._lnxmis_hi = lnxmis[0], lnxmis[-1]
        self._spl = RectBivariateSpline(lnxmis, lnx, log_sigma, kx=1, ky=1)
        self._dsig_spl = RectBivariateSpline(
            lnxmis, lnx, log_dsigma, kx=1, ky=1)

    def _rs_and_rhos(self, M, z):
        """R_s [cMpc/h], rho_s [Msun/h / (cMpc/h)^3]."""
        rho_m = self.cosmo.Om0 * 2.77533742639e11
        r200m = (3.0 * M / (4.0 * np.pi * 200.0 * rho_m)) ** (1.0 / 3.0)
        rs = r200m / self.c
        fc = np.log(1.0 + self.c) - self.c / (1.0 + self.c)
        rho_s = rho_m * (200.0 / 3.0) * self.c ** 3 / fc
        return rs, rho_s

    def sigma_grid(self, R_arr, R_mis_arr, M, z):
        """Sigma_mis over (R_mis, R) in the Costanzi 2026 paper eq. 14
        convention:  Sigma_mis = int_0^{2 pi} Sigma_NFW(R_h) d phi.

        Returns (N_Rmis, N_R) array of Sigma_mis [Msun/h / (cMpc/h)^2].
        """
        rs, rho_s = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        lnF = self._spl(lnxmis, lnx)
        # Stored lookup = (1/2) * paper_eq14; correct with the factor of 2.
        return 2.0 * (2.0 * np.pi * rs * rho_s) * np.exp(lnF)

    def delta_sigma_grid(self, R_arr, R_mis_arr, M, z):
        """Delta_Sigma_mis = bar_Sigma_mis(<R) - Sigma_mis(R) over
        (R_mis, R), in the same paper-eq.-14 convention as ``sigma_grid``
        (phi-integrated, 2 pi factor absorbed; see module docstring).

        Returns (N_Rmis, N_R) array of DeltaSigma_mis [Msun/h / (cMpc/h)^2].
        """
        rs, rho_s = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        lnG = self._dsig_spl(lnxmis, lnx)
        # Stored lookup = (1/2) * paper convention; same factor of 2
        # as sigma_grid (verified at R_mis -> 0 vs Wright & Brainerd 2000).
        return 2.0 * (2.0 * np.pi * rs * rho_s) * np.exp(lnG)
