"""Projection-effects / selection-bias pipeline (Costanzi 2026).

Ports Matteo's cell 16 verbatim, with two known-undefined constants
(`inter_boost_bias`, `slope_boost_bias`) exposed as attributes of the
class with `inter_boost_bias=1.0`, `slope_boost_bias=0.0` as the
literal defaults corresponding to ``boost_bias == 1`` (the default value
in Matteo's `bar_delta_prj_Beff` signature).

Pipeline (Costanzi 2026, Matteo cell 16):

  bar_delta_prj_bkg(ltr, zcl)       -- LoS mass integral of projected richness
  numerator2(lob, zcl)              -- LoS mass integral of (lob-ltr)-style kernel
  eff_bias_ltr(ltr, zcl)            -- Tinker-bias weighted by HOD P(ltr|M,z)
  Delta_prj_lob_ltr(lob, ltr) = lob - ltr
  boost_bias(lob, ltr, zcl)         -- linear model in (Delta - <Delta>)/<Delta>
  b_sel_lob_ltr_in(ltr, zcl, lob)   -- intrinsic (theta -> 0) bias
  b_sel_lob_ltr_theta(theta, ...)   -- sigmoid interpolation in theta

Pure-halo bias b(M, z) lives in ``bias.py`` -- do not pollute this file
with it.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import bisect

from .cosmology import Cosmology
from .pk import PkGrid
from .hmf import HMF
from .bias import Bias
from .mor import MOR
from .geometry import (R_lambda, theta_lambda,
                       Omega_halos as _Omega_halos,
                       f_area as _f_area,
                       area_overlap)
from .photoz import w_z, sigma_z, zmin4zkernel, zmax4zkernel
from .gl import gl_nodes
from .config import DEFAULT_GRID, GridConfig


class SelBias:
    """Projection-effects / selection-bias orchestrator (Matteo cell 16)."""

    # Matteo cell 16 globals promoted to class attributes:
    min_mass4integral: float = 1.0e13
    damping_sigmoid: float = 2.5
    n_th_inf: float = 0.0
    n_th_sup: float = 1.0
    exclusion: bool = True

    def __init__(self, cosmo: Cosmology, pk: PkGrid, hmf: HMF, bias: Bias,
                 mor: MOR, xi_nl, grid: GridConfig = DEFAULT_GRID,
                 # Boost-bias parameters (cell 16 references them but does
                 # not define them).  With the defaults below, boost_bias
                 # collapses to 1, matching the `boost_bias=1.0` default of
                 # Matteo's `bar_delta_prj_Beff` signature.
                 inter_boost_bias: float = 1.0,
                 slope_boost_bias: float = 0.0):
        self.cosmo = cosmo
        self.pk = pk
        self.hmf = hmf
        self.bias = bias
        self.mor = mor
        self.xi_NL = xi_nl
        self.grid = grid
        self.inter_boost_bias = inter_boost_bias
        self.slope_boost_bias = slope_boost_bias
        # Per-sample cache (cleared when a new (cosmo, pk) is wired in --
        # user responsibility).  Prevents double-integration when the
        # same precompute is asked for many ltr nodes.
        self._cache: dict = {}

    # ---------------- simple helpers --------------------------------------

    def _theta_lob(self, lob, zob):
        return theta_lambda(lob, zob, self.cosmo)

    # ---------------- LoS tracer-redshift grid (Matteo ztr_grid) ----------

    def _ztr_grid(self, zcl, n_los: int = 50):
        """Concatenated foreground + background tracer-z grid.

        Reproduces Matteo's bisect-based construction in `int_z_M_dV_n_bias_xi`
        / `bar_delta_prj_Beff` / `numerator2`.
        """
        # Foreground
        z_min = bisect(zmin4zkernel, -2.0, 2.0, args=(zcl,))
        z_max = zcl - 1.0e-5
        chi_z_cl = float(self.cosmo.chi(zcl))
        dis_max = chi_z_cl - float(self.cosmo.chi(z_min))
        dis_min = chi_z_cl - float(self.cosmo.chi(z_max))
        dis = 10.0 ** np.linspace(np.log10(dis_min), np.log10(dis_max), n_los)
        ztr_fg = self._z_of_chi(chi_z_cl - dis)[::-1]
        # Background
        z_max = bisect(zmax4zkernel, -2.0, 2.0, args=(zcl,))
        z_min = zcl + 1.0e-5
        dis_min = float(self.cosmo.chi(z_min)) - chi_z_cl
        dis_max = float(self.cosmo.chi(z_max)) - chi_z_cl
        dis = 10.0 ** np.linspace(np.log10(dis_min), np.log10(dis_max), n_los)
        ztr_bg = self._z_of_chi(chi_z_cl + dis)
        return np.concatenate([ztr_fg, ztr_bg])

    def _z_of_chi(self, chi):
        """Inverse of cosmo.chi(z) via monotone interpolation on the same grid."""
        zs, chi_h = self.cosmo._chi_table if hasattr(self.cosmo, "_chi_table") \
                    else (self.cosmo._z_grid, self.cosmo._chi_grid)
        return np.interp(chi, chi_h, zs)

    # ---------------- Step: eff_bias_ltr (cell 18) -----------------------

    def eff_bias_ltr(self, ltr, zcl):
        """<b_halo>_{P(ltr | M, z_cl)} (cell 18)."""
        key = ("eff_bias_ltr", float(ltr), float(zcl))
        if key in self._cache:
            return self._cache[key]
        m_grid = 10.0 ** np.linspace(np.log10(self.min_mass4integral), 15.5, 100)
        hmf = self.hmf(m_grid, zcl)
        P = self.mor.pdf(np.array([float(ltr)])[:, None],
                         m_grid[None, :], zcl).ravel()
        b_halo = self.bias(m_grid, zcl)
        norm = np.trapz(hmf * P * m_grid, np.log(m_grid))
        num = np.trapz(b_halo * hmf * P * m_grid, np.log(m_grid))
        val = float(num / norm) if norm > 0 else float("nan")
        self._cache[key] = val
        return val

    # ---------------- bar_delta_prj_bkg (cell 16) ------------------------

    def bar_delta_prj_bkg(self, lob, zcl, boost_bias: float = 1.0):
        """LoS-integrated projected-richness background (cell 16)."""
        key = ("bar_delta_prj_bkg", float(lob), float(zcl), float(boost_bias))
        if key in self._cache:
            return self._cache[key]
        ztr_grid = self._ztr_grid(zcl)
        bias_cl = self.eff_bias_ltr(lob, zcl) * boost_bias
        intoverM = np.array([self._int_over_mass2(zcl, lob, ztr_grid[i], bias_cl)
                             for i in range(ztr_grid.size)])
        yarray = (self.cosmo.dV_dzdOm(ztr_grid)
                  * w_z(ztr_grid, zcl)
                  * intoverM)
        integral = float(np.trapz(yarray, ztr_grid, axis=0))
        self._cache[key] = integral
        return integral

    def _int_over_mass2(self, zcl, lob, ztr, bias_cl):
        m_grid = 10.0 ** np.linspace(np.log10(self.min_mass4integral), 15.5, 50)
        hmf = self.hmf(m_grid, ztr)
        bias_h = self.bias(m_grid, ztr)
        yarray = (m_grid * hmf
                  * self._int_over_lambda2(lob, zcl, ztr, m_grid,
                                           bias_cl, bias_h))
        return float(np.trapz(yarray, np.log(m_grid), axis=0))

    def _int_over_lambda2(self, lob, zcl, z, M, bias_cl, bias_h,
                          ltr_n: int = 100):
        ltr_grid = np.linspace(1e-10, lob, ltr_n)
        # Omega_halos * f_area, shape (Nltr,)
        Omf = (_Omega_halos(lob, zcl, ltr_grid, z, self.cosmo)
               * _f_area(lob, zcl, ltr_grid, z, self.cosmo))
        # pdf(ltr, M) broadcast to (Nltr, NM)
        p = self.mor.pdf(ltr_grid[:, None], M[None, :], z)
        # Clustering correction via int_over_phi_Beff (mean field)
        phi_term = self._int_over_phi_Beff(lob, zcl, z, ltr_grid,
                                           bias_cl, bias_h)  # (Nltr, NM)
        yarray = p * ltr_grid[:, None] * (Omf[:, None]
                                          + 2.0 * np.pi * phi_term)
        return np.trapz(yarray, ltr_grid, axis=0)

    def _int_over_phi_Beff(self, lob, zcl, z, ltr_grid, bias_cl, bias_h,
                           n_theta: int = 50):
        """Angular integral (cell 18 `int_over_phi_Beff`)."""
        theta_ltr = R_lambda(ltr_grid) * (1.0 + z) / self.cosmo.chi(z)
        theta_lob = R_lambda(lob) * (1.0 + zcl) / self.cosmo.chi(zcl)
        # theta_grid shape (n_theta, n_ltr)
        theta_grid = np.geomspace(1e-6, theta_lob + theta_ltr, n_theta)
        chi_cl = float(self.cosmo.chi(zcl))
        chi_z = float(self.cosmo.chi(z))
        dis = np.sqrt(chi_cl ** 2 + chi_z ** 2
                      - 2.0 * chi_cl * chi_z * np.cos(theta_grid))
        xi_vals = self.xi_NL(dis.ravel(), zcl).reshape(dis.shape)
        # Sigmoid scale-dependent selection bias in theta (Matteo cell 18)
        b0, b1, b2 = 2.0, 0.5, 0.75
        theta_1Mpc = R_lambda(lob) / chi_cl  # NB: Matteo uses chi(zob)
        x0 = b0 * theta_1Mpc
        k_s = b1 * 10.0 / x0
        eff_bias_lobztr_theta = (
            bias_cl * b2
            + (bias_cl - bias_cl * b2)
            / (1.0 + np.exp(-k_s * (theta_grid - x0))))
        # Shape broadcast to (n_theta, n_ltr, n_M)
        bM_bsel_xi = (bias_h[None, None, :]
                      * eff_bias_lobztr_theta[:, :, None]
                      * xi_vals[:, :, None])
        if self.exclusion:
            mask = dis < R_lambda(lob) * (1.0 + zcl)
            bM_bsel_xi[mask, :] = -1.0
        bM_bsel_xi = np.clip(bM_bsel_xi, -1.0, None)
        yarray = (np.sin(theta_grid)[:, :, None]
                  * (1.0 + bM_bsel_xi)
                  * area_overlap(theta_grid, theta_lob, theta_ltr)[:, :, None])
        return np.trapz(yarray, theta_grid[:, :, None], axis=0)

    # ---------------- numerator2 (cell 16) -------------------------------

    def numerator2(self, lob, zcl):
        """Two-halo projection normalisation (cell 16 `numerator2`)."""
        key = ("numerator2", float(lob), float(zcl))
        if key in self._cache:
            return self._cache[key]
        ztr_grid = self._ztr_grid(zcl)
        intoverM = np.array([self._int_over_mass4(zcl, lob, ztr_grid[i])
                             for i in range(ztr_grid.size)])
        yarray = (self.cosmo.dV_dzdOm(ztr_grid)
                  * w_z(ztr_grid, zcl)
                  * intoverM)
        integral = float(np.trapz(yarray, ztr_grid, axis=0))
        self._cache[key] = integral
        return integral

    def _int_over_mass4(self, zcl, lob, ztr):
        m_grid = 10.0 ** np.linspace(np.log10(self.min_mass4integral), 15.5, 100)
        hmf = self.hmf(m_grid, ztr)
        bias_h = self.bias(m_grid, ztr)
        yarray = (m_grid * bias_h * hmf
                  * self._int_over_lambda4(lob, zcl, ztr, m_grid))
        return float(np.trapz(yarray, np.log(m_grid), axis=0))

    def _int_over_lambda4(self, lob, zcl, z, M, ltr_n: int = 100):
        ltr_grid = np.linspace(1e-10, lob, ltr_n)
        Om_f = ltr_grid * 2.0 * np.pi * self._int_over_phi4(lob, zcl, z, ltr_grid)
        p = self.mor.pdf(ltr_grid[:, None], M[None, :], z)
        yarray = Om_f[:, None] * p
        return np.trapz(yarray, ltr_grid, axis=0)

    def _int_over_phi4(self, lob, zcl, z, ltr_grid, n_theta: int = 50):
        theta_ltr = R_lambda(ltr_grid) * (1.0 + z) / self.cosmo.chi(z)
        theta_lob = R_lambda(lob) * (1.0 + zcl) / self.cosmo.chi(zcl)
        theta_grid = np.geomspace(1e-6, theta_lob + theta_ltr, n_theta)

        x0 = 0.5 * (self.n_th_inf + self.n_th_sup) * theta_lob
        k_s = self.damping_sigmoid / ((self.n_th_sup - self.n_th_inf) * theta_lob)

        chi_cl = float(self.cosmo.chi(zcl))
        chi_z = float(self.cosmo.chi(z))
        dis = np.sqrt(chi_cl ** 2 + chi_z ** 2
                      - 2.0 * chi_cl * chi_z * np.cos(theta_grid))
        xi_vals = self.xi_NL(dis.ravel(), zcl).reshape(dis.shape)
        if self.exclusion:
            mask = dis < R_lambda(lob) * (1.0 + zcl)
            xi_vals = np.where(mask, 0.0, xi_vals)
        yarray = (np.sin(theta_grid)
                  * (1.0 - 1.0 / (1.0 + np.exp(-k_s * (theta_grid - x0))))
                  * xi_vals
                  * area_overlap(theta_grid, theta_lob, theta_ltr))
        return np.trapz(yarray, theta_grid, axis=0)

    # ---------------- intrinsic + scale-dependent bias (cell 16) --------

    @staticmethod
    def Delta_prj_lob_ltr(lob, ltr):
        return float(lob) - float(ltr)

    def boost_bias(self, lob, ltr, zcl):
        """Matteo cell 16 linear model:
            boost = inter + slope * (Delta - <Delta>) / <Delta>
        With defaults (1.0, 0.0), boost == 1.
        """
        DPRJ = self.bar_delta_prj_Beff(lob, zcl)
        Delta = self.Delta_prj_lob_ltr(lob, ltr)
        return (self.inter_boost_bias
                + self.slope_boost_bias * (Delta - DPRJ) / DPRJ
                if DPRJ != 0.0 else self.inter_boost_bias)

    def bar_delta_prj_Beff(self, lob, zcl):
        """Alias: Matteo uses this name in cell 16, identical to bar_delta_prj_bkg."""
        return self.bar_delta_prj_bkg(lob, zcl, boost_bias=1.0)

    def b_sel_lob_ltr_in(self, ltr, zcl, lob):
        """Intrinsic (theta -> 0) selection bias (cell 16)."""
        DPRJ = self.bar_delta_prj_Beff(lob, zcl)
        boost = (self.inter_boost_bias
                 + self.slope_boost_bias
                 * (self.Delta_prj_lob_ltr(lob, ltr) - DPRJ) / DPRJ
                 if DPRJ != 0.0 else self.inter_boost_bias)
        barDeltaPRJ_BKG = self.bar_delta_prj_bkg(ltr, zcl, boost_bias=boost)
        Delta = self.Delta_prj_lob_ltr(lob, ltr)
        numerator = self.numerator2(ltr, zcl)
        return (Delta - barDeltaPRJ_BKG) / numerator

    def b_sel_lob_ltr_theta(self, theta, ltr, zcl, lob,
                            damping: float = None):
        """Scale-dependent selection bias b_sel(theta) (cell 16).

        Sigmoid interpolation between b_sel_in (theta -> 0) and
        bias_eff = eff_bias_ltr(ltr) * boost_bias (theta -> inf).
        """
        if damping is None:
            damping = self.damping_sigmoid
        theta = np.asarray(theta, dtype=float)
        theta_lob = float(self._theta_lob(lob, zcl))

        b_in = self.b_sel_lob_ltr_in(ltr, zcl, lob)
        DPRJ = self.bar_delta_prj_Beff(lob, zcl)
        boost = (self.inter_boost_bias
                 + self.slope_boost_bias
                 * (self.Delta_prj_lob_ltr(lob, ltr) - DPRJ) / DPRJ
                 if DPRJ != 0.0 else self.inter_boost_bias)
        bias_eff = self.eff_bias_ltr(ltr, zcl) * boost

        x0 = 0.5 * (self.n_th_inf + self.n_th_sup) * theta_lob
        k = damping / ((self.n_th_sup - self.n_th_inf) * theta_lob)
        return b_in + (bias_eff - b_in) / (1.0 + np.exp(-k * (theta - x0)))

    # -- convenience wrapper matching old fast-GL API names ---------------

    def b_sel_of_theta(self, theta, lob, zob, ltr, precomp=None):
        """Back-compat: same signature as the old fast-GL module."""
        return self.b_sel_lob_ltr_theta(theta, ltr, zob, lob)

    def b_sel_marginalised(self, theta, lob, zob, ltr_grid_size=None,
                           precomp=None, M_typ=None):
        """<b_sel(theta, ltr)>_{P(ltr | lob)}."""
        if ltr_grid_size is None:
            ltr_grid_size = self.grid.ltr_grid_size
        if M_typ is None:
            M_typ = 1.3e14

        lo = 1.0
        hi = max(lob + 5.0, 1.5 * lob)
        t_nodes, t_wts = gl_nodes(lo, hi, ltr_grid_size)
        p_weights = self.mor.pdf(t_nodes[:, None],
                                 np.array([M_typ])[None, :], zob).ravel()

        theta_arr = np.asarray(theta, dtype=float)
        num = np.zeros_like(theta_arr)
        den = 0.0
        for ltr, w, pw in zip(t_nodes, t_wts, p_weights):
            weight = w * float(pw)
            if weight == 0.0:
                continue
            num = num + weight * self.b_sel_lob_ltr_theta(
                theta_arr, ltr, zob, lob)
            den += weight
        return num / den if den > 0 else np.full_like(theta_arr, np.nan)

    # -- diagnostic "bias_pipeline" for back-compat with old notebooks ----

    def bias_pipeline(self, lob, zob, ltr):
        """Return the key quantities used by cell 16's b_sel(theta)."""
        b_halo = float(self.eff_bias_ltr(lob, zob))
        eff = float(self.eff_bias_ltr(ltr, zob))
        DPRJ = float(self.bar_delta_prj_Beff(lob, zob))
        num2 = float(self.numerator2(ltr, zob))
        b_in = float(self.b_sel_lob_ltr_in(ltr, zob, lob))
        DPRJ_bkg = float(self.bar_delta_prj_bkg(ltr, zob, boost_bias=1.0))
        return dict(lob=lob, zob=zob, ltr=ltr,
                    b_halo_eff_at_lob=b_halo,
                    eff_bias_ltr=eff,
                    bar_delta_prj_Beff=DPRJ,
                    bar_delta_prj_bkg=DPRJ_bkg,
                    numerator2=num2,
                    Delta_prj=self.Delta_prj_lob_ltr(lob, ltr),
                    b_sel_in=b_in)

    def bias_precompute(self, lob, zob):
        """Back-compat stub: everything is cached via self._cache now."""
        # Warm key caches so subsequent marginalisation doesn't re-run them.
        self.bar_delta_prj_Beff(lob, zob)
        return dict(lob=lob, zob=zob)

    def bias_from_precomp(self, precomp, ltr):
        """Back-compat stub returning (b_small, b_large) for old callers."""
        lob = precomp["lob"]
        zob = precomp["zob"]
        return dict(b_small=float(self.b_sel_lob_ltr_in(ltr, zob, lob)),
                    b_large=float(self.eff_bias_ltr(ltr, zob)),
                    delta=(lob - ltr) / self.bar_delta_prj_Beff(lob, zob) - 1.0)
