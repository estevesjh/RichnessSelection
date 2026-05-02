"""Projection-effects / selection-bias pipeline (Costanzi 2026).

This module holds ALL the projection-kernel integrands and the b_sel
assembly pipeline.  It is kept strictly separate from the pure halo-bias
module (``bias.py``) so that the two distinct physical quantities --
halo clustering b(M, z) vs. observational selection b_sel -- do not
collide in one namespace.

Pipeline (Costanzi 2026, TeX docs/costanzi2026_sigma_prj_b_eff.tex):
  1.  Pop_1(lob, zob)  -- 2-D GL in (z, lnM) with S1+S9 closed forms
  2.  I_1_I_2(lob, zob) -- 3-D GL in (z, lnM, theta) with xi_NL + geometry
  3.  b_halo(lob, zob)  -- 1-D GL in lnM, log-normal MOR pdf
  4.  Delta_RND = P1 + b_halo * I2,  denom = I2 - I1
  5.  bias_precompute hoist : all four quantities are ltr-invariant,
      evaluated once per (lob, zob) and reused across ltr nodes.
  6.  bias_from_precomp(ltr) assembles (delta, b_large, b_small)
  7.  b_sel(theta, ltr) = b_small (1 - sigma(theta)) + b_large sigma(theta)
  8.  b_sel_marginalised(theta) = <b_sel(theta, ltr)>_{p(ltr)}
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology
from .pk import PkGrid
from .hmf import HMF
from .bias import Bias
from .mor import MOR
from .geometry import (R_lambda, theta_lambda, two_disk_overlap,
                       sigma_theta as _sigma_theta)
from .photoz import w_z
from .gl import gl_nodes
from .config import DEFAULT_GRID, GridConfig


def xi_NL_toy(r, z):
    """Toy nonlinear correlation function used by the notebook.

    Placeholder until we plug in a real Hankel transform of P_halofit(k, z).
    """
    r = np.asarray(r, dtype=float)
    D_growth = 1.0 / (1.0 + z)
    return D_growth ** 2 * 100.0 * (r / 5.0) ** (-1.7) * np.exp(-r / 80.0)


class SelBias:
    """Selection-bias / projection-effects orchestrator.

    Parameters
    ----------
    cosmo, pk, hmf, bias, mor
        Physics ingredients.  Cosmology is injected through every child,
        so a new Cosmology invalidates every downstream cache automatically.
    grid
        GridConfig with (Nz, NM, Nth, ln_M_min, ln_M_max, ltr_grid_size).
    xi_NL
        Callable xi_NL(r, z) -> array.  Defaults to the notebook toy.
    """

    def __init__(self, cosmo: Cosmology, pk: PkGrid, hmf: HMF, bias: Bias,
                 mor: MOR, grid: GridConfig = DEFAULT_GRID,
                 xi_NL=None):
        self.cosmo = cosmo
        self.pk = pk
        self.hmf = hmf
        self.bias = bias
        self.mor = mor
        self.grid = grid
        self.xi_NL = xi_NL if xi_NL is not None else xi_NL_toy

    # ---- geometry wrappers bound to this cosmology -----------------------

    def _theta_tgt(self, lob, zob):
        return theta_lambda(lob, zob, self.cosmo)

    def _z_range(self, zob):
        from .photoz import sigma_z_of_z
        sig = sigma_z_of_z(zob)
        return max(0.01, zob - sig), zob + sig

    # ---- Step 1: P[1] ----------------------------------------------------

    def Pop_1(self, lob, zob):
        """P[1](lob, zob): 2-D GL in (z, lnM) using S1+S9 closed forms."""
        g = self.grid
        zlo, zhi = self._z_range(zob)
        zs, wzs = gl_nodes(zlo, zhi, g.Nz)
        lnMs, wM = gl_nodes(g.ln_M_min, g.ln_M_max, g.NM)
        Ms = np.exp(lnMs)

        dV = self.cosmo.dV_dzdOm(zs)                                  # (Nz,)
        wz_arr = w_z(zs, zob)                                         # (Nz,)
        theta_tgt = self._theta_tgt(lob, zob)
        prefac_theta = np.pi * theta_tgt ** 2

        mu = (self.mor.A_mu
              + self.mor.B_mu * (np.log(Ms)[:, None] - np.log(self.mor.M_pivot))
              + self.mor.C_mu * np.log(1.0 + zs)[None, :])            # (NM, Nz)
        sig = self.mor.sigma
        arg = (np.log(lob) - mu - sig * sig) / sig
        from scipy.special import erf
        lam_bar = (np.exp(mu + 0.5 * sig * sig)
                   * 0.5 * (1.0 + erf(arg / np.sqrt(2.0))))           # (NM, Nz)

        # HMF on (NM, Nz)
        hmf_mz = self.hmf(Ms[:, None], zs[None, :])                   # (NM, Nz)

        integrand = (Ms[:, None] * hmf_mz
                     * dV[None, :] * wz_arr[None, :] * prefac_theta
                     * lam_bar)
        return (wM[:, None] * wzs[None, :] * integrand).sum()

    # ---- Step 2: I_1, I_2 -----------------------------------------------

    def I_1_I_2(self, lob, zob):
        """3-D GL in (z, lnM, theta) with xi_NL, exclusion, sigmoid."""
        g = self.grid
        zlo, zhi = self._z_range(zob)
        zs, wzs = gl_nodes(zlo, zhi, g.Nz)
        lnMs, wM = gl_nodes(g.ln_M_min, g.ln_M_max, g.NM)
        Ms = np.exp(lnMs)
        th_lam = self._theta_tgt(lob, zob)
        ths, wth = gl_nodes(1e-6, 2.0 * th_lam, g.Nth)
        R_lam_lob = R_lambda(lob)

        dV = self.cosmo.dV_dzdOm(zs)                                  # (Nz,)
        wz_z = w_z(zs, zob)                                           # (Nz,)
        chi_i = self.cosmo.chi(zs)                                    # (Nz,)
        chi_o = float(self.cosmo.chi(zob))
        hmf_mz = self.hmf(Ms[:, None], zs[None, :])                   # (NM, Nz)
        bM_mz = self.bias(Ms[:, None], zs[None, :])                   # (NM, Nz)

        mu = (self.mor.A_mu
              + self.mor.B_mu * (np.log(Ms)[:, None] - np.log(self.mor.M_pivot))
              + self.mor.C_mu * np.log(1.0 + zs)[None, :])
        sig = self.mor.sigma
        arg = (np.log(lob) - mu - sig * sig) / sig
        from scipy.special import erf
        lam_bar = (np.exp(mu + 0.5 * sig * sig)
                   * 0.5 * (1.0 + erf(arg / np.sqrt(2.0))))           # (NM, Nz)

        cos_th = np.cos(ths)
        dchi2 = (chi_i[:, None] ** 2 + chi_o ** 2
                 - 2.0 * chi_i[:, None] * chi_o * cos_th[None, :])
        dchi = np.sqrt(np.maximum(dchi2, 0.0))
        xi = self.xi_NL(dchi.ravel(), zob).reshape(dchi.shape)        # (Nz, Nth)
        excl = dchi < R_lam_lob
        sin_th = np.sin(ths)
        sig_th = 1.0 / (1.0 + np.exp(-(2.5 / th_lam)
                                     * (ths - 0.5 * th_lam)))         # (Nth,)

        w_outer = (wzs[:, None, None] * wM[None, :, None]
                   * wth[None, None, :])                              # (Nz,NM,Nth)
        common = (dV[:, None, None] * wz_z[:, None, None]
                  * Ms[None, :, None] * hmf_mz.T[:, :, None]
                  * bM_mz.T[:, :, None] * lam_bar.T[:, :, None]
                  * (2.0 * np.pi)
                  * sin_th[None, None, :] * xi[:, None, :])
        common = common * (~excl)[:, None, :]

        I2 = (w_outer * common).sum()
        I1 = (w_outer * common * sig_th[None, None, :]).sum()
        return float(I1), float(I2)

    # ---- Step 3: b_halo (observed-richness-weighted mean halo bias) ------

    def b_halo(self, lob, zob):
        """b_halo(lob, zob): 1-D GL in lnM, log-normal MOR pdf."""
        g = self.grid
        lnMs, wM = gl_nodes(g.ln_M_min, g.ln_M_max, g.NM)
        Ms = np.exp(lnMs)
        P = self.mor.pdf(lob, Ms, zob)
        MP = Ms * self.hmf(Ms, zob) * P
        bM = self.bias(Ms, zob)
        num = (wM * MP * bM).sum()
        den = (wM * MP).sum()
        return float(num / den)

    # ---- Step 4-5: hoist precompute and assemble per-ltr -----------------

    def bias_precompute(self, lob, zob):
        """Evaluate the four ltr-invariant pieces once per (lob, zob) sample."""
        P1 = float(self.Pop_1(lob, zob))
        I1, I2 = self.I_1_I_2(lob, zob)
        bhalo = self.b_halo(lob, zob)
        d_RND = P1 + bhalo * I2
        denom = I2 - I1
        return dict(lob=lob, zob=zob, P1=P1, I1=I1, I2=I2,
                    bhalo=bhalo, d_RND=d_RND, denom=denom)

    def bias_from_precomp(self, precomp, ltr):
        P1 = precomp["P1"]; I1 = precomp["I1"]; I2 = precomp["I2"]
        bhalo = precomp["bhalo"]; d_RND = precomp["d_RND"]
        denom = precomp["denom"]; lob = precomp["lob"]
        delta = (lob - ltr) / d_RND - 1.0
        b_large = bhalo * (1.0 + 0.13 * delta)
        if abs(denom) < 1e-12 * (abs(I1) + abs(I2) + 1e-30):
            b_small = b_large
        else:
            b_small = ((lob - ltr) - P1 - b_large * I1) / denom
        return dict(delta=delta, b_large=b_large, b_small=b_small)

    def bias_pipeline(self, lob, zob, ltr):
        pre = self.bias_precompute(lob, zob)
        out = self.bias_from_precomp(pre, ltr)
        out.update(pre)
        return out

    # ---- Step 6-7: b_sel(theta), marginalised over ltr ------------------

    def b_sel_of_theta(self, theta, lob, zob, ltr, precomp=None):
        pre = precomp if precomp is not None else self.bias_precompute(lob, zob)
        pr = self.bias_from_precomp(pre, ltr)
        s = _sigma_theta(np.asarray(theta, dtype=float), lob, zob, self.cosmo)
        return pr["b_small"] * (1.0 - s) + pr["b_large"] * s

    def b_sel_marginalised(self, theta, lob, zob, ltr_grid_size=None,
                           precomp=None):
        """<b_sel(theta, ltr)>_{p(ltr | lob)}.

        Uses a log-normal ltr prior centered near lob for numerical
        stability; see TeX Step 6b.  Precompute is hoisted out so the
        expensive integrals run once, not once per ltr node.
        """
        if ltr_grid_size is None:
            ltr_grid_size = self.grid.ltr_grid_size
        pre = precomp if precomp is not None else self.bias_precompute(lob, zob)

        sig = self.mor.sigma
        lo = np.log(max(1.0, lob - 10.0 * sig * lob))
        hi = np.log(lob + 10.0 * sig * lob)
        t_nodes, t_wts = gl_nodes(lo, hi, ltr_grid_size)

        theta_arr = np.asarray(theta, dtype=float)
        num = np.zeros_like(theta_arr)
        den = 0.0
        for lnl, w in zip(t_nodes, t_wts):
            ltr = np.exp(lnl)
            weight = w * self.mor.pdf(ltr, 1e14, zob)
            num = num + weight * self.b_sel_of_theta(
                theta_arr, lob, zob, ltr, precomp=pre)
            den += weight
        return num / den
