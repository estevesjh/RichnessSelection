"""Sigma_prj(R | lob, zob) orchestrator.

Evaluates the projected surface density of the miscentered halo profile
convolved with the projection-effects selection bias.  Ported verbatim
from notebook cell 20, with the module-global state replaced by class
attributes so a new Cosmology instance gives a clean, cache-free
evaluation -- safe for MCMC sampling.
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology
from .nfw import NFWMiscentered
from .sel_bias import SelBias
from .gl import gl_nodes
from .geometry import R_lambda
from .photoz import sigma_z as sigma_z_of_z


class SigmaPrj:
    """Projected surface density Sigma^prj(R | lob, zob).

    Parameters
    ----------
    cosmo : Cosmology
    sel_bias : SelBias
        Wired with the same cosmology; supplies bias_precompute and
        b_sel_marginalised.
    nfw : NFWMiscentered
        Miscentered NFW Sigma lookup.
    """

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 nfw: NFWMiscentered):
        self.cosmo = cosmo
        self.sel_bias = sel_bias
        self.nfw = nfw
        self.hmf = sel_bias.hmf
        self.bias = sel_bias.bias
        self.grid = sel_bias.grid
        self.xi_NL = sel_bias.xi_NL

    def __call__(self, R, lob, zob):
        g = self.grid
        R = np.atleast_1d(R).astype(float)

        zlo = max(0.01, zob - 5.0 * sigma_z_of_z(zob))
        zhi = zob + 5.0 * sigma_z_of_z(zob)
        zs, wzs = gl_nodes(zlo, zhi, g.Nz)
        lnMs, wM = gl_nodes(g.ln_M_min, g.ln_M_max, g.NM)
        Ms = np.exp(lnMs)
        D_A_o = self.cosmo.chi(zob) / (1.0 + zob)
        theta_max = 30.0 / D_A_o
        ths, wth = gl_nodes(1e-5, theta_max, g.Nth)
        sin_th = np.sin(ths)
        R_theta_arr = ths * D_A_o                                    # (Nth,)

        pre = self.sel_bias.bias_precompute(lob, zob)
        bsel_grid = self.sel_bias.b_sel_marginalised(
            ths, lob, zob, precomp=pre)                              # (Nth,)
        R_lam_lob = R_lambda(lob)

        chi_o = float(self.cosmo.chi(zob))
        chi_i_arr = self.cosmo.chi(zs)                               # (Nz,)
        dV_arr = self.cosmo.dV_dzdOm(zs)                             # (Nz,)

        cos_th = np.cos(ths)
        dchi2 = (chi_i_arr[:, None] ** 2 + chi_o ** 2
                 - 2.0 * chi_i_arr[:, None] * chi_o * cos_th[None, :])
        dchi = np.sqrt(np.maximum(dchi2, 0.0))
        xi = self.xi_NL(dchi.ravel(), zob).reshape(dchi.shape)       # (Nz, Nth)
        excl = dchi < R_lam_lob                                      # (Nz, Nth)

        hmf_mz = self.hmf(Ms[:, None], zs[None, :])                  # (NM, Nz)
        bM_mz = self.bias(Ms[:, None], zs[None, :])                  # (NM, Nz)

        out = np.zeros_like(R)
        for iz, (zi, wzi) in enumerate(zip(zs, wzs)):
            xi_row = xi[iz]                                          # (Nth,)
            excl_row = excl[iz]
            dV_i = dV_arr[iz]

            bracket = 1.0 + bM_mz[:, iz][None, :] * (
                bsel_grid[:, None] * xi_row[:, None])                # (Nth, NM)
            bracket[excl_row, :] = 0.0

            S_mis = np.stack(
                [self.nfw.sigma_grid(R, R_theta_arr, M, zi) for M in Ms],
                axis=0)                                              # (NM, Nth, N_R)

            wM_M_hmf = wM * Ms * hmf_mz[:, iz]                       # (NM,)
            wth_sin = wth * sin_th                                   # (Nth,)
            contrib_M = np.einsum(
                "Mt,tM,MtR->MR",
                wM_M_hmf[:, None] * np.ones((1, g.Nth)),             # (NM, Nth)
                wth_sin[:, None] * bracket,                          # (Nth, NM)
                S_mis)                                               # (NM, Nth, N_R)
            out += wzi * dV_i * contrib_M.sum(axis=0)
        return out
