"""Stage (c): port of the Costanzi-notebook b_sel(theta) chain (cell 16),
compared against our SelBias.marginalised_bias.

Notebook model (with the boost factor neutralised, boost_bias = 1 --
the calibration parameters inter/slope_boost_bias are not in the
public repo):

  b_sel(ltr, z, lob, theta) = b_in + (b_eff(ltr) - b_in) * S(theta)
  S(theta) = 1 / (1 + exp(-k (theta - x0))),  x0 = theta_lob/2,
             k = 2.5 / theta_lob
  b_in(ltr) = (Delta_prj - bar_Delta_bkg(ltr)) / numerator2(ltr)
  Delta_prj = lob - ltr

with bar_Delta_bkg / numerator2 the notebook's kernel-weighted
(fg+bg) LoS integrals over mass, inner richness, and angle, carrying
the aperture-overlap factor and the sigmoid split of the [1 + b b xi]
bracket (int_over_phi2: sigmoid side; int_over_phi4: 1 - sigmoid side).

Marginalisation over ltr (notebook md cell 1):
  <b_sel>(theta) = int dltr b_sel(ltr, theta) P(lob|ltr, zob) n(ltr)
                   / int dltr P(lob|ltr, zob) n(ltr)
  n(ltr) = int dM n(M) P(ltr|M, zob)
using our MC19 P(lob|ltr, z) (same DES-Y3 parameter file) and the
notebook's DES-Y1 MOR with epsilon = 0.2839 (our default has
epsilon = 0; passed explicitly here).

Writes validations/cache/costanzi_bsel_compare.png.
"""
from __future__ import annotations
import os

import numpy as np
from scipy.optimize import bisect

from _common import build_stack, CACHE_DIR

from richness_selection import MOR
from richness_selection.geometry import R_lambda
from richness_selection.photoz import (w_z, zmin4zkernel, zmax4zkernel)
from richness_selection.plob_ltr import P_lob_given_ltr

DAMPING = 2.5
N_LOS = 40            # 20 fg + 20 bg
N_M = 40
N_LIN = 60            # inner (neighbour) richness grid
N_TH = 40


def area_overlap(theta, th_lob, th_l):
    """Notebook aperture-overlap fraction; theta (Nth,), th_l (Nl,)."""
    th = theta[:, None]
    tl = th_l[None, :]
    A = np.ones(np.broadcast_shapes(th.shape, tl.shape))
    A = np.where(tl > th_lob, th_lob ** 2 / tl ** 2, A)
    cond = th > np.abs(th_lob - tl)
    a1 = np.clip((th ** 2 + tl ** 2 - th_lob ** 2) / (2.0 * th * tl),
                 -1.0, 1.0)
    a2 = np.clip((th ** 2 + th_lob ** 2 - tl ** 2) / (2.0 * th * th_lob),
                 -1.0, 1.0)
    s = ((-th + tl + th_lob) * (th + tl - th_lob)
         * (th - tl + th_lob) * (th + tl + th_lob))
    s = np.maximum(s, 0.0)
    Ap = (tl ** 2 * np.arccos(a1) + th_lob ** 2 * np.arccos(a2)
          - 0.5 * np.sqrt(s)) / (np.pi * tl ** 2)
    A = np.where(cond, Ap, A)
    A = np.where(th > th_lob + tl, 0.0, A)
    return A


class CostanziBsel:
    def __init__(self, stack, lob, zob):
        self.stack = stack
        self.cosmo = stack["cosmo"]
        self.hmf = stack["hmf"]
        self.bias = stack["bias"]
        self.xi = stack["xi"]
        self.mor = MOR(epsilon=0.283887020)     # notebook epsi
        self.lob = float(lob)
        self.zob = float(zob)
        self.chi_o = float(self.cosmo.chi(zob))
        self.th_lob = float(R_lambda(lob)) * (1.0 + zob) / self.chi_o
        self.R_excl = float(R_lambda(lob)) * (1.0 + zob)
        self.m_grid = np.logspace(13.0, 15.5, N_M)
        # kernel-bounded fg+bg LoS grid (notebook structure)
        z_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
        z_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
        zs_ref = np.linspace(0.0, 2.0, 3000)
        chi_ref = self.cosmo.chi(zs_ref)
        z_of_chi = lambda c: np.interp(c, chi_ref, zs_ref)
        d_fg = np.geomspace(1e-4, self.chi_o - float(self.cosmo.chi(z_lo)),
                            N_LOS // 2)
        d_bg = np.geomspace(1e-4, float(self.cosmo.chi(z_hi)) - self.chi_o,
                            N_LOS // 2)
        self.ztr = np.concatenate([z_of_chi(self.chi_o - d_fg)[::-1],
                                   z_of_chi(self.chi_o + d_bg)])
        self.chi_tr = self.cosmo.chi(self.ztr)
        self.wk = w_z(self.ztr, zob)
        self.dV = self.cosmo.dV_dzdOm(self.ztr)
        # per-z hmf/bias tables
        self.n_mz = np.array([self.hmf(self.m_grid, z) for z in self.ztr])
        self.b_mz = np.array([self.bias(self.m_grid, z) for z in self.ztr])
        self._sig_k = DAMPING / self.th_lob          # sigmoid steepness
        self._sig_x0 = 0.5 * self.th_lob

    # ---- sigmoid split -----------------------------------------------------
    def _S(self, theta):
        return 1.0 / (1.0 + np.exp(-self._sig_k * (theta - self._sig_x0)))

    # ---- effective (large-scale) bias of the ltr sample --------------------
    def eff_bias_ltr(self, ltr):
        p = self.mor.pdf(ltr, self.m_grid, self.zob)
        n0 = self.hmf(self.m_grid, self.zob)
        b0 = self.bias(self.m_grid, self.zob)
        w = n0 * p * self.m_grid
        return float(np.trapezoid(b0 * w, np.log(self.m_grid))
                     / np.trapezoid(w, np.log(self.m_grid)))

    # ---- angular kernels (per z, inner-lambda grid) ------------------------
    def _phi_kernel(self, z_i, lin_grid, weight):
        """int dtheta sin(theta) W(theta) xi(dis) A_ov, W = S or 1-S.

        Returns (Nlin,) for one LoS redshift.
        """
        c_t = self.chi_tr[z_i]
        z_t = self.ztr[z_i]
        th_l = R_lambda(lin_grid) * (1.0 + z_t) / self.cosmo.chi(z_t)
        out = np.empty(lin_grid.size)
        for j, tl in enumerate(np.atleast_1d(th_l)):
            th = np.geomspace(1e-6, self.th_lob + tl, N_TH)
            dis = np.sqrt(c_t ** 2 + self.chi_o ** 2
                          - 2.0 * c_t * self.chi_o * np.cos(th))
            xiv = self.xi(dis, self.zob)
            xiv = np.where(dis < self.R_excl, 0.0, xiv)
            W = self._S(th) if weight == "S" else 1.0 - self._S(th)
            Aov = area_overlap(th, self.th_lob, np.array([tl]))[:, 0]
            out[j] = np.trapezoid(np.sin(th) * W * xiv * Aov, th)
        return out

    def _geom_kernel(self, z_i, lin_grid):
        """Omega_halos * f_area  (the un-clustered background weight)."""
        z_t = self.ztr[z_i]
        th_l = R_lambda(lin_grid) * (1.0 + z_t) / self.cosmo.chi(z_t)
        om = 2.0 * np.pi * (1.0 - np.cos(th_l + self.th_lob))
        fa = (1.0 + th_l / self.th_lob) ** -2.0
        return om * fa

    # ---- LoS-integrated Delta-prj pieces ------------------------------------
    def _z_mass_lambda(self, per_z_lambda):
        """Assemble int dz dV w int dM M n {int dlin pltr lin K(lin, z)}.

        per_z_lambda(z_i, lin_grid) -> (Nlin,) kernel K.
        """
        lin = np.linspace(1e-10, self.lob, N_LIN)
        tot = 0.0
        vals = np.empty((self.ztr.size,))
        for i in range(self.ztr.size):
            K = per_z_lambda(i, lin)                     # (Nlin,)
            p = self.mor.pdf(lin[:, None], self.m_grid[None, :], self.ztr[i])
            lam_int = np.trapezoid(p * (lin * K)[:, None], lin, axis=0)  # (NM,)
            vals[i] = np.trapezoid(self.m_grid * self.n_mz[i] * lam_int,
                                   np.log(self.m_grid))
        return float(np.trapezoid(self.dV * self.wk * vals, self.ztr))

    def bar_delta_bkg(self, b_eff):
        """Notebook bar_delta_prj_bkg with boost=1: geometric part +
        clustered part carrying b(M) * b_eff * sigmoid-weighted xi."""
        def K(i, lin):
            geom = self._geom_kernel(i, lin)
            phiS = self._phi_kernel(i, lin, "S")
            b_mean = np.trapezoid(
                self.b_mz[i] * self.n_mz[i] * self.m_grid,
                np.log(self.m_grid)) / np.trapezoid(
                self.n_mz[i] * self.m_grid, np.log(self.m_grid))
            return geom + 2.0 * np.pi * b_mean * b_eff * phiS
        return self._z_mass_lambda(K)

    def numerator2(self):
        """Notebook numerator2: b(M)-weighted, (1 - sigmoid) side."""
        lin = np.linspace(1e-10, self.lob, N_LIN)
        vals = np.empty((self.ztr.size,))
        for i in range(self.ztr.size):
            phi1mS = self._phi_kernel(i, lin, "1-S")     # (Nlin,)
            p = self.mor.pdf(lin[:, None], self.m_grid[None, :], self.ztr[i])
            lam_int = np.trapezoid(
                p * (lin * 2.0 * np.pi * phi1mS)[:, None], lin, axis=0)
            vals[i] = np.trapezoid(
                self.m_grid * self.b_mz[i] * self.n_mz[i] * lam_int,
                np.log(self.m_grid))
        return float(np.trapezoid(self.dV * self.wk * vals, self.ztr))

    # ---- assembly ------------------------------------------------------------
    def b_sel_ltr_theta(self, ltr, thetas):
        b_eff = self.eff_bias_ltr(ltr)
        num = self.numerator2()
        bkg = self.bar_delta_bkg(b_eff)
        b_in = (self.lob - ltr - bkg) / num
        S = self._S(thetas)
        return b_in + (b_eff - b_in) * S

    def marginalised(self, thetas, n_ltr=12):
        ltrs = np.linspace(max(1.0, 0.35 * self.lob),
                           self.lob - 0.25, n_ltr)
        n0 = self.hmf(self.m_grid, self.zob)
        w = np.empty(n_ltr)
        curves = np.empty((n_ltr, thetas.size))
        for i, lt in enumerate(ltrs):
            p_m = self.mor.pdf(lt, self.m_grid, self.zob)
            n_ltr_dens = np.trapezoid(n0 * p_m * self.m_grid,
                                      np.log(self.m_grid))
            w[i] = float(P_lob_given_ltr(self.lob, lt, self.zob)) * n_ltr_dens
            curves[i] = self.b_sel_ltr_theta(lt, thetas)
            print(f"  [ltr={lt:6.2f}] b_in-> {curves[i][0]:+.3f} "
                  f"b_ls-> {curves[i][-1]:+.3f}  w={w[i]:.3e}")
        w /= np.trapezoid(w, ltrs)
        return np.trapezoid(w[:, None] * curves, ltrs, axis=0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stack = build_stack()
    sb = stack["sb"]
    LOB, ZOB = 20.0, 0.5
    chi_o = float(stack["cosmo"].chi(ZOB))

    cb = CostanziBsel(stack, LOB, ZOB)
    thetas = np.geomspace(2e-4, 3e-2, 25)
    print("[costanzi-bsel] marginalising ...")
    b_nb = cb.marginalised(thetas)

    pre = sb.bias_precompute(LOB, ZOB)
    b_ours = sb.marginalised_bias(LOB, ZOB, precomp=pre)(thetas)

    s = thetas * chi_o
    print(f"\n{'s [cMpc/h]':>11s} {'notebook':>9s} {'ours':>9s} {'ratio':>7s}")
    for i in range(0, thetas.size, 3):
        print(f"{s[i]:11.2f} {b_nb[i]:9.3f} {b_ours[i]:9.3f} "
              f"{b_nb[i]/b_ours[i]:7.3f}")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.semilogx(s, b_nb, color="#0072B2", lw=2,
                label="notebook chain (boost=1)")
    ax.semilogx(s, b_ours, color="#D55E00", lw=2, ls="--",
                label="ours (P[X] operator)")
    ax.set_xlabel(r"transverse comoving $s=\theta\chi_o$ [cMpc/$h$]")
    ax.set_ylabel(r"$\langle b_{\rm sel}(\theta)\rangle$")
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.15)
    ax.set_title(rf"$b_{{\rm sel}}$: notebook vs package"
                 rf"  ($\lambda^{{\rm ob}}={LOB:.0f}$, $z^{{\rm ob}}={ZOB}$)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(CACHE_DIR, "costanzi_bsel_compare.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
