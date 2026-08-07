"""Pedagogical figures for docs/richness_selection_frozen.tex Sec. 2.

Regenerates, from ``FrozenSelBias`` internals, the three figures the
note includes -- same filenames, colors and plot styles as the
original prototype versions (validations/archive/frozen_kernels.py):

    docs/figs/pedag_frozen_kernels.png   I_lambda(x) + f_A(x, x_lambda)
    docs/figs/pedag_frozen_radial.png    exclusion-zone radial integrand
    docs/figs/pedag_frozen_farzone.png   free-of-exclusion-zone integrand

Run from the repo root:
    python docs/make_frozen_algorithm_figs.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DOCS = Path(__file__).parent
FIG_DIR = _DOCS / "figs"
FIG_DIR.mkdir(exist_ok=True)

from richness_selection import (Cosmology, PkGrid, HMF, Bias, MOR, XiNL,
                                FrozenSelBias)
from richness_selection.sigma_m import SigmaM
from richness_selection.frozen_bsel import PI_S_OVER_REX
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z
from richness_selection.gl import gl_nodes

LOB, ZOB = 20.0, 0.5


def build_frozen():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo); xi.build()
    return FrozenSelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)


def fig_kernels(fsel, lob, zob):
    """I_lambda(x) (eq. Ilam) + the capture fraction f_A family."""
    I = fsel.I_lambda(lob, zob)
    x = np.linspace(1e-3, 2.2, 400)
    Ix = I(x)
    Ix_ls = fsel.sigma_x(x) * Ix
    norm = I(np.array([1e-3]))[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(x, Ix / norm, 'C3-', lw=2,
             label=r'$I_\lambda(x)/I_\lambda(0)$')
    ax1.plot(x, Ix_ls / norm, 'C0-', lw=2,
             label=r'$I_\lambda^{\rm ls}(x)/I_\lambda(0)'
                   r' = \sigma(x)\,I_\lambda(x)/I_\lambda(0)$')
    ax1.plot(x, fsel.sigma_x(x), 'k--', lw=1.2, label=r'$\sigma(x)$')
    ax1.set_xlabel(r'$x = \theta/\theta_{\lambda,\mathrm{ob}} = s/R_{\rm excl}$')
    ax1.set_ylabel('frozen mass-integral moments')
    ax1.legend(fontsize=9); ax1.set_xlim(0, 2.2)
    ax1.axvline(2.0, color='0.7', lw=0.8)
    ax1.set_title(f'$(\\lambda^{{\\rm ob}},z^{{\\rm ob}})=({lob:g},{zob:g})$')

    # right panel: the capture fraction f_A(x, x_lambda) itself,
    # for a few donor sizes x_lambda = theta_lambda / theta_lam_ob
    for rho, c in zip((0.55, 0.70, 0.85, 1.00),
                      ('C0', 'C2', 'C1', 'C3')):
        fA = area_overlap(x[:, None], 1.0, np.array([rho]))[:, 0]
        ax2.plot(x, fA, c + '-', lw=2,
                 label=rf'$x_\lambda={rho:g}$'
                       rf'  ($\lambda={lob * rho ** 5:.1f}$)')
        ax2.axvline(1.0 + rho, color=c, ls=':', lw=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, 2.2)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$f_A(x,x_\lambda) = A_{\rm ov}/(\pi\theta_\lambda^2)$')
    ax2.legend(fontsize=9, title='donor size (support end dotted)',
               title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'pedag_frozen_kernels.png', dpi=150)
    plt.close(fig)


def _G_of_r(fsel, lob, zob, rs, ls_weighted):
    """G(r) of eq. (excl) on an r array (figure-resolution version)."""
    los = fsel._los(zob)
    I = fsel.I_lambda(lob, zob)
    Rex = R_lambda(lob) * (1.0 + zob)
    pi_s = PI_S_OVER_REX * Rex
    mu_t, mu_w = gl_nodes(0.0, 1.0, fsel.n_mu)

    mu_lo = np.sqrt(np.maximum(0.0, 1.0 - (2.0 * Rex / rs) ** 2))
    mu_hi = np.minimum(1.0, pi_s / rs)
    active = mu_hi > mu_lo
    mus = mu_lo[:, None] + (mu_hi - mu_lo)[:, None] * mu_t[None, :]
    wmu = (mu_hi - mu_lo)[:, None] * mu_w[None, :]
    xg = (rs[:, None] / Rex) * np.sqrt(np.maximum(1.0 - mus ** 2, 0.0))
    Ix = I(xg.reshape(-1)).reshape(xg.shape)
    if ls_weighted:
        Ix = fsel.sigma_x(xg) * Ix
    pi_grid = rs[:, None] * mus
    wbar = 0.5 * (w_z(fsel._z_of_pi(los, pi_grid), zob)
                  + w_z(fsel._z_of_pi(los, -pi_grid), zob))
    return 2.0 * np.sum(wmu * Ix * wbar, axis=1) * active


def fig_radial(fsel, lob, zob):
    """Exclusion-zone radial integrand r^2 xi_mm(r) G(r) (eq. excl)."""
    Rex = R_lambda(lob) * (1.0 + zob)
    pi_s = PI_S_OVER_REX * Rex
    r_max = np.sqrt(pi_s ** 2 + 4.0 * Rex ** 2)
    rs = np.geomspace(Rex, r_max, 300)
    xi_r = fsel.xi_NL(rs, zob)
    y_tot = rs ** 2 * xi_r * _G_of_r(fsel, lob, zob, rs, ls_weighted=False)
    y_ls = rs ** 2 * xi_r * _G_of_r(fsel, lob, zob, rs, ls_weighted=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(rs / Rex, y_tot, 'C3-', lw=2,
              label=r'$r^2\,\xi_{\rm mm}(r)\,G_{\rm ss+ls}(r)$')
    ax.loglog(rs / Rex, y_ls, 'C0-', lw=2,
              label=r'$r^2\,\xi_{\rm mm}(r)\,G_{\rm ls}(r)$')
    ax.axvline(1.0, color='k', ls=':', lw=1,
               label=r'$r=R_{\rm excl}$ (hard lower limit)')
    ax.axvline(2.0, color='0.5', ls='--', lw=1,
               label=r'$r=2R_{\rm excl}$ (cylinder wall kink)')
    ax.set_xlabel(r'$r/R_{\rm excl}$')
    ax.set_ylabel('exclusion-zone radial integrand')
    ax.set_title('Twin peaks + plateau are gone: smooth power-law decay')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'pedag_frozen_radial.png', dpi=150)
    plt.close(fig)


def fig_farzone(fsel, lob, zob):
    """Free-of-exclusion-zone line-of-sight integrand (eq. cyl),
    background side, log-pi measure."""
    los = fsel._los(zob)
    ml = fsel._frozen_ml(lob, zob)
    Rex = R_lambda(lob) * (1.0 + zob)
    pi_s = PI_S_OVER_REX * Rex

    # A_{ss+ls} moment of eq. (moments)
    x_n, x_w = gl_nodes(0.0, 2.0, fsel.n_x)
    I_n = area_overlap(x_n, 1.0, ml["x_lam"]) @ ml["w_b"]
    A_tot = Rex ** 2 * float(np.sum(x_w * x_n * I_n))

    pi_max_bg = float(fsel.cosmo.chi(los["z_hi"])) - los["chi_o"]
    pis = np.geomspace(pi_s, pi_max_bg, 400)
    zP = fsel._z_of_pi(los, pis)
    wzv = w_z(zP, zob)
    xi_v = fsel.xi_NL(pis, zob)
    y = pis * wzv * xi_v * A_tot          # log-measure integrand
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(pis, y / y.max(), 'C3-', lw=2,
                label=r'$\pi\,w(\pi)\,A_{\rm ss+ls}\,\xi_{\rm mm}(\pi)$ (norm.)')
    ax.semilogx(pis, wzv, 'k--', lw=1,
                label=r'photo-$z$ window $w_z$ (parabolic)')
    ax.axvline(105.0, color='0.5', ls=':', lw=1,
               label=r'BAO $\sim105\,h^{-1}$Mpc')
    ax.set_xlabel(r'$\pi = |\chi(z)-\chi_o|\;[h^{-1}{\rm Mpc}]$')
    ax.set_ylabel('free-of-exclusion-zone integrand (background side)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'pedag_frozen_farzone.png', dpi=150)
    plt.close(fig)


def main():
    print("Building FrozenSelBias (one-time CAMB/halofit cost)...")
    fsel = build_frozen()
    fig_kernels(fsel, LOB, ZOB)
    print(f"  wrote {FIG_DIR / 'pedag_frozen_kernels.png'}")
    fig_radial(fsel, LOB, ZOB)
    print(f"  wrote {FIG_DIR / 'pedag_frozen_radial.png'}")
    fig_farzone(fsel, LOB, ZOB)
    print(f"  wrote {FIG_DIR / 'pedag_frozen_farzone.png'}")


if __name__ == "__main__":
    main()
