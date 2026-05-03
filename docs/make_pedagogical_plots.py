"""Pedagogical figures for docs/richness_selection.tex.

Four panels aimed at a first-time user of the pipeline:
    1. xi_NL(r, z) with the exclusion radius R_excl marked
    2. b_sel(theta) sigmoid at lob in {20, 50, 100}, zob=0.5
    3. Sigma_prj(R) across R = [0.1, 30] cMpc/h at the same three lob
    4. The z-integrand f_{I_2}(z) showing the exclusion-peak structure

Run from the repo root with the y3cl_je env:
    /global/common/software/des/common/Conda_Envs/y3cl_je/bin/python \\
        docs/make_pedagogical_plots.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, XiNL, SelBias,
    SigmaPrj,
)
from richness_selection.sigma_m import SigmaM
from richness_selection.geometry import R_lambda


# ---- global style: serif, DES-Y3-ish colour palette, figure size tuned
# for two-column width in the LaTeX doc (3.3" tall at 6.5" wide). ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

CRIMSON = "#8C1D40"
NAVY = "#1f3a5f"
TEAL = "#2a7f8a"
OCHRE = "#c58a2d"

FIG_DIR = Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)


# ---- single shared pipeline stack (rebuild once) ----
def build_stack():
    cosmo = Cosmology(Om0=0.286, Ob0=0.047, H0=70.0, ns=0.96,
                      sigma8=0.82, mnu=0.0)
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo)
    xi.build()
    nfw = NFWMiscentered(cosmo)
    sel = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
    sp = SigmaPrj(cosmo, sel, nfw)
    return dict(cosmo=cosmo, pk=pk, hmf=hmf, bias=bias, mor=mor,
                xi=xi, nfw=nfw, sel=sel, sp=sp)


def fig_xi_nl(stack):
    xi = stack["xi"]
    r = np.geomspace(1e-2, 1e2, 400)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colours = [NAVY, CRIMSON, OCHRE]
    zs = [0.3, 0.5, 0.7]
    for z, c in zip(zs, colours):
        ax.loglog(r, xi(r, z), lw=1.6, color=c, label=rf"$z={z:.1f}$")
    # exclusion radius at lob=20, zob=0.5
    R_excl = R_lambda(20.0) * (1.0 + 0.5)
    ax.axvline(R_excl, ls="--", lw=1.0, color="0.3")
    ax.text(R_excl * 1.05, 30.0,
            rf"$R_{{\rm excl}}({{\lambda^{{\rm ob}}}}\!=\!20,\,z^{{\rm ob}}\!=\!0.5)$",
            color="0.3", fontsize=9, rotation=90, va="top")
    ax.set_xlabel(r"$r\;[h^{-1}\,{\rm Mpc}]$")
    ax.set_ylabel(r"$\xi_{\rm NL}(r, z)$")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-4, 1e3)
    ax.legend(loc="upper right")
    ax.set_title(r"Nonlinear matter correlation function from halofit + FFTlog")
    fig.tight_layout()
    out = FIG_DIR / "pedag_xi_nl.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_bsel_theta(stack):
    """b_sel(theta) for several Delta^prj = lob - ltr at fixed (lob, zob).

    Delta^prj is the natural variable of the Costanzi model: b_small / b_large
    depend on it linearly through Steps 4-5 of the pipeline, while the sigmoid
    shape in theta is fixed.  The curves fan out as Delta^prj grows because
    more contamination from projection makes b_small larger.
    """
    sel = stack["sel"]
    cosmo = stack["cosmo"]
    lob = 20.0
    zob = 0.5
    pre = sel.bias_precompute(lob, zob)

    # Delta^prj = lob - ltr; fix lob = 20 and vary ltr so Delta^prj in [0, 15].
    deltas = [0.0, 3.0, 6.0, 10.0, 15.0]
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(deltas)))

    chi_o = float(cosmo.chi(zob))
    D_A_o = chi_o / (1.0 + zob)
    th_lam = R_lambda(lob) / D_A_o
    th = np.geomspace(1e-2 * th_lam, 5.0 * th_lam, 240)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for dprj, c in zip(deltas, colours):
        ltr = lob - dprj
        bsel = sel.b_sel_of_theta(th, lob, zob, ltr, precomp=pre)
        ax.semilogx(th / th_lam, bsel, lw=1.8, color=c,
                    label=rf"$\Delta^{{\rm prj}}\!=\!{dprj:g}$")
    ax.axvline(0.5, ls=":", lw=0.8, color="0.4")
    ax.text(0.52, ax.get_ylim()[1] * 0.95, r"$\theta_0=\theta_\lambda/2$",
            color="0.3", fontsize=9, va="top")
    ax.set_xlabel(r"$\theta\,/\,\theta_\lambda$")
    ax.set_ylabel(r"$b_{\rm sel}(\theta\,|\,\lambda^{\rm ob},\,\lambda^{\rm tr},\,z^{\rm ob})$")
    ax.set_xlim(1e-2, 5.0)
    ax.set_title(rf"Selection-bias sigmoid at $\lambda^{{\rm ob}}\!=\!{int(lob)}$, "
                 rf"$z^{{\rm ob}}\!=\!{zob:.1f}$; varying $\Delta^{{\rm prj}}=\lambda^{{\rm ob}}-\lambda^{{\rm tr}}$")
    ax.legend(loc="upper left", title=None, ncol=1)
    fig.tight_layout()
    out = FIG_DIR / "pedag_bsel_theta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_sigma_prj(stack):
    sp = stack["sp"]
    zob = 0.5
    lobs = [20.0, 50.0, 100.0]
    colours = [NAVY, CRIMSON, OCHRE]

    R = np.geomspace(0.1, 30.0, 24)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for lob, c in zip(lobs, colours):
        y = sp(R, lob, zob)
        ax.loglog(R, y, lw=1.8, color=c, marker="o", ms=3.0,
                  label=rf"$\lambda^{{\rm ob}}={int(lob)}$")
        R_excl = R_lambda(lob) * (1.0 + zob)
        ax.axvline(R_excl, ls=":", lw=0.8, color=c, alpha=0.7)
    ax.set_xlabel(r"$R\;[h^{-1}\,{\rm Mpc}]$")
    ax.set_ylabel(r"$\langle \Sigma^{\rm prj}(R\,|\,\lambda^{\rm ob},z^{\rm ob})\rangle"
                  r"\;[h\,M_\odot / {\rm Mpc}^2]$")
    ax.set_title(rf"Two-halo projected surface density at $z^{{\rm ob}}\!=\!{zob:.1f}$;"
                 r" dotted lines mark $R_{\rm excl}$ per bin")
    ax.set_xlim(0.1, 30.0)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = FIG_DIR / "pedag_sigma_prj.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _fz_integrand(sel, stack, zs, lob, zob, which="I2"):
    """Evaluate f_X(z) at an arbitrary z-grid using the same inner
    (M, lambda, theta split-at-exclusion) kernels that _P_operator uses.

    which: 'P1', 'I2', or 'I1'.
    Returns an array of same length as zs.
    """
    from richness_selection.gl import gl_nodes
    from richness_selection.geometry import R_lambda, area_overlap
    from richness_selection.photoz import w_z

    cosmo = stack["cosmo"]
    hmf = stack["hmf"]
    bias = stack["bias"]
    mor = stack["mor"]
    xi_nl = stack["xi"]

    g = sel.grid
    theta_lob = sel._theta_lob(lob, zob)
    chi_o = float(cosmo.chi(zob))
    R_excl = R_lambda(lob) * (1.0 + zob)

    ln_M_min = np.log(10.0 ** np.log10(sel.min_mass4integral))
    ln_M_max = np.log(10.0 ** sel.ln_M_max_log10)
    lnMs, wM = gl_nodes(ln_M_min, ln_M_max, g.NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, float(lob), sel.n_ltr)
    theta_max = 2.0 * theta_lob
    eps_theta = 1e-6

    out = np.zeros_like(np.asarray(zs, dtype=float))
    for i, z in enumerate(zs):
        chi_z = float(cosmo.chi(z))
        dV = float(cosmo.dV_dzdOm(z))
        wz_val = float(w_z(np.array([z]), zob)[0])
        if wz_val <= 0:
            continue
        cos_excl = (chi_z**2 + chi_o**2 - R_excl**2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = min(max(cos_excl, -1.0), 1.0)
        th_lo = (np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12
                 else eps_theta)
        th_lo = max(th_lo, eps_theta)
        if th_lo >= theta_max:
            continue
        ths, wth = gl_nodes(th_lo, theta_max, g.Nth)
        sin_th = np.sin(ths)
        th_weight = wth * 2.0 * np.pi * sin_th
        cos_th = np.cos(ths)
        dchi = np.sqrt(np.maximum(
            chi_z**2 + chi_o**2 - 2.0 * chi_z * chi_o * cos_th, 0.0))
        xi_th = xi_nl(dchi, zob)
        theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
        fA = area_overlap(ths, theta_lob, theta_lam_l)
        sigmoid = sel._sigmoid_theta(ths, lob, zob)
        if which == "P1":
            ang = np.einsum('t,tL->L', th_weight, fA)
            need_b = False
        elif which == "I2":
            ang = np.einsum('t,tL,t->L', th_weight, fA, xi_th)
            need_b = True
        elif which == "I1":
            ang = np.einsum('t,t,tL,t->L', th_weight, sigmoid, fA, xi_th)
            need_b = True
        else:
            raise ValueError(which)
        P_lmz = mor.pdf(lam_grid[:, None], Ms[None, :], z)
        lam_int = np.einsum('L,LM,L->M', wlam, P_lmz,
                            wz_val * lam_grid * ang)
        n_m = hmf(Ms, z)
        if need_b:
            b_m = bias(Ms, z)
            out[i] = dV * np.sum(M_weight * n_m * b_m * lam_int)
        else:
            out[i] = dV * np.sum(M_weight * n_m * lam_int)
    return out


def fig_fz_integrand(stack):
    """f_{I_2}(z) and f_{I_1}(z): the z-integrand showing the twin
    exclusion peaks."""
    sel = stack["sel"]
    cosmo = stack["cosmo"]
    zob = 0.5
    lob = 20.0
    R_excl = R_lambda(lob) * (1.0 + zob)
    zs_ref = np.linspace(0.0, 2.0, 2000)
    chi_ref = cosmo.chi(zs_ref)
    dchi_dz = np.gradient(chi_ref, zs_ref)
    c_over_H = float(np.interp(zob, zs_ref, dchi_dz))
    dz_excl = R_excl / c_over_H

    # Wide window (shows ring + peaks + outer decay at a photo-z scale)
    zs_wide = np.linspace(max(0.01, zob - 0.12), zob + 0.12, 600)
    fI2_wide = _fz_integrand(sel, stack, zs_wide, lob, zob, which="I2")
    fP1_wide = _fz_integrand(sel, stack, zs_wide, lob, zob, which="P1")
    # Zoom: +/- 5 dz_excl (resolves the two sharp peaks)
    zs_zoom = np.linspace(zob - 5.0 * dz_excl, zob + 5.0 * dz_excl, 800)
    fI2_zoom = _fz_integrand(sel, stack, zs_zoom, lob, zob, which="I2")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2),
                                    gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Left: wide view, symlog so the near-zero ring sits above the axis
    ax1.plot(zs_wide, fI2_wide, lw=1.4, color=CRIMSON, label=r"$f_{I_2}(z)$")
    ax1.plot(zs_wide, fP1_wide, lw=1.4, color=NAVY, ls="--",
             label=r"$f_{\mathcal{P}[1]}(z)$")
    ax1.axvline(zob - dz_excl, ls=":", lw=0.8, color="0.35")
    ax1.axvline(zob + dz_excl, ls=":", lw=0.8, color="0.35")
    ax1.axvline(zob, ls="-", lw=0.4, color="0.7")
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xlabel(r"$z$")
    ax1.set_ylabel(r"$f(z)\;\equiv\;(dV/dz\,d\Omega)\,w_z\,\mathcal{I}_{\rm inner}$")
    ax1.set_title("photo-$z$ window")
    ax1.legend(loc="upper right", fontsize=8)

    # Right: zoom around zob so the twin peaks are clearly visible (linear).
    ax2.plot(zs_zoom, fI2_zoom, lw=1.4, color=CRIMSON)
    ax2.axvspan(zob - dz_excl, zob + dz_excl, alpha=0.08, color=CRIMSON,
                label="ring band")
    ax2.axvline(zob - dz_excl, ls=":", lw=0.8, color="0.35")
    ax2.axvline(zob + dz_excl, ls=":", lw=0.8, color="0.35")
    ax2.axvline(zob, ls="-", lw=0.4, color="0.7")
    ax2.set_xlabel(r"$z - z^{\rm ob}$  (units of $\delta z_{\rm excl}$)")
    ax2.set_ylabel(r"$f_{I_2}(z)$")
    xt = np.array([-5, -2.5, -1, 0, 1, 2.5, 5])
    ax2.set_xticks(zob + xt * dz_excl)
    ax2.set_xticklabels([f"{v:g}" for v in xt])
    ax2.set_title(r"zoom: twin exclusion peaks at $z^{\rm ob}\pm\delta z_{\rm excl}$")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        rf"$z$-integrand at $\lambda^{{\rm ob}}\!=\!{int(lob)}$, "
        rf"$z^{{\rm ob}}\!=\!{zob:.1f}$  "
        rf"($\delta z_{{\rm excl}}\!\approx\!{dz_excl:.1e}$)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "pedag_fz_integrand.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print("Building RichnessSelection stack (one-time CAMB/halofit cost)...")
    stack = build_stack()

    outs = []
    print("Figure 1: xi_NL(r, z) ...")
    outs.append(fig_xi_nl(stack))
    print("Figure 2: b_sel(theta) sigmoid ...")
    outs.append(fig_bsel_theta(stack))
    print("Figure 3: Sigma_prj(R) ...")
    outs.append(fig_sigma_prj(stack))
    print("Figure 4: f_{I_2}(z) integrand (optional) ...")
    fz_out = fig_fz_integrand(stack)
    if fz_out is not None:
        outs.append(fz_out)

    print("Wrote:")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
