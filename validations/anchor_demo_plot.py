"""Why the anchor construction works:

    Sigma_cl^anchored(R) = TwoHalo(R) * [pipeline_cl(R) / plateau],
    plateau = < pipeline_cl / TwoHalo >_{R in [8, 20]}

Claim 1 (factorisation): pipeline_cl(R) = A * Phi(R) with A an
uncontrolled amplitude (mass budget x kernel overcount x measure) and
Phi(R) the correct scale-dependence.  Evidence: cl/TwoHalo is flat in
R at two-halo scales.

Claim 2 (robustness): A shifts with arbitrary numerical choices
(M_low, R_max) but cancels in cl(R)/plateau.  Evidence: raw curves
spread, anchored curves collapse onto TwoHalo.

Writes validations/cache/anchor_demo.png.
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import build_stack, CACHE_DIR

from richness_selection import SigmaPrj, DeltaSigmaPrj, TwoHalo

C_CFG = ["#0072B2", "#009E73", "#CC79A7"]     # per config, fixed order
C_REF = "#D55E00"

LOB, ZOB = 20.0, 0.5
R = np.geomspace(0.1, 20.0, 28)
PLATEAU_MASK = R >= 8.0

CONFIGS = [
    ("default (M_low=1e13, R_max=30)", dict(M_low=1e13, R_max=30.0)),
    ("M_low = 1e12",                   dict(M_low=1e12, R_max=30.0)),
    ("R_max = 60",                     dict(M_low=1e13, R_max=60.0)),
]


def build_alt_stack():
    """Second cosmology: the anchored ratio must not move."""
    from richness_selection import (
        Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, SelBias)
    from richness_selection.sigma_m import SigmaM
    from richness_selection.xi_nl import XiNL
    from _common import NFW_TABLE_DIR
    cosmo = Cosmology(Om0=0.32, Ob0=0.05, H0=67.0, ns=0.96,
                      sigma8=0.75, mnu=0.0)
    pk = PkGrid(cosmo); sm = SigmaM(pk)
    hmf = HMF(sm); bias = Bias(sm); mor = MOR()
    xi = XiNL(cosmo); xi.build()
    nfw = NFWMiscentered(cosmo, table_dir=NFW_TABLE_DIR)
    sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
    return cosmo, sb, nfw


def main():
    stack = build_stack()
    cosmo, sb, nfw = stack["cosmo"], stack["sb"], stack["nfw"]

    th = TwoHalo(cosmo, sb)
    T = th.sigma(R, LOB, ZOB)

    saved_min = sb.min_mass4integral
    raw, anchored, plateaus = [], [], []
    for label, cfg in CONFIGS:
        sb.min_mass4integral = cfg["M_low"]
        sp = SigmaPrj(cosmo, sb, nfw, R_max_cMpch=cfg["R_max"])
        cl = sp(R, LOB, ZOB)
        A = float(np.mean((cl / T)[PLATEAU_MASK]))
        raw.append(cl); anchored.append(cl / A); plateaus.append(A)
        print(f"[cfg] {label:34s} plateau A = {A:.4f}")
    sb.min_mass4integral = saved_min

    # ---- cosmology swap: own TwoHalo target, own plateau ----------------
    cosmo2, sb2, nfw2 = build_alt_stack()
    T2 = TwoHalo(cosmo2, sb2).sigma(R, LOB, ZOB)
    sp2 = SigmaPrj(cosmo2, sb2, nfw2)
    cl2 = sp2(R, LOB, ZOB)
    A2 = float(np.mean((cl2 / T2)[PLATEAU_MASK]))
    print(f"[cfg] {'cosmology Om=0.32 s8=0.75 h=0.67':34s} plateau A = {A2:.4f}")

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.8, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.1], hspace=0.08))

    ax.loglog(R, T, color=C_REF, ls="--", lw=2.4,
              label=r"TwoHalo target $\rho_m b_{\rm sel}^{\rm ls} C_\xi$")
    for i, (label, _) in enumerate(CONFIGS):
        ax.loglog(R, raw[i], color=C_CFG[i], lw=1.2, alpha=0.35)
        ax.loglog(R, anchored[i], color=C_CFG[i], lw=2,
                  label=f"anchored: {label}")
    ax.annotate("raw (faded): amplitude A moves with choices",
                xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=8.5, color="#555555")
    ax.set_ylabel(r"$\Sigma_{\rm cl}\;[M_\odot h/{\rm pc}^2]$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.15)

    for i, (label, _) in enumerate(CONFIGS):
        axr.semilogx(R, anchored[i] / T, color=C_CFG[i], lw=2)
    axr.semilogx(R, (cl2 / A2) / T2, color="#E69F00", lw=2, ls="-.",
                 label=r"anchored, $\Omega_m{=}0.32,\sigma_8{=}0.75,h{=}0.67$")
    axr.legend(frameon=False, fontsize=8, loc="upper right")
    axr.axhline(1.0, color=C_REF, ls="--", lw=1.5)
    axr.set_ylim(0.96, 1.10)
    axr.set_xlabel(r"$R\;[{\rm cMpc}/h]$")
    axr.set_ylabel("ratio to TwoHalo")
    axr.tick_params(which="both", direction="in", top=True, right=True)
    axr.grid(alpha=0.15)

    fig.suptitle(
        rf"Anchor construction: $\Sigma_{{\rm cl}}^{{\rm anch}} = "
        rf"{{\rm TwoHalo}}\times[{{\rm cl}}/A]$"
        rf"   ($\lambda^{{\rm ob}}={LOB:.0f}$, $z^{{\rm ob}}={ZOB}$)",
        y=0.945, fontsize=11)

    out = os.path.join(CACHE_DIR, "anchor_demo.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out}")

    # ================= DeltaSigma version =================
    R_ds = R
    mask_ds = R_ds >= 8.0
    T_ds = th.delta_sigma(R_ds, LOB, ZOB)

    ds_anch, ds_A = [], []
    for label, cfg in CONFIGS:
        sb.min_mass4integral = cfg["M_low"]
        dsp = DeltaSigmaPrj(cosmo, sb, nfw, R_max_cMpch=cfg["R_max"])
        ds = dsp(R_ds, LOB, ZOB)
        A = float(np.mean((ds / T_ds)[mask_ds]))
        ds_anch.append(ds / A); ds_A.append(A)
        print(f"[cfg DS] {label:34s} plateau A = {A:.4f}")
    sb.min_mass4integral = saved_min

    T2_ds = TwoHalo(cosmo2, sb2).delta_sigma(R_ds, LOB, ZOB)
    dsp2 = DeltaSigmaPrj(cosmo2, sb2, nfw2)
    ds2 = dsp2(R_ds, LOB, ZOB)
    A2_ds = float(np.mean((ds2 / T2_ds)[mask_ds]))
    print(f"[cfg DS] {'cosmology Om=0.32 s8=0.75 h=0.67':34s} plateau A = {A2_ds:.4f}")

    fig2, (bx, bxr) = plt.subplots(
        2, 1, figsize=(6.8, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.1], hspace=0.08))
    bx.loglog(R_ds, T_ds, color=C_REF, ls="--", lw=2.4,
              label=r"TwoHalo target $\rho_m b_{\rm sel}^{\rm ls}\Delta[C_\xi]$")
    for i, (label, _) in enumerate(CONFIGS):
        bx.loglog(R_ds, ds_anch[i], color=C_CFG[i], lw=2,
                  label=f"anchored: {label}")
    bx.set_ylabel(r"$\Delta\Sigma_{\rm cl}\;[M_\odot h/{\rm pc}^2]$")
    bx.legend(frameon=False, fontsize=8.5, loc="upper right")
    bx.tick_params(which="both", direction="in", top=True, right=True)
    bx.grid(alpha=0.15)

    for i, (label, _) in enumerate(CONFIGS):
        bxr.semilogx(R_ds, ds_anch[i] / T_ds, color=C_CFG[i], lw=2)
    bxr.semilogx(R_ds, (ds2 / A2_ds) / T2_ds, color="#E69F00", lw=2, ls="-.",
                 label=r"anchored, $\Omega_m{=}0.32,\sigma_8{=}0.75,h{=}0.67$")
    bxr.axhline(1.0, color=C_REF, ls="--", lw=1.5)
    bxr.set_ylim(0.96, 1.10)
    bxr.set_xlabel(r"$R\;[{\rm cMpc}/h]$")
    bxr.set_ylabel("ratio to TwoHalo")
    bxr.legend(frameon=False, fontsize=8, loc="upper right")
    bxr.tick_params(which="both", direction="in", top=True, right=True)
    bxr.grid(alpha=0.15)
    fig2.suptitle(
        rf"Anchor construction, $\Delta\Sigma$:"
        rf" $\Delta\Sigma^{{\rm anch}} = {{\rm TwoHalo}}\times[{{\rm cl}}/A]$"
        rf"   ($\lambda^{{\rm ob}}={LOB:.0f}$, $z^{{\rm ob}}={ZOB}$)",
        y=0.945, fontsize=11)
    out2 = os.path.join(CACHE_DIR, "anchor_demo_dsigma.png")
    fig2.savefig(out2, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out2}")


if __name__ == "__main__":
    main()
