"""Frozen-physics reformulation vs production, same figures as the main doc.

Reuses the *exact* production code paths from the existing pedagogical
scripts (no re-implementation of the current-repo curves):

    - b_sel(theta):        SelBias.b_sel_of_theta, same call signature
                            as docs/make_pedagogical_plots.py::fig_bsel_theta
                            (Fig. 1 pattern).
    - DeltaSigma^prj(R):   docs/make_delta_sigma_plots.py's
                            _patch_bsel_for_ltr / _restore_bsel and the
                            DeltaSigmaPrj call, same as fig_delta_sigma_prj
                            (Fig. 3 pattern). build_stack() below mirrors
                            make_pedagogical_plots.build_stack, with the
                            NFW table dir resolved the way
                            make_delta_sigma_plots.build_stack does
                            (RICHNESS_SELECTION_NFW_DIR env var).

The frozen-physics numbers come from richness_selection.FrozenSelBias
-- the exclusion / free-of-exclusion zone reformulation of
docs/richness_selection_frozen.tex. Swapping the bias_precompute
source (same downstream sigmoid/plateau algebra) is the *only* change
between the "production" and "frozen" curves in both figures --
everything else (NFW lookup, theta-grid, marginalisation) is shared,
identical code.

NOTE on conventions: FrozenSelBias follows the note's eq. (bls)
Poisson delta_prj convention (Delta_RND = <Dprj>_rnd) and the clean
random channel (no exclusion carve-out), while production SelBias
uses Delta_RND = P1 + b_eff I2 and carves the exclusion ball out of
P1.  The frozen-vs-production offsets in these figures therefore
include those *convention* differences on top of the operator
numerics; the convention-free accuracy gate is
validations/frozen_bsel_validation.py (operators vs scipy.quad).

Outputs:
    docs/figs/pedag_frozen_vs_prod_bsel.png
    docs/figs/pedag_frozen_vs_prod_delta_sigma.png

Run from the repo root (same env as the other pedagogical scripts):
    RICHNESS_SELECTION_NFW_DIR=<path to nfw_off_center> \\
        python docs/make_validation_plots_frozen_physics.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_DOCS = Path(__file__).parent
_ROOT = _DOCS.parent
sys.path.insert(0, str(_DOCS))

import make_delta_sigma_plots as dsig             # noqa: E402

from richness_selection import (Cosmology, PkGrid, HMF, Bias, MOR,      # noqa: E402
                                NFWMiscentered, XiNL, SelBias,
                                FrozenSelBias, SigmaPrj, DeltaSigmaPrj)
from richness_selection.sigma_m import SigmaM       # noqa: E402
from richness_selection.geometry import R_lambda    # noqa: E402

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

FIG_DIR = _DOCS / "figs"
FIG_DIR.mkdir(exist_ok=True)

LOB = 20.0
ZOB = 0.5
DELTAS = [0.0, 3.0, 6.0, 10.0, 15.0]     # same sweep as Fig. 1 / Fig. 3


def build_stack():
    """Identical to make_pedagogical_plots.build_stack, except the NFW
    table dir is resolved the same way make_delta_sigma_plots.build_stack
    does (RICHNESS_SELECTION_NFW_DIR env var, since pedag.build_stack
    hardcodes the NERSC default which doesn't exist off-cluster)."""
    cosmo = Cosmology(Om0=0.286, Ob0=0.047, H0=70.0, ns=0.96,
                      sigma8=0.82, mnu=0.0)
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo); xi.build()
    nfw_dir = os.environ.get("RICHNESS_SELECTION_NFW_DIR")
    nfw = NFWMiscentered(cosmo, table_dir=nfw_dir) if nfw_dir \
        else NFWMiscentered(cosmo)
    sel = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
    fsel = FrozenSelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
    sp = SigmaPrj(cosmo, sel, nfw)
    return dict(cosmo=cosmo, pk=pk, hmf=hmf, bias=bias, mor=mor,
                xi=xi, nfw=nfw, sel=sel, fsel=fsel, sp=sp)


def frozen_precomp_for(sel, stack, lob, zob):
    """Frozen-method precomp dict from FrozenSelBias (see module
    docstring for the two convention differences vs production)."""
    return stack["fsel"].bias_precompute(lob, zob)


# ---------------------------------------------------------------------
# Figure 1 analogue: b_sel(theta), production (dashed) vs frozen (solid)
# ---------------------------------------------------------------------

def fig_bsel_frozen_vs_prod(stack):
    sel = stack["sel"]
    cosmo = stack["cosmo"]

    pre_prod = sel.bias_precompute(LOB, ZOB)
    pre_frozen = frozen_precomp_for(sel, stack, LOB, ZOB)

    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(DELTAS)))
    chi_o = float(cosmo.chi(ZOB))
    D_A_o = chi_o / (1.0 + ZOB)
    th_lam = R_lambda(LOB) / D_A_o
    th = np.geomspace(1e-2 * th_lam, 5.0 * th_lam, 240)
    x = th / th_lam

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08})

    for dprj, c in zip(DELTAS, colours):
        ltr = LOB - dprj
        b_prod = sel.b_sel_of_theta(th, LOB, ZOB, ltr, precomp=pre_prod)
        b_froz = sel.b_sel_of_theta(th, LOB, ZOB, ltr, precomp=pre_frozen)
        ax1.semilogx(x, b_prod, lw=1.8, color=c, ls="--")
        ax1.semilogx(x, b_froz, lw=1.8, color=c, ls="-",
                     label=rf"$\Delta^{{\rm prj}}\!=\!{dprj:g}$")
        rel = 100.0 * (b_froz - b_prod) / np.max(np.abs(b_prod))
        ax2.semilogx(x, rel, lw=1.5, color=c)

    ax1.axvline(0.5, ls=":", lw=0.8, color="0.4")
    ax1.plot([], [], ls="--", color="0.3", label="production (current repo)")
    ax1.plot([], [], ls="-", color="0.3", label="frozen reformulation")
    ax1.set_ylabel(r"$b_{\rm sel}(\theta\,|\,\lambda^{\rm ob},\,"
                   r"\lambda^{\rm tr},\,z^{\rm ob})$")
    ax1.set_title(rf"Selection-bias sigmoid at $\lambda^{{\rm ob}}\!=\!{int(LOB)}$, "
                 rf"$z^{{\rm ob}}\!=\!{ZOB:.1f}$: production vs frozen "
                 r"reformulation")
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlim(1e-2, 5.0)
    ax2.set_xlabel(r"$\theta\,/\,\theta_\lambda$")
    ax2.set_ylabel(r"$(b^{\rm froz}-b^{\rm prod})/\max|b^{\rm prod}|$ [%]")

    fig.tight_layout()
    out = FIG_DIR / "pedag_frozen_vs_prod_bsel.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
# Figure 3 analogue: DeltaSigma^prj(R), production (dashed) vs frozen (solid)
# ---------------------------------------------------------------------

def fig_delta_sigma_frozen_vs_prod(stack):
    sel = stack["sel"]
    dsp = DeltaSigmaPrj(stack["cosmo"], sel, stack["nfw"])
    R = np.geomspace(0.1, 30.0, 24)
    R_excl = R_lambda(LOB) * (1.0 + ZOB)
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(DELTAS)))

    pre_frozen = frozen_precomp_for(sel, stack, LOB, ZOB)
    orig_bp = sel.bias_precompute

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08})

    for dprj, c in zip(DELTAS, colours):
        ltr = LOB - dprj

        sel.bias_precompute = orig_bp
        orig_bsel = dsig._patch_bsel_for_ltr(dsp, LOB, ZOB, ltr)
        try:
            y_prod = dsp(R, LOB, ZOB)
        finally:
            dsig._restore_bsel(dsp, orig_bsel)

        sel.bias_precompute = lambda lob, zob: pre_frozen
        orig_bsel = dsig._patch_bsel_for_ltr(dsp, LOB, ZOB, ltr)
        try:
            y_froz = dsp(R, LOB, ZOB)
        finally:
            dsig._restore_bsel(dsp, orig_bsel)
            sel.bias_precompute = orig_bp

        ax1.loglog(R, y_prod, lw=1.6, color=c, ls="--", marker="o", ms=3.0,
                   alpha=0.9)
        ax1.loglog(R, y_froz, lw=1.6, color=c, ls="-", marker="s", ms=3.0,
                   alpha=0.9, label=rf"$\Delta^{{\rm prj}}\!=\!{dprj:g}$")
        rel = 100.0 * (y_froz / y_prod - 1.0)
        # near a sign crossing y_prod -> 0 and the fractional residual
        # diverges even though the absolute difference is tiny; drop
        # points close to the crossing (not a reformulation failure,
        # see caption)
        near_zero = np.abs(y_prod) < 0.05 * np.max(np.abs(y_prod))
        rel = np.where(near_zero, np.nan, rel)
        ax2.semilogx(R, rel, lw=1.5, color=c)

    ax1.plot([], [], ls="--", marker="o", ms=3.0, color="0.3",
             label="production (current repo)")
    ax1.plot([], [], ls="-", marker="s", ms=3.0, color="0.3",
             label="frozen reformulation")
    ax1.axvline(R_excl, ls=":", lw=0.8, color="0.3", alpha=0.8)
    ax1.set_ylabel(r"$\langle \Delta\Sigma^{\rm prj}(R\,|\,\lambda^{\rm ob},"
                  r"z^{\rm ob})\rangle\;[h\,M_\odot / {\rm Mpc}^2]$")
    ax1.set_title(rf"$\langle\Delta\Sigma^{{\rm prj}}\rangle$ (cl+LSS) at "
                 rf"$\lambda^{{\rm ob}}\!=\!{int(LOB)}$, $z^{{\rm ob}}\!=\!"
                 rf"{ZOB:.1f}$: production vs frozen reformulation")
    ax1.legend(loc="lower left", ncol=2, fontsize=8)

    ax2.axhline(0, color="k", lw=0.6)
    ax2.axvline(R_excl, ls=":", lw=0.8, color="0.3", alpha=0.8)
    ax2.set_xlim(0.1, 30.0)
    ax2.set_xlabel(r"$R\;[h^{-1}\,{\rm Mpc}]$")
    ax2.set_ylabel(r"$(\Delta\Sigma^{\rm froz}/\Delta\Sigma^{\rm prod}-1)$ [%]")

    fig.tight_layout()
    out = FIG_DIR / "pedag_frozen_vs_prod_delta_sigma.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    print("Building RichnessSelection stack (one-time CAMB/halofit cost)...")
    stack = build_stack()

    print("Figure 1 analogue: b_sel(theta), production vs frozen ...")
    out1 = fig_bsel_frozen_vs_prod(stack)
    print(f"  wrote {out1}")

    print("Figure 3 analogue: DeltaSigma^prj(R), production vs frozen ...")
    out2 = fig_delta_sigma_frozen_vs_prod(stack)
    print(f"  wrote {out2}")


if __name__ == "__main__":
    main()
