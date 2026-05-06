"""DeltaSigma^prj figures for docs/richness_selection.tex.

Generates three panels (Fig. 1 of the doc, pedag_bsel_theta, fixes
the pattern: fixed (lob, zob) = (20, 0.5), vary Delta^prj = lob - ltr):

    1. ``pedag_sigma_prj.png``       -- <Sigma^prj(R)> at (20, 0.5)
       with Delta^prj in {0, 3, 6, 10, 15}.
    2. ``pedag_delta_sigma_prj.png`` -- <DeltaSigma^prj(R)> on the
       same grid.
    3. ``pedag_delta_sigma_quad.png`` -- precision audit against the
       scipy.quad reference in
       validations/cache/delta_sigma_prj_quad_ref.csv (same reference
       point, marginalised over ltr as the production code does by
       default).

Run:
    RICHNESS_SELECTION_NFW_DIR=<path> \\
        /opt/homebrew/Caskroom/mambaforge/base/envs/astro/bin/python \\
        docs/make_delta_sigma_plots.py
"""
from __future__ import annotations
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, XiNL, SelBias,
    SigmaPrj, DeltaSigmaPrj,
)
from richness_selection.sigma_m import SigmaM
from richness_selection.geometry import R_lambda


LOB = 20.0
ZOB = 0.5
# Same Delta^prj sweep as Fig. 1 (pedag_bsel_theta).
DELTAS = [0.0, 3.0, 6.0, 10.0, 15.0]


def M_of_lob(mor, lob, zob, M_min_search=1e12, M_max_search=5e15):
    """Invert the HOD MOR <lambda>(M, z) = lob - 1 for M.

    Uses scipy.optimize.brentq on log M over a decade range that covers
    typical cluster masses.  `l_sat` is monotonic in M, so the root
    is unique.
    """
    from scipy.optimize import brentq
    target = float(lob) - 1.0        # l_sat = <lambda> - 1 (central)
    fn = lambda lnM: float(mor.l_sat(np.exp(lnM), zob)) - target
    return np.exp(brentq(fn, np.log(M_min_search), np.log(M_max_search)))


def nfw_sigma_centered(nfw, R, M, z):
    """Centred NFW Sigma(R) via the lookup at R_mis -> its table floor.

    At the smallest tabulated R_mis/R_s ~ 1e-2 the miscentered surface
    density tracks the centred one to <1% for R >~ R_mis.  This is the
    convention downstream code uses -- we go slightly above the table
    floor to stay inside the spline.
    """
    # Paper-eq-14 convention: int_0^{2pi} d phi Sigma_NFW(R) = 2 pi Sigma_NFW
    # since the integrand is phi-independent at R_mis=0.  nfw.sigma_grid
    # returns that same 2pi-integrated quantity.
    rs, _ = nfw._rs_and_rhos(M, z)
    # Use R_mis slightly above the table lower bound to avoid clipping.
    R_mis = np.array([np.exp(nfw._lnxmis_lo + 1e-3) * rs])
    return nfw.sigma_grid(np.asarray(R), R_mis, float(M), z).ravel()


def nfw_delta_sigma_centered(nfw, R, M, z):
    rs, _ = nfw._rs_and_rhos(M, z)
    R_mis = np.array([np.exp(nfw._lnxmis_lo + 1e-3) * rs])
    return nfw.delta_sigma_grid(np.asarray(R), R_mis, float(M), z).ravel()


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif", "STIXGeneral",
                   "Times New Roman"],
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
OCHRE = "#c58a2d"

FIG_DIR = Path(__file__).parent / "figs"
FIG_DIR.mkdir(exist_ok=True)

QUAD_CSV = (Path(__file__).parent.parent / "validations" / "cache"
            / "delta_sigma_prj_quad_ref.csv")


def build_stack():
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
    return dict(cosmo=cosmo, hmf=hmf, bias=bias, mor=mor,
                xi=xi, nfw=nfw, sel=sel)


def _patch_bsel_for_ltr(obj, lob, zob, ltr):
    """Patch obj._bsel_at to evaluate b_sel at a fixed true richness ltr
    (instead of marginalising over ltr).  Returns the original method
    so the caller can restore it.

    We do this by monkey-patching the instance method -- SigmaPrj /
    DeltaSigmaPrj calls ``self._bsel_at(thetas, lob, zob, precomp)``
    once per __call__ to evaluate b_sel on the theta-grid.  Replacing
    it with a closure that calls ``b_lob_theta(theta, ltr, zob, lob)``
    reproduces the Fig. 1 exercise (b_sel at fixed Delta^prj) inside
    the two-halo integral.
    """
    sel = obj.sel_bias
    original = obj._bsel_at

    def _bsel_fixed_ltr(thetas, lob_arg, zob_arg, precomp):
        return sel.b_lob_theta(thetas, ltr, zob_arg, lob_arg, precomp=precomp)

    obj._bsel_at = _bsel_fixed_ltr
    return original


def _restore_bsel(obj, original):
    obj._bsel_at = original


def _overlay_total(ax, stack, R, y_two_halo, which, lob=LOB):
    """Overlay the full one-halo + two-halo observable for a given lob.

    Plots NFW_centred(R | M(lob)) + y_two_halo as a dashed grey curve,
    where y_two_halo is the two-halo projection term already computed
    by the caller (expected to be the posterior mean over ltr).
    """
    mor = stack["mor"]
    nfw = stack["nfw"]
    kern = (nfw_sigma_centered if which == "sigma"
            else nfw_delta_sigma_centered)
    M = M_of_lob(mor, lob, ZOB)
    y_nfw = kern(nfw, R, M, ZOB)
    y_total = y_nfw + y_two_halo
    if which == "sigma":
        lab = (rf"$\Sigma_{{\rm NFW}}(M(\lambda^{{\rm ob}}\!=\!{int(lob)}))"
               rf" + \langle\Sigma^{{\rm prj}}\rangle$")
    else:
        lab = (rf"$\Delta\Sigma_{{\rm NFW}}(M(\lambda^{{\rm ob}}\!=\!{int(lob)}))"
               rf" + \langle\Delta\Sigma^{{\rm prj}}\rangle$")
    ax.loglog(R, y_total, lw=1.4, color="0.25", ls="--", alpha=0.95,
              label=lab)


def fig_sigma_prj(stack):
    """<Sigma^prj(R)> at fixed (lob, zob) = (20, 0.5), varying Delta^prj.

    Parallel to Fig. 1 (pedag_bsel_theta): same axes, same Delta^prj
    grid, same colour palette.  The coloured curves evaluate b_sel at
    fixed ltr inside the two-halo integral; the thick black curve is
    the production default (b_sel marginalised over ltr), and the
    grey dashed curve is the centred-NFW one-halo reference at
    M(lob=20) from the HOD MOR.
    """
    sp = SigmaPrj(stack["cosmo"], stack["sel"], stack["nfw"])
    R = np.geomspace(0.1, 30.0, 24)
    R_excl = R_lambda(LOB) * (1.0 + ZOB)
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(DELTAS)))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    # Fixed-ltr family first (to sit behind the marginalised curve).
    for dprj, c in zip(DELTAS, colours):
        ltr = LOB - dprj
        orig = _patch_bsel_for_ltr(sp, LOB, ZOB, ltr)
        try:
            y = sp(R, LOB, ZOB)
        finally:
            _restore_bsel(sp, orig)
        ax.loglog(R, y, lw=1.4, color=c, marker="o", ms=3.0, alpha=0.9,
                  label=rf"$\Delta^{{\rm prj}}\!=\!{dprj:g}$")
    # Posterior mean: default production output, integrates the
    # fixed-ltr family against P(ltr | lob, zob).
    y_marg = sp(R, LOB, ZOB)
    ax.loglog(R, y_marg, lw=2.2, color="black", marker="D", ms=3.5,
              label=r"posterior mean (default)")
    _overlay_total(ax, stack, R, y_marg, which="sigma")
    ax.axvline(R_excl, ls=":", lw=0.8, color="0.3", alpha=0.8)
    ax.text(R_excl * 1.05, ax.get_ylim()[0] * 1.6,
            r"$R_{\rm excl}$", color="0.3", fontsize=9)
    ax.set_xlabel(r"$R\;[h^{-1}\,{\rm Mpc}]$")
    ax.set_ylabel(r"$\langle \Sigma^{\rm prj}(R\,|\,\lambda^{\rm ob},"
                  r"z^{\rm ob})\rangle\;[h\,M_\odot / {\rm Mpc}^2]$")
    ax.set_title(rf"$\langle\Sigma^{{\rm prj}}\rangle$ at $\lambda^{{\rm ob}}"
                 rf"\!=\!{int(LOB)}$, $z^{{\rm ob}}\!=\!{ZOB:.1f}$;"
                 rf" varying $\Delta^{{\rm prj}}=\lambda^{{\rm ob}}"
                 rf"-\lambda^{{\rm tr}}$")
    ax.set_xlim(0.1, 30.0)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "pedag_sigma_prj.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_delta_sigma_prj(stack):
    """<DeltaSigma^prj(R)> at fixed (lob, zob) = (20, 0.5), varying Delta^prj.

    Parallel to Fig. 1 and fig_sigma_prj above, with the ltr-marginalised
    production default (thick black) and the centred-NFW one-halo
    reference at M(lob=20) from the HOD MOR.
    """
    dsp = DeltaSigmaPrj(stack["cosmo"], stack["sel"], stack["nfw"])
    R = np.geomspace(0.1, 30.0, 24)
    R_excl = R_lambda(LOB) * (1.0 + ZOB)
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, len(DELTAS)))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for dprj, c in zip(DELTAS, colours):
        ltr = LOB - dprj
        orig = _patch_bsel_for_ltr(dsp, LOB, ZOB, ltr)
        try:
            y = dsp(R, LOB, ZOB)
        finally:
            _restore_bsel(dsp, orig)
        ax.loglog(R, y, lw=1.4, color=c, marker="o", ms=3.0, alpha=0.9,
                  label=rf"$\Delta^{{\rm prj}}\!=\!{dprj:g}$")
    y_marg = dsp(R, LOB, ZOB)
    ax.loglog(R, y_marg, lw=2.2, color="black", marker="D", ms=3.5,
              label=r"posterior mean (default)")
    _overlay_total(ax, stack, R, y_marg, which="delta_sigma")
    ax.axvline(R_excl, ls=":", lw=0.8, color="0.3", alpha=0.8)
    ax.text(R_excl * 1.05, ax.get_ylim()[0] * 1.6,
            r"$R_{\rm excl}$", color="0.3", fontsize=9)
    ax.set_xlabel(r"$R\;[h^{-1}\,{\rm Mpc}]$")
    ax.set_ylabel(r"$\langle \Delta\Sigma^{\rm prj}(R\,|\,\lambda^{\rm ob},"
                  r"z^{\rm ob})\rangle\;[h\,M_\odot / {\rm Mpc}^2]$")
    ax.set_title(rf"$\langle\Delta\Sigma^{{\rm prj}}\rangle$ (cl+LSS) "
                 rf"at $\lambda^{{\rm ob}}\!=\!{int(LOB)}$, "
                 rf"$z^{{\rm ob}}\!=\!{ZOB:.1f}$;"
                 rf" varying $\Delta^{{\rm prj}}=\lambda^{{\rm ob}}"
                 rf"-\lambda^{{\rm tr}}$")
    ax.set_xlim(0.1, 30.0)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "pedag_delta_sigma_prj.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_delta_sigma_quad(stack):
    """Precision audit vs scipy.quad reference."""
    if not QUAD_CSV.exists():
        raise FileNotFoundError(
            f"{QUAD_CSV} not found.  Run "
            "validations/delta_sigma_prj_diagnostics.py first.")
    rows = np.loadtxt(QUAD_CSV, delimiter=",", skiprows=1)
    # columns: lob, zob, R, total_quad, rnd_quad, cl_quad
    R_ref = rows[:, 2]
    total_q = rows[:, 3]
    rnd_q = rows[:, 4]
    cl_q = rows[:, 5]

    # Production (pin R_max=30 to match the quad convention)
    dsp = DeltaSigmaPrj(stack["cosmo"], stack["sel"], stack["nfw"],
                        R_max_cMpch=30.0)
    dec = dsp(R_ref, 20.0, 0.5, return_decomposition=True)
    total_p, rnd_p, cl_p = dec["total"], dec["rnd"], dec["cl"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08})

    # --- top: absolute curves
    ax_top.loglog(R_ref, total_p, "o-", color=NAVY, lw=1.5, ms=5.0,
                  label="production, total")
    ax_top.loglog(R_ref, cl_p, "s-", color=CRIMSON, lw=1.5, ms=4.5,
                  label="production, cl")
    ax_top.loglog(R_ref, rnd_p, "^-", color=OCHRE, lw=1.3, ms=4.5, alpha=0.7,
                  label="production, rnd")
    ax_top.loglog(R_ref, total_q, "o", color=NAVY, mfc="white", ms=7.5,
                  mew=1.5, label="scipy.quad, total")
    ax_top.loglog(R_ref, cl_q, "s", color=CRIMSON, mfc="white", ms=7.0,
                  mew=1.5, label=r"scipy.quad, cl")
    ax_top.loglog(R_ref, rnd_q, "^", color=OCHRE, mfc="white", ms=7.0,
                  mew=1.5, alpha=0.7, label=r"scipy.quad, rnd")
    ax_top.set_ylabel(
        r"$\langle \Delta\Sigma^{\rm prj}\rangle\;[h\,M_\odot / {\rm Mpc}^2]$")
    ax_top.set_title(r"Production ${\tt DeltaSigmaPrj}$ vs "
                     r"${\tt scipy.quad}$ reference, "
                     r"$(\lambda^{\rm ob}, z^{\rm ob}) = (20, 0.5)$")
    ax_top.legend(loc="lower left", ncol=2, fontsize=8)
    ax_top.set_xlim(0.2, 15.0)

    # --- bottom: fractional residuals
    def _rel(p, q):
        return (p - q) / np.abs(q)
    ax_bot.axhline(0.0, color="0.4", lw=0.8, ls="-")
    ax_bot.axhline(0.005, color="0.7", lw=0.6, ls=":")
    ax_bot.axhline(-0.005, color="0.7", lw=0.6, ls=":")
    ax_bot.plot(R_ref, _rel(total_p, total_q), "o-", color=NAVY,
                lw=1.5, ms=5.0, label="total")
    ax_bot.plot(R_ref, _rel(cl_p, cl_q), "s-", color=CRIMSON,
                lw=1.5, ms=4.5, label="cl")
    ax_bot.plot(R_ref, _rel(rnd_p, rnd_q), "^-", color=OCHRE,
                lw=1.3, ms=4.5, alpha=0.7, label="rnd")
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"$R\;[h^{-1}\,{\rm Mpc}]$")
    ax_bot.set_ylabel(r"$({\rm prod}-{\rm quad})/{\rm quad}$")
    ax_bot.set_ylim(-0.03, 0.03)
    ax_bot.legend(loc="upper right", ncol=3, fontsize=8)
    out = FIG_DIR / "pedag_delta_sigma_quad.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    stack = build_stack()
    out1 = fig_sigma_prj(stack)
    print(f"wrote {out1}")
    out2 = fig_delta_sigma_prj(stack)
    print(f"wrote {out2}")
    out3 = fig_delta_sigma_quad(stack)
    print(f"wrote {out3}")


if __name__ == "__main__":
    main()
