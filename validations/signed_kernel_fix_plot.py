"""Figure: the signed DeltaSigma_mis kernel fix, against the broken
ln-table kernel and the traditional two-halo reference.

Writes validations/cache/signed_kernel_fix.png with two panels:
  top    -- DeltaSigma_2h,cl(R) at (lob=20, zob=0.5): old ln-table
            kernel (sign-broken), production signed kernel, frozen
            factorisation (markers), and the TwoHalo reference
            rho_m * b_sel_ls * Delta[C_xi].
  bottom -- same curves as ratios to the TwoHalo reference.

Run:  RICHNESS_SELECTION_NFW_DIR=... python validations/signed_kernel_fix_plot.py
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from _common import build_stack, CACHE_DIR, NFW_TABLE_DIR

from richness_selection import DeltaSigmaPrj, FrozenDeltaSigmaPrj, TwoHalo

# Okabe-Ito colorblind-safe hues, fixed assignment per entity.
C_OLD = "#999999"      # broken ln-table kernel (muted -- it is the bug)
C_NEW = "#0072B2"      # production signed kernel
C_REF = "#D55E00"      # TwoHalo reference
C_FRZ = "#000000"      # frozen markers

LOB, ZOB = 20.0, 0.5


class _LnTableShim:
    """Emulate the retired ln-space DeltaSigma lookup: exp(spline)."""

    def __init__(self, spl):
        self._spl = spl

    def __call__(self, a, b):
        return np.exp(self._spl(a, b))


def main():
    stack = build_stack()
    cosmo, sb, nfw = stack["cosmo"], stack["sb"], stack["nfw"]

    R = np.geomspace(2.0, 20.0, 25)

    dsp = DeltaSigmaPrj(cosmo, sb, nfw)
    fz = FrozenDeltaSigmaPrj(cosmo, sb, nfw)
    th = TwoHalo(cosmo, sb)

    cl_new = dsp(R, LOB, ZOB)
    cl_frz = fz(R, LOB, ZOB)
    ds_ref = th.delta_sigma(R, LOB, ZOB)

    # Old kernel: shipped ln-table, exp on lookup (the retired behaviour).
    log_dsig = np.loadtxt(os.path.join(
        NFW_TABLE_DIR, "table_1000_1e-03_5e+03_log_deltasigma_single.txt"))
    lnxmis = nfw._log_xmis[: log_dsig.shape[0]]
    lnx = nfw._log_x[: log_dsig.shape[1]]
    old_spl = _LnTableShim(RectBivariateSpline(
        lnxmis, lnx, log_dsig, kx=1, ky=1))
    dsp_old = DeltaSigmaPrj(cosmo, sb, nfw)
    dsp_old.nfw = type(nfw).__new__(type(nfw))     # shallow shim carrier
    dsp_old.nfw.__dict__.update(nfw.__dict__)
    dsp_old.nfw._dsig_spl = old_spl
    cl_old = dsp_old(R, LOB, ZOB)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.4, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.08))

    ax.loglog(R, cl_old, ls=":", lw=2, color=C_OLD,
              label="old ln-table kernel (sign-broken)")
    ax.loglog(R, ds_ref, ls="--", lw=2, color=C_REF,
              label=r"TwoHalo ref: $\rho_m\,b_{\rm sel}\,\Delta[C_\xi]$")
    ax.loglog(R, cl_new, ls="-", lw=2, color=C_NEW,
              label="signed kernel (production)")
    ax.loglog(R[::3], cl_frz[::3], ls="none", marker="o", ms=6,
              mfc="none", mec=C_FRZ, mew=1.2,
              label="frozen factorisation")
    ax.set_ylabel(r"$\Delta\Sigma_{\rm 2h,cl}\;[M_\odot h\,/\,{\rm pc}^2]$")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.15, which="major")

    axr.semilogx(R, cl_old / ds_ref, ls=":", lw=2, color=C_OLD)
    axr.semilogx(R, cl_new / ds_ref, ls="-", lw=2, color=C_NEW)
    axr.semilogx(R[::3], (cl_frz / ds_ref)[::3], ls="none", marker="o",
                 ms=6, mfc="none", mec=C_FRZ, mew=1.2)
    axr.axhline(1.0, color=C_REF, ls="--", lw=1.5)
    axr.set_ylim(0.4, 1.6)
    axr.set_xlabel(r"$R\;[{\rm cMpc}/h]$")
    axr.set_ylabel("ratio to TwoHalo")
    axr.tick_params(which="both", direction="in", top=True, right=True)
    axr.grid(alpha=0.15, which="major")

    fig.suptitle(
        rf"$\Delta\Sigma^{{\rm prj}}_{{\rm cl}}$ kernel fix"
        rf"  ($\lambda^{{\rm ob}}={LOB:.0f}$, $z^{{\rm ob}}={ZOB}$)",
        y=0.945, fontsize=11)

    out = os.path.join(CACHE_DIR, "signed_kernel_fix.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
