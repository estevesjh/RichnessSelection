"""Sum-rule hypothesis test: plateau(M_low) of the fixed cl integral.

Hypothesis: the ~0.67 amplitude deficit of Sigma_prj_cl against the
traditional two-halo target is the neighbour mass-budget sum rule
    int_{M_low} dM n(M) b(M) M  =  rho_m
violated by (i) the mass-range truncation at 1e13, masked by (ii) the
R_mis = theta * D_A_o map's (1+z)^2 measure overcount and (iii) the
untruncated NFW tail overcount.

Fixed kernel under test (validation-level, not yet wired into source):
  - comoving map: R_mis = theta * chi_o;
  - mass-normalised neighbour: rho_eff -> rho_eff / Omega_m
    (reconstructed halo carries M, not Omega_m * M);
  - transverse truncation at r_200: the neighbour deposits surface
    density only where the measurement ring intersects its disk,
    |R - R_mis| < r_200(M).

Prediction: plateau(M_low) tracks F(M_low) = int_{M_low} n b M dM /
rho_m and approaches the Tinker closure value (~0.95) by
M_low ~ 1e10-1e11 Msun/h.

Writes validations/cache/sumrule_Mlow.png.
"""
from __future__ import annotations
import os
from dataclasses import replace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import build_stack, CACHE_DIR

from richness_selection import SigmaPrj, TwoHalo
from richness_selection.nfw import RHO_CRIT_0

C_PIPE = "#0072B2"
C_RULE = "#D55E00"
C_MARK = "#666666"

LOB, ZOB = 20.0, 0.5
R_FIX = 10.0
M_LOW_GRID = np.array([1e10, 3e10, 1e11, 3e11, 1e12, 3e12, 1e13])


def fixed_cl_over_T(sp, sb, lob, zob, R_fix, T):
    """cl(R_fix)/T with the fixed kernel: comoving map, mass-normalised
    neighbour, r_200 transverse truncation."""
    R = np.array([R_fix])
    ctx = sp._build_zM_context(lob, zob)
    ctx["D_A_o"] = ctx["chi_o"]                 # comoving map
    pre = sb.bias_precompute(lob, zob)
    thetas, w_theta, _ = sp._theta_grid(lob, zob, R, ctx)
    bsel = sp._bsel_at(thetas, lob, zob, pre)

    chi_o = ctx["chi_o"]
    chi_z = ctx["chi_z"]
    outer_weight = ctx["outer_weight"]
    M_weight = ctx["M_weight"]
    n_mz = ctx["n_mz"]; bM_mz = ctx["bM_mz"]
    theta_excl_z = ctx["theta_excl_z"]
    rs_M = ctx["rs_M"]
    r200_M = rs_M * sp.nfw.c                    # r_200 = c * r_s

    # kernel: Sigma table, mass-normalised (1/Om0), comoving offsets
    Om0 = sp.cosmo.Om0
    base_kernel = sp._kernel_closure(R, ctx)    # uses ctx["D_A_o"]=chi_o

    total = 0.0
    for it, (th, wth) in enumerate(zip(thetas, w_theta)):
        Sig_MR = base_kernel(th)[:, 0] / Om0    # (NM,) mass-normalised
        # transverse truncation: ring R_fix intersects the neighbour's
        # r_200 disk only if |R - R_mis| < r_200(M)
        R_mis = th * chi_o
        Sig_MR = np.where(np.abs(R_fix - R_mis) < r200_M, Sig_MR, 0.0)
        wM_Sig = M_weight * Sig_MR
        cos_th = np.cos(th)
        dchi = np.sqrt(np.maximum(
            chi_z ** 2 + chi_o ** 2 - 2.0 * chi_z * chi_o * cos_th, 0.0))
        xi_v = sp.xi_NL(dchi, ZOB)
        xi_v = np.where(th > theta_excl_z, xi_v, 0.0)
        per_z = (wM_Sig @ (n_mz * bM_mz)) * outer_weight * xi_v
        total += wth * 2.0 * np.pi * np.sin(th) * bsel[it] * per_z.sum()
    return total / T


def main():
    stack = build_stack()
    cosmo, sb, nfw = stack["cosmo"], stack["sb"], stack["nfw"]
    hmf, bias = stack["hmf"], stack["bias"]

    rho_m = cosmo.Om0 * RHO_CRIT_0
    T = float(TwoHalo(cosmo, sb).sigma(np.array([R_FIX]), LOB, ZOB)[0])

    sp = SigmaPrj(cosmo, sb, nfw)
    # denser mass grid for the wide sweeps
    sp.grid = replace(sp.grid, NM=48)

    # analytic sum rule F(M_low) at zob
    lnM_f = np.linspace(np.log(1e10), np.log(10.0 ** sb.ln_M_max_log10), 800)
    M_f = np.exp(lnM_f)
    nbM = hmf(M_f, ZOB) * bias(M_f, ZOB) * M_f ** 2       # dM = M dlnM
    F_of_Mlow = np.array([
        np.trapezoid(np.where(M_f >= Ml, nbM, 0.0), lnM_f) / rho_m
        for Ml in M_LOW_GRID])

    plateaus = np.empty(M_LOW_GRID.size)
    saved_min = sb.min_mass4integral
    for i, Ml in enumerate(M_LOW_GRID):
        sb.min_mass4integral = Ml
        plateaus[i] = fixed_cl_over_T(sp, sb, LOB, ZOB, R_FIX, T)
        print(f"[sweep] M_low = {Ml:.1e}:  plateau = {plateaus[i]:.4f}"
              f"   sum rule F = {F_of_Mlow[i]:.4f}")
    sb.min_mass4integral = saved_min

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.semilogx(M_LOW_GRID, F_of_Mlow, color=C_RULE, lw=2, ls="--",
                label=r"sum rule  $\int_{M_{\rm low}} n\,b\,M\,dM/\rho_m$")
    ax.semilogx(M_LOW_GRID, plateaus, color=C_PIPE, lw=2, marker="o",
                ms=6, label=r"fixed pipeline  $\Sigma_{\rm cl}(R{=}10)/T$")
    ax.axhline(1.0, color="k", lw=1.2, ls=":")
    ax.annotate("target = 1", xy=(M_LOW_GRID[0] * 1.2, 1.015), fontsize=9)
    ax.axvline(1e13, color=C_MARK, lw=1, ls=":")
    ax.annotate("current default", xy=(1e13, 0.05),
                xycoords=("data", "axes fraction"), ha="center",
                fontsize=8, color=C_MARK)

    ax.set_xlabel(r"$M_{\rm low}$  [$M_\odot/h$]")
    ax.set_ylabel(r"amplitude / target")
    ax.set_ylim(0.0, 1.15)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(alpha=0.15)
    ax.set_title(
        rf"Sum-rule test: fixed kernel (comoving map, mass-normalised,"
        rf" $r_{{200}}$-truncated), $R={R_FIX:.0f}$ cMpc/$h$",
        fontsize=10)
    fig.tight_layout()

    out = os.path.join(CACHE_DIR, "sumrule_Mlow.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
