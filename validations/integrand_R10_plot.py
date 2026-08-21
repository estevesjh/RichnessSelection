"""Convergence of the Sigma_prj cl integral at fixed R = 10 cMpc/h.

Plots the running integral

    I(<theta_max) = int_0^{theta_max} dtheta 2 pi sin(theta) b_sel(theta)
                    int dz [outer_weight xi_NL mask] int dM [n b Sigma_mis]

normalised by the traditional two-halo target
T = rho_m * b_sel_ls * C_xi(R) * 1e-12, as a function of theta_max.
If the pipeline realised the first-principles amplitude, the curve
would converge to 1; the plateau reads the actual amplitude ratio
(the known ~0.68 deficit).

The differential dI/dln(theta) (peak-normalised, light fill) shows
where the integral accumulates.  Markers: transverse comoving
s = theta * chi_o at s = R (comoving expectation), s = (1+zob) R
(where the R_mis = theta * D_A_o map puts the kernel peak), R_excl.

Writes validations/cache/integrand_R10.png.
"""
from __future__ import annotations
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import build_stack, CACHE_DIR

from richness_selection import SigmaPrj, TwoHalo
from richness_selection.geometry import R_lambda

C_DIFF = "#0072B2"     # differential (light fill)
C_CUM = "#D55E00"      # running integral
C_MARK = "#666666"

LOB, ZOB = 20.0, 0.5
R_FIX = 10.0


def main():
    stack = build_stack()
    cosmo, sb, nfw = stack["cosmo"], stack["sb"], stack["nfw"]

    # closure mode: comoving map, mass-conserving truncated NFW,
    # unresolved counter-term.  theta_max = 10 deg.
    from richness_selection import NFWMiscentered
    from _common import NFW_TABLE_DIR
    nfw_m = NFWMiscentered(cosmo, table_dir=NFW_TABLE_DIR, kind="m200m")
    chi_o_ = float(cosmo.chi(ZOB))
    sp = SigmaPrj(cosmo, sb, nfw_m, tmap="comoving", closure=True,
                  R_max_cMpch=np.deg2rad(10.0) * chi_o_)
    th2h = TwoHalo(cosmo, sb)
    T = float(th2h.sigma(np.array([R_FIX]), LOB, ZOB)[0])   # target norm

    R = np.array([R_FIX])
    ctx = sp._build_zM_context(LOB, ZOB)
    pre = sb.bias_precompute(LOB, ZOB)
    thetas, w_theta, info = sp._theta_grid(LOB, ZOB, R, ctx)
    bsel_vals = sp._bsel_at(thetas, LOB, ZOB, pre)
    ctx["Sigma_mis_per_theta"] = sp._kernel_closure_trunc(R, ctx)

    chi_o = ctx["chi_o"]
    M_weight = ctx["M_weight"]
    n_mz = ctx["n_mz"]; bM_mz = ctx["bM_mz"]
    theta_excl_z = ctx["theta_excl_z"]
    chi_z = ctx["chi_z"]
    outer_weight = ctx["outer_weight"]

    # per-theta cl contribution for R = R_FIX (z and M contracted)
    dI = np.zeros(thetas.size)
    for it, (th, wth) in enumerate(zip(thetas, w_theta)):
        Sig_MR = ctx["Sigma_mis_per_theta"](th)[:, 0]       # (NM,)
        wM_Sig = M_weight * Sig_MR
        cos_th = np.cos(th)
        dchi = np.sqrt(np.maximum(
            chi_z ** 2 + chi_o ** 2 - 2.0 * chi_z * chi_o * cos_th, 0.0))
        xi_v = sp.xi_NL(dchi, ZOB)
        xi_v = np.where(th > theta_excl_z, xi_v, 0.0)
        per_z = (wM_Sig @ (n_mz * bM_mz)) * outer_weight * xi_v
        dI[it] = wth * 2.0 * np.pi * np.sin(th) * bsel_vals[it] * per_z.sum()

    # counter-term: point-mass collapse at theta_R (comoving map)
    bsel_fn = sb.marginalised_bias(LOB, ZOB, precomp=pre)
    cl_c, _ = sp._closure_counter(R, ZOB, ctx, bsel_fn)
    theta_R_probe = float(R[0] / ctx["chi_o"])

    order = np.argsort(thetas)
    th_o, dI_o, w_o = thetas[order], dI[order], w_theta[order]
    th_deg = np.rad2deg(th_o)
    running = np.cumsum(dI_o) / T
    running += np.where(th_o >= theta_R_probe, float(cl_c[0]) / T, 0.0)
    dens_lnth = dI_o / w_o * th_o / T          # dI/dln(theta) / T
    print(f"[check] I(<theta_max_full) / T = {running[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.fill_between(th_deg, dens_lnth / dens_lnth.max(), 0.0,
                    color=C_DIFF, alpha=0.25, lw=0,
                    label=r"$dI/d\ln\theta$ (peak-normalised)")
    ax.semilogx(th_deg, running, color=C_CUM, lw=2.2,
                label=r"running integral $I(<\theta_{\max})\,/\,T$")
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.annotate("target = 1", xy=(2e-3, 1.02), fontsize=9)
    ax.axhline(running[-1], color=C_CUM, lw=1, ls=":")
    ax.annotate(f"plateau = {running[-1]:.3f}",
                xy=(2e-3, running[-1] + 0.02),
                fontsize=9, color=C_CUM)

    for s_val, lab, ls in ((R_FIX, r"$s=R$", "--"),
                           ((1 + ZOB) * R_FIX, r"$s=(1+z_{\rm ob})R$", ":"),
                           (float(R_lambda(LOB)) * (1 + ZOB),
                            r"$R_{\rm excl}$", ":")):
        xv = np.rad2deg(s_val / chi_o)
        ax.axvline(xv, color=C_MARK, lw=1, ls=ls)
        ax.annotate(lab, xy=(xv, 1.06), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8, color=C_MARK)

    def th2s(x):
        return np.deg2rad(x) * chi_o

    def s2th(x):
        return np.rad2deg(x / chi_o)

    secax = ax.secondary_xaxis("top", functions=(th2s, s2th))
    secax.set_xlabel(r"transverse comoving  $s=\theta_{\max}\chi_o$  [cMpc/$h$]",
                     fontsize=9)
    secax.tick_params(direction="in", labelsize=8)

    ax.set_xlabel(r"$\theta_{\max}$  [deg]")
    ax.set_xlim(1e-3, 10.0)
    ax.set_ylabel(
        r"$\int^{\theta_{\max}}\!d\theta\,\sin\theta \int dz\; I\;/\;"
        r"\rho_m b_{\rm sel}^{\rm ls} C_\xi(R)$")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    ax.tick_params(which="both", direction="in", right=True)
    ax.grid(alpha=0.15)
    ax.set_title(
        rf"Closure-mode $\Sigma^{{\rm prj}}_{{\rm cl}}$ convergence at $R={R_FIX:.0f}$ cMpc/$h$"
        rf"  ($\lambda^{{\rm ob}}={LOB:.0f}$, $z^{{\rm ob}}={ZOB}$)",
        fontsize=11)
    fig.tight_layout()

    out = os.path.join(CACHE_DIR, "integrand_R10.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
