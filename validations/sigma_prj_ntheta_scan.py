"""Dense N_theta(theta) and N_theta(theta) * Sigma_mis(R, R_mis(theta)) scans.

These are the diagnostic integrands the audit notebook plots in its
bottom panel. Reference point: (lob=20, zob=0.5), matching
validations/sigma_prj_diag_results.md. Evaluates:

    N_theta(theta) = 2 pi sin(theta)
                     * int dz dV/dzdOm w_z(z, zob)
                     * int dM n(M, z) [1 + b(M, z) b_sel(theta) xi_NL(|Dr|, zob)]

    N_theta * Sigma_mis for R in {0.3, 1, 3, 10} cMpc/h at M_bar = 3e14.

Output
------
validations/cache/ntheta_per_R.npz
"""
from __future__ import annotations
import os
import sys

import numpy as np
from scipy.optimize import bisect

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _common import build_stack, CACHE_DIR  # noqa: E402

from richness_selection.geometry import R_lambda
from richness_selection.photoz import (
    w_z, sigma_z, zmin4zkernel, zmax4zkernel,
)
from richness_selection.gl import gl_nodes


LOB = 20.0
ZOB = 0.5
M_BAR = 3.0e14
R_VALS = np.array([0.3, 1.0, 3.0, 10.0])
NTHETA = 200
NM = 24
NZ = 80

OUT_NPZ = os.path.join(CACHE_DIR, "ntheta_per_R.npz")


def build_z_and_M(cosmo, sb, lob, zob):
    chi_o = float(cosmo.chi(zob))
    R_excl = R_lambda(lob) * (1.0 + zob)
    try:
        z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
        z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    except ValueError:
        sig = float(sigma_z(zob))
        z_fg_lo, z_bg_hi = max(0.01, zob - sig), zob + sig
    zs_ref = np.linspace(0.0, 2.0, 2000)
    chi_ref = cosmo.chi(zs_ref)
    dchi_dz_ref = np.gradient(chi_ref, zs_ref)
    zs, wzs = sb._z_grid(lob, zob, NZ, chi_o, R_excl, z_fg_lo, z_bg_hi,
                         zs_ref, chi_ref, dchi_dz_ref)
    lnMs, wM = gl_nodes(np.log(1e13), np.log(10 ** 15.5), NM)
    Ms = np.exp(lnMs)
    return zs, wzs, Ms, wM * Ms, chi_o, R_excl


def compute_Ntheta(cosmo, hmf, bias, xi, sb, zs, wzs, Ms, M_weight,
                   chi_o, R_excl, lob, zob, thetas, apply_excl=True):
    """Return (N_total, N_rnd, N_cl_lss, bsel) as arrays over theta.

    Per-theta breakdown:
        N_rnd(theta)     = 2 pi sin(theta) * int dz dV w_z * int dM n(M,z)
        N_cl_lss(theta)  = 2 pi sin(theta) * b_sel(theta)
                           * int dz dV w_z xi_NL(|Dr|, zob) 1[theta > theta_excl(z)]
                           * int dM n(M,z) b(M,z)
        N_total          = N_rnd + N_cl_lss
    """
    chi_z = cosmo.chi(zs)
    dV = cosmo.dV_dzdOm(zs)
    wz_k = w_z(zs, zob)
    n_zm = np.stack([hmf(Ms, z) for z in zs])      # (Nz, NM)
    b_zm = np.stack([bias(Ms, z) for z in zs])     # (Nz, NM)
    Msum = np.sum(M_weight * n_zm, axis=1)         # (Nz,) int n
    Mbsum = np.sum(M_weight * n_zm * b_zm, axis=1) # (Nz,) int n b

    pre = sb.bias_precompute(lob, zob)
    bsel = sb.b_sel_marginalised(thetas, lob, zob, precomp=pre)

    outer_w = wzs * dV * wz_k
    # N_rnd(theta) is only theta-dependent via sin(theta); the z,M factor
    # is constant in theta. Precompute it.
    rnd_zM_int = float(np.sum(outer_w * Msum))

    N_rnd = np.zeros_like(thetas)
    N_cl = np.zeros_like(thetas)
    for it, th in enumerate(thetas):
        sin_th = np.sin(th); cos_th = np.cos(th)
        dchi = np.sqrt(np.maximum(
            chi_z ** 2 + chi_o ** 2 - 2 * chi_z * chi_o * cos_th, 0.0))
        xi_vals = xi(dchi, zob)
        if apply_excl:
            xi_vals = np.where(dchi < R_excl, 0.0, xi_vals)
        cl_zM_int = float(np.sum(outer_w * xi_vals * Mbsum))
        prefac = 2.0 * np.pi * sin_th
        N_rnd[it] = prefac * rnd_zM_int
        N_cl[it] = prefac * bsel[it] * cl_zM_int
    N_total = N_rnd + N_cl
    return N_total, N_rnd, N_cl, bsel


def compute_sigma_mis_curves(nfw, cosmo, Ms, zob, thetas, R_vals, M_bar):
    """Sigma_mis(R_val, R_mis=theta*D_A(zob), M_bar, zob) for each R_val."""
    D_A_o = cosmo.D_A(zob)
    out = np.zeros((R_vals.size, thetas.size))
    for it, th in enumerate(thetas):
        R_theta = np.array([th * D_A_o])
        val = nfw.sigma_grid(R_vals, R_theta, M_bar, zob)  # (1, N_R)
        out[:, it] = val.ravel()
    return out


def main():
    stack = build_stack()
    cosmo = stack["cosmo"]; hmf = stack["hmf"]; bias = stack["bias"]
    xi = stack["xi"]; nfw = stack["nfw"]; sb = stack["sb"]

    chi_o = float(cosmo.chi(ZOB))
    D_A_o = chi_o / (1.0 + ZOB)
    R_excl = R_lambda(LOB) * (1.0 + ZOB)
    theta_lob = R_lambda(LOB) * (1.0 + ZOB) / chi_o
    theta_max = 30.0 / D_A_o
    theta_excl_ob = R_excl / chi_o
    theta_R = R_VALS / D_A_o

    thetas = np.geomspace(1e-5, 1.5 * theta_max, NTHETA)

    zs, wzs, Ms, M_weight, chi_o_chk, R_excl_chk = build_z_and_M(
        cosmo, sb, LOB, ZOB)
    print(f"[ntheta] zs.size={zs.size}, thetas.size={thetas.size}")

    (N_excl, N_rnd, N_cl, bsel) = compute_Ntheta(
        cosmo, hmf, bias, xi, sb, zs, wzs,
        Ms, M_weight, chi_o_chk, R_excl_chk,
        LOB, ZOB, thetas, apply_excl=True)
    (N_noexcl, _, _, _) = compute_Ntheta(
        cosmo, hmf, bias, xi, sb, zs, wzs,
        Ms, M_weight, chi_o_chk, R_excl_chk,
        LOB, ZOB, thetas, apply_excl=False)
    Sigma_mis = compute_sigma_mis_curves(
        nfw, cosmo, Ms, ZOB, thetas, R_VALS, M_bar=M_BAR)
    integrand = Sigma_mis * N_excl[None, :]  # (N_R, Ntheta)

    np.savez_compressed(
        OUT_NPZ,
        thetas=thetas,
        N_theta_excl=N_excl,
        N_theta_noexcl=N_noexcl,
        N_rnd=N_rnd,
        N_cl_lss=N_cl,
        bsel=bsel,
        Sigma_mis=Sigma_mis,
        integrand=integrand,
        R_vals=R_VALS,
        theta_R=theta_R,
        theta_lob=np.array(theta_lob),
        theta_excl_ob=np.array(theta_excl_ob),
        theta_max=np.array(theta_max),
        D_A_o=np.array(D_A_o),
        chi_o=np.array(chi_o),
        R_excl=np.array(R_excl),
        lob=np.array(LOB),
        zob=np.array(ZOB),
        M_bar=np.array(M_BAR),
    )
    print(f"[ntheta] wrote {OUT_NPZ}")

    # Quick N_rnd vs N_cl summary
    ratio = N_cl / np.maximum(N_rnd, 1e-30)
    i_eq = int(np.argmin(np.abs(ratio - 1.0))) if np.any(ratio > 1) else None
    print(f"\nN_rnd / N_cl_lss ratio at theta_lob: "
          f"{float(N_rnd[np.argmin(np.abs(thetas - theta_lob))]) / float(N_cl[np.argmin(np.abs(thetas - theta_lob))]):.2f}")
    print(f"N_rnd / N_cl_lss ratio at theta_max:   "
          f"{float(N_rnd[-1]) / max(float(N_cl[-1]), 1e-30):.2e}")
    if i_eq is not None and ratio[i_eq] > 0.9:
        print(f"N_cl crosses N_rnd near theta = {thetas[i_eq]:.3e} rad")

    peaks = []
    for iR, R in enumerate(R_VALS):
        i_peak = int(np.argmax(integrand[iR]))
        peaks.append((R, thetas[i_peak], theta_R[iR],
                      thetas[i_peak] / theta_R[iR]))

    print(f"\nPeak of N_theta * Sigma_mis(R):")
    print(f"{'R':>6s}  {'theta_R':>10s}  {'theta_peak':>10s}  "
          f"{'peak/theta_R':>12s}")
    for R, tp, tR, r in peaks:
        print(f"{R:>6.2f}  {tR:>10.3e}  {tp:>10.3e}  {r:>12.3f}")


if __name__ == "__main__":
    main()
