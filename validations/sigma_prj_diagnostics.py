"""Numerical audit of Sigma_prj(R | lob, zob).

Partner diagnostics script for the two-agent Sigma_prj audit.

Runs (all unbuffered):
  1. Current SigmaPrj(...) at R = {0.3, 1, 3, 10}.
  2. A reference via nested scipy.quad:
       outer quad over z, inner quad over theta (at fixed z, R).
       The M-integral uses a fixed GL-18 grid for speed.
  3. Diagnostic 1: N_theta(theta).
  4. Diagnostic 2: N_theta(theta) * Sigma_mis(R | M_bar, zob).
  5. theta/Nz/NM axis-isolation convergence.
  6. Exclusion step scan along z.

All outputs are CSV-style to stdout.
"""
from __future__ import annotations
import os, sys, time, functools
import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

sys.stdout.reconfigure(line_buffering=True)

NFW_TABLE_DIR = "/Users/esteves/Documents/Projetos/y3_cluster_cpp/data/nfw_off_center"
SRC = "/Users/esteves/Documents/github/RichnessSelection/src"
sys.path.insert(0, SRC)

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, SelBias, SigmaPrj,
)
from richness_selection.sigma_m import SigmaM
from richness_selection.xi_nl import XiNL
from richness_selection.geometry import R_lambda
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel
from richness_selection.gl import gl_nodes
from richness_selection.config import GridConfig


def p(*a, **k):
    print(*a, **k, flush=True)


def build_all():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sigma_m = SigmaM(pk)
    hmf = HMF(sigma_m)
    bias = Bias(sigma_m)
    mor = MOR()
    p("[setup] Building XiNL / halofit ...")
    t0 = time.time()
    xi = XiNL(cosmo); xi.build()
    p(f"[setup] XiNL built in {time.time()-t0:.1f}s")
    nfw = NFWMiscentered(cosmo, table_dir=NFW_TABLE_DIR)
    sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
    sp = SigmaPrj(cosmo, sb, nfw)
    return cosmo, hmf, bias, mor, xi, nfw, sb, sp


def shared_M_grid(NM=18, log10_Mmin=13.0, log10_Mmax=15.5):
    ln_M_min = np.log(10.0 ** log10_Mmin)
    ln_M_max = np.log(10.0 ** log10_Mmax)
    lnMs, wM = gl_nodes(ln_M_min, ln_M_max, NM)
    Ms = np.exp(lnMs)
    return Ms, wM * Ms


# -------- Quad reference: outer quad in z, inner quad in theta, GL in M -----
def quad_reference(lob, zob, R_vals, hmf, bias, xi, nfw, cosmo, sb,
                   use_bar_z=True, use_z_for_Rtheta=True,
                   NM=18, epsrel_theta=1e-3, epsrel_z=5e-3,
                   exclusion=True):
    """Reference via nested scipy.quad.

    Implements the draft equation (prj_model_draft.tex lines 164-167):

        <Sigma_prj(R)> = int dz dV/dzdOm w_z(z,zob)
                        int dtheta 2 pi sin(theta)
                          int dM n(M,z) [1 + b(M,z) b_sel(theta)
                                           xi_NL(|Delta r|, zbar)]
                          * Sigma_mis(R | M, z, R_mis(theta))

    with configurable:
      - use_bar_z: if True use zbar=(z+zob)/2 (draft convention),
                   else use zob (current code convention, line 160).
      - use_z_for_Rtheta: if True R_mis = theta*D_A(z) (draft),
                   else R_mis = theta*D_A(zob) (current code line 155).
    """
    Ms, M_weight = shared_M_grid(NM=NM)
    chi_o = float(cosmo.chi(zob))
    D_A_o = chi_o / (1.0 + zob)
    R_excl = R_lambda(lob) * (1.0 + zob)
    theta_max = 30.0 / D_A_o
    z_fg_lo = float(bisect(zmin4zkernel, -2., 2., args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2., 2., args=(zob,)))

    pre = sb.bias_precompute(lob, zob)

    # Cache b_sel spline on a fixed log-theta grid
    from scipy.interpolate import CubicSpline
    th_cache = np.geomspace(1e-8, theta_max, 300)
    bsel_cache = sb.b_sel_marginalised(th_cache, lob, zob, precomp=pre)
    bsel_spline = CubicSpline(np.log(th_cache), bsel_cache, extrapolate=False)

    def inner_theta_integrand(theta, z, R_val, chi_z, D_A_for_R):
        if theta <= 0.0:
            return 0.0
        dchi = np.sqrt(max(
            chi_z**2 + chi_o**2 - 2.0*chi_z*chi_o*np.cos(theta), 0.0))
        zxi = 0.5*(z + zob) if use_bar_z else zob
        xi_val = float(xi(np.array([dchi]), zxi).ravel()[0])
        if exclusion and dchi < R_excl:
            xi_val = 0.0
        bsel_th = float(bsel_spline(np.log(max(theta, 1e-9))))
        R_mis = np.array([theta * D_A_for_R])
        n_mz = hmf(Ms, z)
        b_mz = bias(Ms, z)
        S_mis = np.empty(Ms.size)
        for i, M in enumerate(Ms):
            S_mis[i] = nfw.sigma_grid(np.array([R_val]), R_mis,
                                       float(M), z).ravel()[0]
        bracket = 1.0 + b_mz * bsel_th * xi_val
        M_int = float(np.sum(M_weight * n_mz * bracket * S_mis))
        return 2.0*np.pi*np.sin(theta) * M_int

    def outer_z_integrand(z, R_val):
        wz_k = float(w_z(np.array([z]), zob)[0])
        if wz_k <= 0.0:
            return 0.0
        chi_z = float(cosmo.chi(z))
        dV = float(cosmo.dV_dzdOm(z))
        D_A_for_R = chi_z/(1.0 + z) if use_z_for_Rtheta else D_A_o
        # Split theta at the exclusion angle (if in band) AND at theta_R
        theta_R = max(R_val/D_A_for_R, 1e-7)
        if exclusion:
            cos_excl = (chi_z**2 + chi_o**2 - R_excl**2)/(2.0*chi_z*chi_o+1e-30)
            cos_excl = np.clip(cos_excl, -1.0, 1.0)
            th_lo = np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12 else 1e-7
        else:
            th_lo = 1e-7
        breakpoints = sorted({th_lo, theta_R, theta_max})
        val = 0.0
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            if a < b:
                vi, _ = quad(inner_theta_integrand, a, b,
                             args=(z, R_val, chi_z, D_A_for_R),
                             epsrel=epsrel_theta, limit=80)
                val += vi
        return wz_k * dV * val

    out = np.zeros_like(R_vals, dtype=float)
    for iR, R_val in enumerate(R_vals):
        t0 = time.time()
        v1, _ = quad(outer_z_integrand, z_fg_lo, zob, args=(R_val,),
                     epsrel=epsrel_z, limit=80)
        v2, _ = quad(outer_z_integrand, zob, z_bg_hi, args=(R_val,),
                     epsrel=epsrel_z, limit=80)
        out[iR] = v1 + v2
        p(f"  quad R={R_val:.2f}: val={out[iR]:.4e}  [{time.time()-t0:.1f}s]")
    return out


# -------- Diagnostic 1: N_theta(theta) --------
def N_theta(theta_vals, lob, zob, hmf, bias, xi, cosmo, sb,
            use_bar_z=True, apply_excl=True, NM=18, Nz=80):
    Ms, M_weight = shared_M_grid(NM=NM)
    chi_o = float(cosmo.chi(zob))
    R_excl = R_lambda(lob) * (1.0 + zob)
    z_fg_lo = float(bisect(zmin4zkernel, -2., 2., args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2., 2., args=(zob,)))
    zs_ref = np.linspace(0.0, 2.0, 2000)
    chi_ref = cosmo.chi(zs_ref)
    dchi_dz_ref = np.gradient(chi_ref, zs_ref)
    zs, wzs = sb._z_grid(lob, zob, Nz, chi_o, R_excl, z_fg_lo, z_bg_hi,
                         zs_ref, chi_ref, dchi_dz_ref)
    chi_z = cosmo.chi(zs)
    dV = cosmo.dV_dzdOm(zs)
    wz_k = w_z(zs, zob)

    pre = sb.bias_precompute(lob, zob)
    bsel_theta = sb.b_sel_marginalised(np.asarray(theta_vals), lob, zob,
                                        precomp=pre)
    # Precompute n_mz, b_mz across z
    n_zm = np.stack([hmf(Ms, z) for z in zs])   # (Nz, NM)
    b_zm = np.stack([bias(Ms, z) for z in zs])  # (Nz, NM)
    Msum = np.sum(M_weight * n_zm, axis=1)       # (Nz,): int n(M)
    Mbsum = np.sum(M_weight * n_zm * b_zm, axis=1)  # (Nz,): int n b

    out = np.zeros_like(theta_vals, dtype=float)
    for it, th in enumerate(theta_vals):
        sin_th = np.sin(th); cos_th = np.cos(th)
        dchi = np.sqrt(np.maximum(
            chi_z**2 + chi_o**2 - 2*chi_z*chi_o*cos_th, 0.0))
        zxi_arr = 0.5*(zs + zob) if use_bar_z else np.full_like(zs, zob)
        xi_vals = np.array([float(xi(np.array([dchi[i]]), zxi_arr[i])[0])
                            for i in range(zs.size)])
        if apply_excl:
            xi_vals = np.where(dchi < R_excl, 0.0, xi_vals)
        # integrand_per_z = Msum + bsel*xi*Mbsum
        ipz = Msum + bsel_theta[it] * xi_vals * Mbsum
        out[it] = 2*np.pi*sin_th * np.sum(wzs * dV * wz_k * ipz)
    return out


def main():
    p("=" * 70)
    p("SIGMA_PRJ NUMERICAL DIAGNOSTICS")
    p("=" * 70)
    cosmo, hmf, bias, mor, xi, nfw, sb, sp = build_all()
    lob, zob = 20.0, 0.5
    R_vals = np.array([0.3, 1.0, 3.0, 10.0])
    chi_o = float(cosmo.chi(zob))
    D_A_o = chi_o/(1.0 + zob)
    R_excl = R_lambda(lob) * (1.0 + zob)
    theta_lob_ob = R_lambda(lob)*(1.0 + zob)/chi_o
    theta_max = 30.0/D_A_o
    p(f"lob={lob}, zob={zob}, chi_o={chi_o:.3f}, D_A_o={D_A_o:.3f}, "
      f"R_excl={R_excl:.3f}, theta_lob={theta_lob_ob:.5f}, "
      f"theta_max={theta_max:.5f}")

    # Task 1 --------------------------------------------------------
    p()
    p("--- Task 1. current SigmaPrj vs quad reference ---")
    t0 = time.time()
    prof_code = sp(R_vals, lob, zob)
    p(f"SigmaPrj(R_vals) = {prof_code}  [{time.time()-t0:.1f}s]")

    p("quad ref A: code convention (xi at zob, R_mis = theta*D_A(zob)):")
    ref_code = quad_reference(lob, zob, R_vals, hmf, bias, xi, nfw, cosmo, sb,
                               use_bar_z=False, use_z_for_Rtheta=False,
                               NM=18)
    p("quad ref B: draft convention (xi at zbar, R_mis = theta*D_A(z)):")
    ref_draft = quad_reference(lob, zob, R_vals, hmf, bias, xi, nfw, cosmo, sb,
                                use_bar_z=True, use_z_for_Rtheta=True,
                                NM=18)

    p()
    p("CSV summary: R, code, ref_code_conv, ref_draft_conv, "
      "dCode-Ref_code/Ref_code, dCode-Ref_draft/Ref_draft")
    for i, R in enumerate(R_vals):
        c = prof_code[i]; r1 = ref_code[i]; r2 = ref_draft[i]
        p(f"{R:.2f},{c:.4e},{r1:.4e},{r2:.4e},"
          f"{(c-r1)/r1:+.4f},{(c-r2)/r2:+.4f}")

    # Task 2: N_theta  ---------------------------------------------
    p()
    p("--- Task 2. N_theta(theta) scan ---")
    theta_grid = np.geomspace(1e-4, 2.0*theta_max, 20)
    Ns_excl = N_theta(theta_grid, lob, zob, hmf, bias, xi, cosmo, sb,
                      use_bar_z=True, apply_excl=True, NM=14, Nz=60)
    Ns_noexcl = N_theta(theta_grid, lob, zob, hmf, bias, xi, cosmo, sb,
                         use_bar_z=True, apply_excl=False, NM=14, Nz=60)
    p("CSV: theta, theta/theta_max, N_theta(excl), N_theta(no_excl)")
    for i in range(theta_grid.size):
        p(f"{theta_grid[i]:.4e},{theta_grid[i]/theta_max:.4e},"
          f"{Ns_excl[i]:.4e},{Ns_noexcl[i]:.4e}")

    # Task 3: N_theta * Sigma_mis  ----------------------------------
    p()
    p("--- Task 3. N_theta(theta) * Sigma_mis(R | M=3e14, zob) ---")
    theta_fine = np.geomspace(1e-4, 2.0*theta_max, 30)
    Ns_fine = N_theta(theta_fine, lob, zob, hmf, bias, xi, cosmo, sb,
                       use_bar_z=True, apply_excl=True, NM=14, Nz=60)
    M_bar = 3e14
    R_mis_vals = theta_fine * D_A_o
    integ2 = np.zeros((len(R_vals), theta_fine.size))
    for iR, R_val in enumerate(R_vals):
        Smis = nfw.sigma_grid(np.array([R_val]), R_mis_vals,
                              float(M_bar), zob).ravel()
        integ2[iR, :] = Ns_fine * Smis

    p("CSV: theta, " + ",".join([f"R={R:.2f}" for R in R_vals]))
    for i in range(theta_fine.size):
        row = f"{theta_fine[i]:.4e}," + ",".join(
            [f"{integ2[iR,i]:.4e}" for iR in range(len(R_vals))])
        p(row)
    p()
    p("Per-R peak + half-max width:")
    for iR, R in enumerate(R_vals):
        theta_R = R/D_A_o
        ip = int(np.argmax(integ2[iR, :]))
        peak_theta = theta_fine[ip]
        peak = integ2[iR, ip]
        halfmask = integ2[iR, :] >= 0.5*peak
        if halfmask.any():
            lo = theta_fine[np.argmax(halfmask)]
            hi = theta_fine[len(halfmask)-1-np.argmax(halfmask[::-1])]
        else:
            lo = hi = peak_theta
        p(f"  R={R:.2f}: theta_R={theta_R:.4e}, peak_theta={peak_theta:.4e}, "
          f"peak/theta_R={peak_theta/theta_R:.3f}, half-max=[{lo:.4e},{hi:.4e}]")

    # Task 4: axis-isolation convergence ----------------------------
    p()
    p("--- Task 4. Axis-isolation convergence ---")
    p("(a) theta convergence: (n_inner, n_outer)")
    thconfigs = [(5,50),(10,150),(20,300),(40,600),(80,1200)]
    for (ni, no) in thconfigs:
        sp2 = SigmaPrj(cosmo, sb, nfw, n_theta_inner=ni, n_theta_outer=no)
        t0 = time.time()
        val = sp2(R_vals, lob, zob)
        p(f"  (ni={ni:3d}, no={no:4d}): {val}   [{time.time()-t0:.1f}s]")

    p()
    p("(b) Nz convergence (ni=10, no=150; NM=24)")
    for Nz in [40, 80, 160, 320]:
        sp2 = SigmaPrj(cosmo, sb, nfw)
        sp2.grid = GridConfig(Nz=Nz, NM=24, Nth=sb.grid.Nth)
        t0 = time.time()
        val = sp2(R_vals, lob, zob)
        p(f"  Nz={Nz:4d}: {val}   [{time.time()-t0:.1f}s]")

    p()
    p("(c) NM convergence (ni=10, no=150; Nz=80)")
    for NM in [12, 24, 48, 96]:
        sp2 = SigmaPrj(cosmo, sb, nfw)
        sp2.grid = GridConfig(Nz=80, NM=NM, Nth=sb.grid.Nth)
        t0 = time.time()
        val = sp2(R_vals, lob, zob)
        p(f"  NM={NM:3d}: {val}   [{time.time()-t0:.1f}s]")

    # Task 5: exclusion behaviour -----------------------------------
    p()
    p("--- Task 5. Exclusion-region behaviour ---")
    zs_ring = np.linspace(zob - 0.04, zob + 0.04, 9)
    for z in zs_ring:
        chi_z = float(cosmo.chi(z))
        cos_excl = (chi_z**2 + chi_o**2 - R_excl**2)/(2.0*chi_z*chi_o + 1e-30)
        cos_excl_c = np.clip(cos_excl, -1.0, 1.0)
        th_excl = np.arccos(cos_excl_c) if cos_excl_c < 1.0 - 1e-12 else 0.0
        p(f"  z={z:.3f}: chi_z-chi_o={chi_z-chi_o:+.4f},"
          f"  theta_excl={th_excl:.4e}")

    p()
    p("Step-scan xi*bracket across theta_excl for z in ring:")
    z_scan = [0.48, 0.495, 0.505, 0.52]
    th_scan = np.geomspace(1e-4, 2.0*theta_max, 25)
    header = "theta," + ",".join([f"xi_z={z:.3f}" for z in z_scan])
    p(header)
    for i, th in enumerate(th_scan):
        row_vals = [f"{th:.4e}"]
        for z in z_scan:
            chi_z = float(cosmo.chi(z))
            dchi = np.sqrt(max(chi_z**2+chi_o**2-2*chi_z*chi_o*np.cos(th),0.0))
            zbar = 0.5*(z+zob)
            xi_v = float(xi(np.array([dchi]), zbar)[0])
            if dchi < R_excl: xi_v = 0.0
            row_vals.append(f"{xi_v:+.3e}")
        p(",".join(row_vals))


if __name__ == "__main__":
    main()
