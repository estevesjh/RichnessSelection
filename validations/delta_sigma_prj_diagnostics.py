"""scipy.quad regression reference for <DeltaSigma^prj(R | lob, zob)>.

Mirrors the structure of ``sigma_prj_diagnostics.py::quad_reference``
but swaps ``nfw.sigma_grid`` for ``nfw.delta_sigma_grid`` in the inner
(theta, M) integrand.  Writes a markdown table + CSV to
``validations/cache/`` that the regression test in
``tests/test_delta_sigma_prj.py`` consumes.

Reference point: (lob=20, zob=0.5), R in {0.3, 1, 3, 10} cMpc/h.
Convention: "code" (xi at zob, R_mis = theta * D_A(zob)) -- matches the
production DeltaSigmaPrj we will ship.

Parallelism
-----------

The 4 R values x {total, cl_only} = 8 quad integrals are independent.
A ``multiprocessing.Pool(8)`` runs them concurrently; each worker builds
its own (cosmo, HMF, XiNL, nfw, SelBias) stack once on first call and
reuses it across its jobs.  Wall-clock scales ~1/N on 8 physical cores.

Output
------
validations/cache/delta_sigma_prj_quad_ref.csv
validations/cache/delta_sigma_prj_quad_ref.md
"""
from __future__ import annotations
import os
import sys
import time
import multiprocessing as mp

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _common import build_stack, CACHE_DIR  # noqa: E402

from richness_selection.geometry import R_lambda, theta_lambda
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel
from richness_selection.gl import gl_nodes


LOB = 20.0
ZOB = 0.5
R_VALS = np.array([0.3, 1.0, 3.0, 10.0])
NM_QUAD = 18
EPSREL_THETA = 3e-3
EPSREL_Z = 1e-2
R_MAX_CMPCH_DSIG = 30.0
N_WORKERS = 8

OUT_CSV = os.path.join(CACHE_DIR, "delta_sigma_prj_quad_ref.csv")
OUT_MD = os.path.join(CACHE_DIR, "delta_sigma_prj_quad_ref.md")


def shared_M_grid(NM=NM_QUAD, log10_Mmin=13.0, log10_Mmax=15.5):
    ln_M_min = np.log(10.0 ** log10_Mmin)
    ln_M_max = np.log(10.0 ** log10_Mmax)
    lnMs, wM = gl_nodes(ln_M_min, ln_M_max, NM)
    Ms = np.exp(lnMs)
    return Ms, wM * Ms


# -------- worker-local lazy stack -------------------------------------------
_WORKER_STACK = None   # type: ignore[var-annotated]


def _get_stack():
    """Build (or return cached) per-process stack.

    Called lazily on first job; subsequent jobs in the same worker
    reuse the cached stack so CAMB / halofit cost is paid once per
    process, not per job.
    """
    global _WORKER_STACK
    if _WORKER_STACK is None:
        print(f"[worker {os.getpid()}] building stack...", flush=True)
        t0 = time.time()
        _WORKER_STACK = build_stack()
        print(f"[worker {os.getpid()}] stack built in "
              f"{time.time()-t0:.1f}s", flush=True)
    return _WORKER_STACK


def _quad_one(args):
    """Run one (R_val, kind) quad integral in this worker.

    kind in {"total", "cl_only"}.
    """
    R_val, kind, lob, zob = args
    stack = _get_stack()
    cosmo = stack["cosmo"]
    hmf = stack["hmf"]
    bias = stack["bias"]
    xi = stack["xi"]
    nfw = stack["nfw"]
    sb = stack["sb"]

    Ms, M_weight = shared_M_grid()
    chi_o = float(cosmo.chi(zob))
    D_A_o = chi_o / (1.0 + zob)
    R_excl = R_lambda(lob) * (1.0 + zob)
    theta_max = R_MAX_CMPCH_DSIG / D_A_o
    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))

    pre = sb.bias_precompute(lob, zob)
    th_cache = np.geomspace(1e-8, theta_max, 300)
    bsel_cache = sb.b_sel_marginalised(th_cache, lob, zob, precomp=pre)
    bsel_spline = CubicSpline(np.log(th_cache), bsel_cache, extrapolate=False)

    # b_sel sigmoid breakpoints: the sigmoid pivots at theta_lambda/2
    # with slope 2.5/theta_lambda; we split the theta quadrature on
    # theta_lambda/2, theta_lambda, 2*theta_lambda so scipy.quad sees
    # a smooth integrand on each segment -- matches the production
    # _theta_grid breakpoint set (sigma_prj.py:178-192).
    theta_lam = float(theta_lambda(lob, zob, cosmo))

    # NFW per-M prefactor + scale radii (C++ convention: z-independent
    # at fixed c; see nfw.py module docstring).
    rs, rho_eff = nfw._rs_and_rhos(Ms, zob)
    lnx_R = np.clip(np.log(R_val / rs), nfw._lnx_lo, nfw._lnx_hi)  # (NM,)
    prefac_M = 2.0 * rs * rho_eff * 1.0e-12                         # (NM,)  Msun/h/pc^2
    _dsig_spl = nfw._dsig_spl
    _lnxmis_lo = nfw._lnxmis_lo
    _lnxmis_hi = nfw._lnxmis_hi

    def DS_mis_over_M(theta):
        R_mis = theta * D_A_o
        lnxmis = np.clip(np.log(R_mis / rs), _lnxmis_lo, _lnxmis_hi)
        gs = np.empty(Ms.size)
        for i in range(Ms.size):
            gs[i] = float(_dsig_spl(
                np.array([lnxmis[i]]), np.array([lnx_R[i]])).ravel()[0])
        return prefac_M * gs   # signed reconstruction: linear-space values

    def inner_integrand(theta, z, chi_z):
        if theta <= 0.0:
            return 0.0
        dchi = np.sqrt(max(
            chi_z ** 2 + chi_o ** 2 - 2.0 * chi_z * chi_o * np.cos(theta),
            0.0))
        xi_val = float(xi(np.array([dchi]), zob).ravel()[0])
        if dchi < R_excl:
            xi_val = 0.0
        bsel_th = float(bsel_spline(np.log(max(theta, 1e-9))))
        n_mz = hmf(Ms, z)
        b_mz = bias(Ms, z)
        DS_mis = DS_mis_over_M(theta)
        if kind == "total":
            bracket = 1.0 + b_mz * bsel_th * xi_val
        else:   # "cl_only"
            bracket = b_mz * bsel_th * xi_val
        M_int = float(np.sum(M_weight * n_mz * bracket * DS_mis))
        return 2.0 * np.pi * np.sin(theta) * M_int

    def outer_z_integrand(z):
        wz_k = float(w_z(np.array([z]), zob)[0])
        if wz_k <= 0.0:
            return 0.0
        chi_z = float(cosmo.chi(z))
        dV = float(cosmo.dV_dzdOm(z))
        theta_R = max(R_val / D_A_o, 1e-7)
        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = np.clip(cos_excl, -1.0, 1.0)
        th_lo = np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12 else 1e-7
        # Full breakpoint set matching the production _theta_grid:
        # exclusion edge, every requested theta_R, sigmoid pivot
        # theta_lambda/2, sigmoid widths theta_lambda and 2*theta_lambda,
        # outer truncation.  At R=0.3 the integrand has steep structure
        # where theta_R lands inside the sigmoid transition; splitting
        # on theta_lambda/2 breaks that into two smooth pieces.
        raw = [th_lo, theta_R,
               0.5 * theta_lam, theta_lam, 2.0 * theta_lam,
               theta_max]
        breakpoints = sorted({b for b in raw
                              if th_lo <= b <= theta_max})
        val = 0.0
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            if a < b:
                vi, _ = quad(inner_integrand, a, b,
                             args=(z, chi_z),
                             epsrel=EPSREL_THETA, limit=80)
                val += vi
        return wz_k * dV * val

    t0 = time.time()
    v1, _ = quad(outer_z_integrand, z_fg_lo, zob,
                 epsrel=EPSREL_Z, limit=80)
    v2, _ = quad(outer_z_integrand, zob, z_bg_hi,
                 epsrel=EPSREL_Z, limit=80)
    val = v1 + v2
    dt = time.time() - t0
    print(f"[worker {os.getpid()}] R={R_val:.2f} kind={kind}: "
          f"val={val:.4e}  [{dt:.1f}s]", flush=True)
    return (R_val, kind, val)


def main():
    print(f"[dsig-quad] launching {N_WORKERS} workers for "
          f"{len(R_VALS)} R values x 2 kinds = {2*len(R_VALS)} jobs",
          flush=True)
    t0 = time.time()
    jobs = [(R, kind, LOB, ZOB)
            for R in R_VALS for kind in ("total", "cl_only")]
    # ``spawn`` avoids fork-with-open-file-descriptor issues on macOS
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        results = pool.map(_quad_one, jobs)
    print(f"[dsig-quad] all jobs finished in {time.time()-t0:.1f}s",
          flush=True)

    total = np.zeros_like(R_VALS, dtype=float)
    cl = np.zeros_like(R_VALS, dtype=float)
    for R_val, kind, val in results:
        iR = int(np.argmin(np.abs(R_VALS - R_val)))
        if kind == "total":
            total[iR] = val
        else:
            cl[iR] = val
    rnd = total - cl

    with open(OUT_CSV, "w") as f:
        f.write("lob,zob,R_cMpch,DeltaSigma_total_quad,"
                "DeltaSigma_rnd_quad,DeltaSigma_cl_quad\n")
        for iR, R in enumerate(R_VALS):
            f.write(f"{LOB:.2f},{ZOB:.3f},{R:.3f},"
                    f"{total[iR]:.6e},{rnd[iR]:.6e},{cl[iR]:.6e}\n")
    print(f"[dsig-quad] wrote {OUT_CSV}", flush=True)

    with open(OUT_MD, "w") as f:
        f.write("# DeltaSigma^prj scipy.quad reference\n\n")
        f.write(f"Reference point: (lob={LOB}, zob={ZOB}).  "
                f"R_max_cMpch = {R_MAX_CMPCH_DSIG}.  NM_quad = {NM_QUAD}.  "
                f"epsrel: theta={EPSREL_THETA}, z={EPSREL_Z}.\n\n")
        f.write("| R [cMpc/h] | total | rnd (boundary term) | "
                "cl (default return) |\n")
        f.write("|---|---|---|---|\n")
        for iR, R in enumerate(R_VALS):
            f.write(f"| {R:.2f} | {total[iR]:.4e} | "
                    f"{rnd[iR]:.4e} | {cl[iR]:.4e} |\n")
    print(f"[dsig-quad] wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
