"""Validate FrozenSelBias: scipy.quad truth + production fiducial.

Companion to docs/richness_selection_frozen.tex Sec. "Validation".
Three-way comparison at the DES-Y3-like reference points:

  (i)   scipy.quad reference on the original (z, theta, M, lambda)
        formulation (eq. step1 / eq. Pop), adaptive in z, matched
        inner grids, with the note's conventions: the random channel
        carries NO exclusion carve-out (eq. Dprj_rnd is the clean
        formula); the clustered operators carry exclusion through
        xi_mm = 0 inside R_excl (realised as the theta_excl(z) lower
        limit -- identical, since their integrand is proportional to
        xi_mm).
  (ii)  production ``SelBias`` (the FIDUCIAL method).  Note its P1
        applies the exclusion carve-out to the random channel, so its
        P1 error vs the doc-convention quad includes that (small,
        known) convention difference.
  (iii) ``FrozenSelBias`` (the frozen algorithm under test).

Also compares the lambda_tr-marginalised plateaus b_rm_ss / b_rm_ls
and the marginalised profile b_rm(theta) between frozen and fiducial
-- these differ BOTH by operator numerics and by the delta_prj
convention (Poisson denominator, eq. bls) -- and asserts the budget
identity eq. (budget) for the frozen closure.

Writes validations/cache/frozen_bsel_validation.csv.

Run from the repo root:
    python validations/frozen_bsel_validation.py
"""
import csv
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

from _common import CACHE_DIR

from richness_selection import (Cosmology, PkGrid, HMF, Bias, MOR, XiNL,
                                SelBias, FrozenSelBias)
from richness_selection.sigma_m import SigmaM
from richness_selection.gl import gl_nodes
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel

NM, NLTR, NTH = 24, 60, 10
POINTS = [(20.0, 0.5), (52.5, 0.425), (130.0, 0.575)]


def build_stack():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo); xi.build()
    return dict(cosmo=cosmo, pk=pk, hmf=hmf, bias=bias, mor=mor, xi=xi)


def quad_truth(stack, lob, zob):
    """Adaptive-z quad on eq. (step1)/(Pop), doc conventions."""
    cosmo = stack['cosmo']
    chi_o = float(cosmo.chi(zob))
    theta_lob = R_lambda(lob) * (1.0 + zob) / chi_o
    R_excl = R_lambda(lob) * (1.0 + zob)
    lnMs, wM = gl_nodes(np.log(1e13), np.log(10 ** 15.5), NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, lob, NLTR)
    theta_max = 2.0 * theta_lob
    eps_theta = 1e-6

    def f_inner(z, which):
        chi_z = float(cosmo.chi(z))
        dV = float(cosmo.dV_dzdOm(z))
        wz_val = float(w_z(np.array([z]), zob)[0])
        if wz_val <= 0:
            return 0.0
        if which == 'P1':
            # eq. (Dprj_rnd): clean random channel, no exclusion cut
            th_lo = eps_theta
        else:
            # exclusion via xi_mm = 0 for r < R_excl == theta_excl cut
            cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
                2.0 * chi_z * chi_o + 1e-30)
            cos_excl = min(max(cos_excl, -1.0), 1.0)
            th_lo = (np.arccos(cos_excl)
                     if cos_excl < 1.0 - 1e-12 else eps_theta)
            th_lo = max(th_lo, eps_theta)
            if th_lo >= theta_max:
                return 0.0
        ths, wth = gl_nodes(th_lo, theta_max, NTH)
        th_weight = wth * 2.0 * np.pi * np.sin(ths)
        sig = 1.0 / (1.0 + np.exp(
            -(2.5 / theta_lob) * (ths - 0.5 * theta_lob)))
        dchi = np.sqrt(np.maximum(
            chi_z ** 2 + chi_o ** 2 - 2 * chi_z * chi_o * np.cos(ths), 0.0))
        xi_th = stack['xi'](dchi, zob)
        theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
        fA = area_overlap(ths, theta_lob, theta_lam_l)
        if which == 'P1':
            ang = np.einsum('t,tL->L', th_weight, fA)
        elif which == 'I2':
            ang = np.einsum('t,tL,t->L', th_weight, fA, xi_th)
        else:
            ang = np.einsum('t,t,tL,t->L', th_weight, sig, fA, xi_th)
        P_lmz = stack['mor'].pdf(lam_grid[:, None], Ms[None, :], z)
        lam_int = np.einsum('L,LM,L->M', wlam, P_lmz,
                            wz_val * lam_grid * ang)
        n_m = stack['hmf'](Ms, z)
        if which == 'P1':
            return dV * np.sum(M_weight * n_m * lam_int)
        b_m = stack['bias'](Ms, z)
        return dV * np.sum(M_weight * n_m * b_m * lam_int)

    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    out = {}
    for which in ('P1', 'I1', 'I2'):
        v_fg, _ = quad(f_inner, z_fg_lo, zob, args=(which,),
                       epsrel=1e-6, limit=200)
        v_bg, _ = quad(f_inner, zob, z_bg_hi, args=(which,),
                       epsrel=1e-6, limit=200)
        out[which] = v_fg + v_bg
    return out


def main():
    print("Building stack (one-time CAMB/halofit cost)...")
    stack = build_stack()
    args = (stack['cosmo'], stack['pk'], stack['hmf'], stack['bias'],
            stack['mor'])
    sel = SelBias(*args, xi_nl=stack['xi'])          # fiducial
    fsel = FrozenSelBias(*args, xi_nl=stack['xi'])   # under test

    rows = []
    hdr = (f"{'lob':>6} {'zob':>6} {'source':>10} "
           f"{'P1 err':>10} {'I1 err':>10} {'I2 err':>10}")
    print("\n--- operators vs scipy.quad (doc conventions) ---")
    print(hdr)
    print("-" * len(hdr))
    for lob, zob in POINTS:
        truth = quad_truth(stack, lob, zob)
        for name, obj in (("fiducial", sel), ("frozen", fsel)):
            pre = obj.bias_precompute(lob, zob)
            errs = {k: abs(pre[k] / truth[k] - 1.0) * 100.0
                    for k in ('P1', 'I1', 'I2')}
            note = (" (P1 incl. carve-out convention)"
                    if name == "fiducial" else "")
            print(f"{lob:6.1f} {zob:6.3f} {name:>10} "
                  f"{errs['P1']:9.4f}% {errs['I1']:9.4f}% "
                  f"{errs['I2']:9.4f}%{note}")
            rows.append(dict(lob=lob, zob=zob, source=name,
                             P1=pre['P1'], I1=pre['I1'], I2=pre['I2'],
                             P1_quad=truth['P1'], I1_quad=truth['I1'],
                             I2_quad=truth['I2'],
                             P1_err_pct=errs['P1'],
                             I1_err_pct=errs['I1'],
                             I2_err_pct=errs['I2']))

    print("\n--- marginalised plateaus: frozen vs fiducial ---")
    print("(differences include BOTH operator numerics and the "
          "Poisson delta_prj convention of eq. bls)")
    for lob, zob in POINTS:
        pf = sel.plateaus(lob, zob)
        pz = fsel.plateaus(lob, zob)
        vec_dev = float(np.max(np.abs(
            pz.b_rm_ss_ltr_vec / pf.b_rm_ss_ltr_vec - 1.0)))
        print(f"lob={lob:6.1f} zob={zob:5.3f}  "
              f"b_rm_ss: {pf.b_rm_ss:8.4f} -> {pz.b_rm_ss:8.4f}  "
              f"b_rm_ls: {pf.b_rm_ls:7.4f} -> {pz.b_rm_ls:7.4f}  "
              f"max per-ltr ss dev: {100*vec_dev:.2f}%")

    print("\n--- budget closure (eq. budget), frozen ---")
    lob, zob = POINTS[0]
    pre = fsel.bias_precompute(lob, zob)
    worst_budget = 0.0
    for ltr in (18.0, 15.0, 10.0, 5.0):
        pr = fsel.bias_from_precomp(pre, ltr)
        budget = (pre["P1"] + pr["b_infty"] * pre["I1"]
                  + pr["b_zero"] * (pre["I2"] - pre["I1"]))
        worst_budget = max(worst_budget, abs(budget - (lob - ltr)))
    print(f"max |budget - (lob-ltr)| = {worst_budget:.2e}")

    out = os.path.join(CACHE_DIR, "frozen_bsel_validation.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    worst = max(max(r['P1_err_pct'], r['I1_err_pct'], r['I2_err_pct'])
                for r in rows if r['source'] == 'frozen')
    ok = worst < 0.1 and worst_budget < 1e-9
    print(f"worst frozen operator error: {worst:.4f}%  "
          f"({'PASS' if ok else 'FAIL'} vs 0.1% tolerance)")


if __name__ == "__main__":
    main()
