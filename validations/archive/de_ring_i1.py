"""Quick test: Takahasi-Mori DE quadrature on the ring vs GL, for I_1.

I_1's z-integrand has the twin-exclusion-peak + ring-plateau structure
of docs/richness_selection.tex Sec. "z-axis": two sharp peaks sitting
*exactly at* the ring's endpoints z_ob +/- dz_excl, with a flat plateau
between them. That is precisely the shape Takahasi-Mori DE quadrature
is built for (it clusters nodes doubly-exponentially at the interval
endpoints, the opposite of GL which is densest mid-interval).

This script does NOT touch SelBias._P_operator. It duplicates the
ring+outer z-grid construction (mirroring SelBias._z_grid) with a
`ring_mode` switch so only the ring's quadrature rule changes between
runs -- outer fg/bg (log-GL in |Delta chi|) and all inner (M, lambda,
theta) grids stay identical, and both are checked against a
scipy.quad reference (matched-inner, same convention as
tests/test_integrals.py::_quad_P1_I1_I2).
"""
import csv
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

from _common import CACHE_DIR

from richness_selection import Cosmology, PkGrid, HMF, Bias, MOR, XiNL
from richness_selection.sigma_m import SigmaM
from richness_selection.gl import gl_nodes
from richness_selection.de import de_nodes
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel

LOB, ZOB = 20.0, 0.5
NM, NLTR, NTH = 24, 60, 10


def f_inner(z, which, cosmo, stack, chi_o, theta_lob, R_excl, Ms, M_weight,
            lam_grid, wlam):
    """f_X(z): the (M, lambda, theta) inner integral at a single z.

    Same split-at-exclusion theta convention as SelBias._P_operator;
    duplicated here (not imported) so the quad truth is independent of
    production code, per the project's matched-inner-quad convention.
    """
    chi_z = float(cosmo.chi(z))
    dV = float(cosmo.dV_dzdOm(z))
    wz_val = float(w_z(np.array([z]), ZOB)[0])
    if wz_val <= 0:
        return 0.0
    cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
        2.0 * chi_z * chi_o + 1e-30)
    cos_excl = min(max(cos_excl, -1.0), 1.0)
    eps_theta = 1e-6
    th_lo = np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12 else eps_theta
    th_lo = max(th_lo, eps_theta)
    theta_max = 2.0 * theta_lob
    if th_lo >= theta_max:
        return 0.0
    ths, wth = gl_nodes(th_lo, theta_max, NTH)
    sin_th = np.sin(ths)
    th_weight = wth * 2.0 * np.pi * sin_th
    sigmoid = 1.0 / (1.0 + np.exp(-(2.5 / theta_lob) * (ths - 0.5 * theta_lob)))
    cos_th = np.cos(ths)
    dchi = np.sqrt(np.maximum(
        chi_z ** 2 + chi_o ** 2 - 2 * chi_z * chi_o * cos_th, 0.0))
    xi_th = stack['xi'](dchi, ZOB)
    theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
    fA = area_overlap(ths, theta_lob, theta_lam_l)
    if which == 'I2':
        ang = np.einsum('t,tL,t->L', th_weight, fA, xi_th)
    else:  # I1
        ang = np.einsum('t,t,tL,t->L', th_weight, sigmoid, fA, xi_th)
    P_lmz = stack['mor'].pdf(lam_grid[:, None], Ms[None, :], z)
    lam_int = np.einsum('L,LM,L->M', wlam, P_lmz, wz_val * lam_grid * ang)
    n_m = stack['hmf'](Ms, z)
    b_m = stack['bias'](Ms, z)
    return dV * np.sum(M_weight * n_m * b_m * lam_int)


def z_grid_variant(cosmo, Nz, ring_mode, chi_o, R_excl, z_fg_lo, z_bg_hi,
                    zs_ref, chi_ref, dchi_dz_ref):
    """SelBias._z_grid, duplicated, with a `ring_mode` switch.

    ring_mode='gl': plain GL in z on the ring (current production
    behaviour). ring_mode='de': Takahasi-Mori DE in z on the ring,
    same [z_ring_lo, z_ring_hi] bounds -- the peaks sit at those exact
    endpoints, so DE should resolve them with far fewer ring nodes.
    Outer fg/bg (log-GL in |Delta chi|) is untouched either way, so
    any difference in the final integral is attributable to the ring
    rule alone.
    """
    dchi_dz_at_zob = float(np.interp(ZOB, zs_ref, dchi_dz_ref))
    dz_excl = R_excl / dchi_dz_at_zob

    n_ring = max(9, Nz // 4)
    z_ring_lo = max(ZOB - dz_excl, z_fg_lo)
    z_ring_hi = min(ZOB + dz_excl, z_bg_hi)
    if ring_mode == 'gl':
        z_ring, w_ring = gl_nodes(z_ring_lo, z_ring_hi, n_ring)
    elif ring_mode == 'de':
        z_ring, w_ring = de_nodes(z_ring_lo, z_ring_hi, n_ring)
    else:
        raise ValueError(ring_mode)

    n_outer = max(15, (Nz - n_ring) // 2)
    dis_fg_max = chi_o - float(cosmo.chi(z_fg_lo))
    dis_bg_max = float(cosmo.chi(z_bg_hi)) - chi_o

    if R_excl < dis_fg_max:
        u_fg, w_u_fg = gl_nodes(np.log(R_excl), np.log(dis_fg_max), n_outer)
        dis_fg = np.exp(u_fg)
        z_fg = np.interp(chi_o - dis_fg, chi_ref, zs_ref)
        dchi_dz_fg = np.interp(z_fg, zs_ref, dchi_dz_ref)
        w_z_fg = w_u_fg * dis_fg / dchi_dz_fg
    else:
        z_fg = np.array([]); w_z_fg = np.array([])

    if R_excl < dis_bg_max:
        u_bg, w_u_bg = gl_nodes(np.log(R_excl), np.log(dis_bg_max), n_outer)
        dis_bg = np.exp(u_bg)
        z_bg = np.interp(chi_o + dis_bg, chi_ref, zs_ref)
        dchi_dz_bg = np.interp(z_bg, zs_ref, dchi_dz_ref)
        w_z_bg = w_u_bg * dis_bg / dchi_dz_bg
    else:
        z_bg = np.array([]); w_z_bg = np.array([])

    z_fg_sort = z_fg[::-1] if z_fg.size else z_fg
    w_fg_sort = w_z_fg[::-1] if w_z_fg.size else w_z_fg
    zs = np.concatenate([z_fg_sort, z_ring, z_bg])
    wzs = np.concatenate([w_fg_sort, w_ring, w_z_bg])
    return zs, wzs, n_ring


def integrate(cosmo, stack, which, Nz, ring_mode, chi_o, theta_lob, R_excl,
              Ms, M_weight, lam_grid, wlam, z_fg_lo, z_bg_hi, zs_ref,
              chi_ref, dchi_dz_ref):
    zs, wzs, n_ring = z_grid_variant(cosmo, Nz, ring_mode, chi_o, R_excl,
                                      z_fg_lo, z_bg_hi, zs_ref, chi_ref,
                                      dchi_dz_ref)
    vals = np.array([f_inner(z, which, cosmo, stack, chi_o, theta_lob,
                              R_excl, Ms, M_weight, lam_grid, wlam)
                      for z in zs])
    return float(np.sum(wzs * vals)), n_ring


def build_stack():
    """SelBias substrate only -- no NFW table needed for this test."""
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo); xi.build()
    return dict(cosmo=cosmo, pk=pk, sm=sm, hmf=hmf, bias=bias, mor=mor, xi=xi)


def main():
    stack = build_stack()
    cosmo = stack['cosmo']

    chi_o = float(cosmo.chi(ZOB))
    theta_lob = R_lambda(LOB) * (1.0 + ZOB) / chi_o
    R_excl = R_lambda(LOB) * (1.0 + ZOB)

    lnMs, wM = gl_nodes(np.log(1e13), np.log(10 ** 15.5), NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, LOB, NLTR)

    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(ZOB,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(ZOB,)))
    zs_ref = np.linspace(0.0, 2.0, 2000)
    chi_ref = cosmo.chi(zs_ref)
    dchi_dz_ref = np.gradient(chi_ref, zs_ref)

    print("Building scipy.quad reference (matched-inner) for I1, I2 ...")
    truth = {}
    for which in ('I1', 'I2'):
        v_fg, _ = quad(f_inner, z_fg_lo, ZOB,
                        args=(which, cosmo, stack, chi_o, theta_lob, R_excl,
                              Ms, M_weight, lam_grid, wlam),
                        epsrel=1e-6, limit=200)
        v_bg, _ = quad(f_inner, ZOB, z_bg_hi,
                        args=(which, cosmo, stack, chi_o, theta_lob, R_excl,
                              Ms, M_weight, lam_grid, wlam),
                        epsrel=1e-6, limit=200)
        truth[which] = v_fg + v_bg
        print(f"  quad {which} = {truth[which]:.6e}")

    rows = []
    print(f"\n{'Nz':>4} {'n_ring':>6} {'mode':>4} "
          f"{'I1 err %':>10} {'I2 err %':>10}")
    for Nz in (80, 48, 32, 24, 20, 16, 12):
        for ring_mode in ('gl', 'de'):
            errs = {}
            n_ring = None
            for which in ('I1', 'I2'):
                val, n_ring = integrate(
                    cosmo, stack, which, Nz, ring_mode, chi_o, theta_lob,
                    R_excl, Ms, M_weight, lam_grid, wlam, z_fg_lo, z_bg_hi,
                    zs_ref, chi_ref, dchi_dz_ref)
                errs[which] = abs(val / truth[which] - 1.0) * 100.0
            print(f"{Nz:>4d} {n_ring:>6d} {ring_mode:>4} "
                  f"{errs['I1']:>9.4f}% {errs['I2']:>9.4f}%")
            rows.append(dict(Nz=Nz, n_ring=n_ring, ring_mode=ring_mode,
                              err_I1_pct=errs['I1'], err_I2_pct=errs['I2']))

    out_path = os.path.join(CACHE_DIR, "de_ring_i1.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
