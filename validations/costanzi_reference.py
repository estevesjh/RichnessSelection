"""Verbatim port of the Costanzi SelectionBias notebook Sigma_prj
(cell 15: Sigma_prj_lobsel_CIRC / int_over_ThetaRs / int_over_M /
int_over_phi / twoD_prj_NFW), parametrised by convention flags so each
notebook-vs-package divergence can be flipped one at a time.

Flags (defaults = full Matteo convention):
  los      : "slab50" hard +-prj_depth cMpc/h, weight 1 (notebook)
             "wz"     photo-z parabolic kernel over its support (ours)
  excl     : "ball"   3D dis < R_excl sets the TOTAL integrand to 0,
                      plus the floor b*bsel*xi >= -1 (notebook)
             "slab"   zero xi only where theta < theta_excl(z) (ours)
             "none"
  tmap     : "chi"    kernel offset = theta * chi(ztr) (notebook, comoving)
             "DA"     kernel offset = theta * D_A(zob) (ours)
  nfw_kind : "m200m"  mass-conserving M200m NFW with c(M,z) (notebook)
             "cpp"    r200c, c=4, rho_eff = delta_c rho_crit Omega_m (ours)
  trunc    : per-halo hard cut of the NFW Sigma at r > trunc [cMpc/h]
             (notebook: 30.0); None = untruncated (ours)

Everything else (structure, grids, trapz, phi-integral with b_sel and
xi at the exact cluster-halo separation) follows the notebook verbatim.
The shared substrate (chi, dV, HMF, bias, xi_NL, b_sel) is taken from
the package stack -- byte-identical inputs isolate the Sigma_prj
machinery itself.

Units: returns Sigma in Msun h / (cMpc)^2 COMOVING (the notebook's
final /a^2 physical conversion is left out; convert the package side
with *1e12 to compare).
"""
from __future__ import annotations
import numpy as np

from richness_selection.geometry import R_lambda
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel
from richness_selection.nfw import RHO_CRIT_0


def _fun_wb(x):
    """Notebook `fun`: Wright & Brainerd projected NFW kernel f(x)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    hi = x > 1.0
    lo = x < 1.0
    out[hi] = (1.0 - 2.0 / np.sqrt(x[hi] ** 2 - 1.0)
               * np.arctan(np.sqrt((x[hi] - 1.0) / (x[hi] + 1.0)))) \
        / (x[hi] ** 2 - 1.0)
    out[lo] = (1.0 - 2.0 / np.sqrt(1.0 - x[lo] ** 2)
               * np.arctanh(np.sqrt((1.0 - x[lo]) / (1.0 + x[lo])))) \
        / (x[lo] ** 2 - 1.0)
    out[x == 1.0] = 1.0 / 3.0
    return out


def _c_of_m(M, z):
    """Notebook mass-concentration relation (M200m)."""
    return 10.14 * (np.asarray(M) / 2.0e12) ** -0.081 * (1.0 + z) ** -1.01


def _sigma_nfw(r, M, ztr, zob, stack, nfw_kind, trunc):
    """Sigma_NFW(r | M) [Msun h / cMpc^2], notebook or cpp convention.

    r : (Nth,) comoving offsets; M : (NM,).  Returns (Nth, NM).
    """
    r = np.atleast_1d(r)
    M = np.atleast_1d(M)
    if nfw_kind == "m200m":
        rho_m = stack["cosmo"].Om0 * RHO_CRIT_0
        r200 = (3.0 * M / (4.0 * np.pi * 200.0 * rho_m)) ** (1.0 / 3.0)
        c = _c_of_m(M, ztr)
        r_s = r200 / c
        f_c = np.log(1.0 + c) - c / (1.0 + c)
        rho_s = rho_m * (200.0 / 3.0) * c ** 3 / f_c
    else:  # "cpp"
        nfw = stack["nfw"]
        r_s, rho_eff = nfw._rs_and_rhos(M, zob)
        r_s = np.atleast_1d(r_s)
        rho_s = np.broadcast_to(np.atleast_1d(rho_eff), r_s.shape)
    x = r[:, None] / r_s[None, :]
    Sig = 2.0 * rho_s[None, :] * r_s[None, :] * _fun_wb(x).reshape(x.shape)
    if trunc is not None:
        Sig = np.where(r[:, None] > trunc, 0.0, Sig)
    return Sig


def sigma_prj_ref(R, lob, zob, stack, bsel_fn, *,
                  los="slab50", excl="ball", tmap="chi",
                  nfw_kind="m200m", trunc=30.0,
                  prj_depth=50.0, theta_max_s=30.0,
                  n_th=50, n_phi=50, n_M=50, n_los=50):
    """Notebook Sigma_prj_lobsel_CIRC with convention flags.

    Returns (total, cl): the [1 + b b_sel xi] and the [b b_sel xi]
    pieces [Msun h / cMpc^2, comoving], each shape (NR,).  Note: in
    ball-exclusion mode the floored interior contributes -1 to the cl
    piece (the notebook has no rnd/cl split; cl here = total - rnd).
    """
    cosmo = stack["cosmo"]; hmf = stack["hmf"]; bias = stack["bias"]
    xi_NL = stack["xi"]
    R = np.atleast_1d(np.asarray(R, dtype=float))
    chi_o = float(cosmo.chi(zob))
    R_excl = float(R_lambda(lob)) * (1.0 + zob)

    # ---- LoS grid (fg + bg, log-spaced in |Delta chi|, notebook) ----------
    if los == "slab50":
        d_lo, d_hi = 1.0e-6, prj_depth
    else:  # "wz": kernel support via the bisect bounds (photoz.py)
        from scipy.optimize import bisect
        z_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
        z_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
        d_lo = 1.0e-6
        d_hi = max(chi_o - float(cosmo.chi(z_lo)),
                   float(cosmo.chi(z_hi)) - chi_o)
    dis_g = np.geomspace(d_lo, d_hi, n_los)
    zs_ref = np.linspace(0.0, 2.0, 3000)
    chi_ref = cosmo.chi(zs_ref)
    z_of_chi = lambda c: np.interp(c, chi_ref, zs_ref)
    ztr = np.concatenate([z_of_chi(chi_o - dis_g)[::-1],
                          z_of_chi(chi_o + dis_g)])
    chi_tr = cosmo.chi(ztr)
    w_los = w_z(ztr, zob) if los == "wz" else np.ones_like(ztr)
    dV = cosmo.dV_dzdOm(ztr)

    # ---- theta grid (notebook: s/chi(zob), geomspace to 30 cMpc/h) --------
    thetas = np.geomspace(1.0e-6, theta_max_s, n_th) / chi_o

    # ---- mass grid ---------------------------------------------------------
    m_grid = np.logspace(13.0, 15.5, n_M)

    # ---- phi machinery: per theta, b_sel at the exact cluster-halo
    #      transverse separation Rtilde(phi) (notebook int_over_phi) --------
    phi = np.linspace(0.0, np.pi, n_phi)
    R_th = thetas[:, None] * chi_o                       # (Nth, 1)
    Rtil = np.sqrt(R_th[:, :, None] ** 2 + R[None, :, None] ** 2
                   - 2.0 * R_th[:, :, None] * R[None, :, None]
                   * np.cos(phi)[None, None, :])         # (Nth, NR, Nphi)
    th_til = Rtil / chi_o
    bsel_til = bsel_fn(th_til.ravel()).reshape(th_til.shape)

    out = np.zeros((ztr.size, R.size))
    out_cl = np.zeros((ztr.size, R.size))
    for iz, (z_t, c_t) in enumerate(zip(ztr, chi_tr)):
        n_m = hmf(m_grid, z_t)
        b_m = bias(m_grid, z_t)
        # kernel offsets
        r_off = thetas * (c_t if tmap == "chi" else chi_o / (1.0 + zob))
        Sig = _sigma_nfw(r_off, m_grid, z_t, zob, stack, nfw_kind, trunc)
        # 3D separations at each (theta, R, phi) via th_til
        dis3 = np.sqrt(c_t ** 2 + chi_o ** 2
                       - 2.0 * c_t * chi_o * np.cos(th_til))
        xiv = xi_NL(dis3, zob)
        if excl == "slab":
            # our convention: zero xi where theta < theta_excl(z_t)
            cos_e = np.clip((c_t ** 2 + chi_o ** 2 - R_excl ** 2)
                            / (2.0 * c_t * chi_o + 1e-30), -1.0, 1.0)
            th_e = 0.0 if cos_e >= 1.0 - 1e-12 else np.arccos(cos_e)
            xiv = np.where(th_til > th_e, xiv, 0.0)
        bbxi = (b_m[None, None, None, :]
                * (bsel_til * xiv)[:, :, :, None])   # (Nth, NR, Nphi, NM)
        if excl == "ball":
            bbxi = np.where(dis3[:, :, :, None] < R_excl, -1.0, bbxi)
        bbxi = np.maximum(bbxi, -1.0)                # notebook floor
        for piece, arr in (("total", 1.0 + bbxi), ("cl", bbxi)):
            int_phi = 2.0 * np.trapezoid(arr, phi, axis=2)   # (Nth, NR, NM)
            integ_M = ((m_grid * n_m)[None, None, :]
                       * Sig[:, None, :] * int_phi)
            int_M = np.trapezoid(integ_M, np.log(m_grid), axis=2)
            int_th = np.trapezoid(np.sin(thetas)[:, None] * int_M, thetas,
                                  axis=0)                    # (NR,)
            (out if piece == "total" else out_cl)[iz] = int_th
    tot = np.trapezoid(dV[:, None] * w_los[:, None] * out, ztr, axis=0)
    cl = np.trapezoid(dV[:, None] * w_los[:, None] * out_cl, ztr, axis=0)
    return tot, cl
