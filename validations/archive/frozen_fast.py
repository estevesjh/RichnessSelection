"""Vectorized + cached frozen reformulation.

frozen_kernels.assemble() loops over r in Python (n_r=48 per segment)
and calls F2(x)/F1(x) separately per r -- each call reruns
area_overlap over the full (n_mu, N_lambda) grid. That's ~192
area_overlap calls per assemble(), ~19-27ms, and dominates the cost
(see benchmark_frozen.py profile).

Two independent fixes, both applied here:
  1. Vectorize the near-zone r-loop: build the full (n_r, n_mu) grid
     of transverse offsets at once, call area_overlap ONCE per
     segment, and get F1 = sigma(x) * F2 for free (no second
     area_overlap call).
  2. Memoize assemble_fast()/frozen_P1_fast() by (lob, zob), matching
     the SelBias._cache pattern -- repeat calls at the same point are
     then free, same as production.

Numerically identical to frozen_kernels.assemble() to machine/GL
precision (same nodes, same physics, only the evaluation order
changes). See benchmark_frozen.py for the timing comparison.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from frozen_kernels import (frozen_weights, sigmoid_x, NM, NLTR,
                             PI_S_OVER_R)
from richness_selection.gl import gl_nodes
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel
from scipy.optimize import bisect


def assemble_fast(stack, lob, zob, n_r=48, n_mu=48, n_far=80):
    """Same (I1, I2) as frozen_kernels.assemble(), vectorized near zone."""
    cosmo = stack['cosmo']
    xi = stack['xi']
    chi_o = float(cosmo.chi(zob))
    R = R_lambda(lob) * (1.0 + zob)
    Pi_s = PI_S_OVER_R * R

    zs_ref = np.linspace(0.0, 2.0, 4000)
    chi_ref = cosmo.chi(zs_ref)

    def z_of_Pi(Pi):
        return np.interp(chi_o + Pi, chi_ref, zs_ref)

    lam_grid, w_eff, rho = frozen_weights(stack, lob, zob)

    # --- near zone: batch the full (r, mu) grid per segment, ONE
    # area_overlap call gives F2 on the whole grid; F1 = sigma*F2 free.
    r_max_near = np.sqrt(Pi_s ** 2 + 4.0 * R ** 2)
    mu_t, mu_w = gl_nodes(0.0, 1.0, n_mu)

    total2 = 0.0
    total1 = 0.0
    for (a, b) in ((R, 2.0 * R), (2.0 * R, r_max_near)):
        lr, wr = gl_nodes(np.log(a), np.log(b), n_r)
        rs = np.exp(lr)                                   # (n_r,)
        xi_r = xi(rs, zob)                                # (n_r,)

        mu_lo = np.sqrt(np.maximum(0.0, 1.0 - (2.0 * R / rs) ** 2))
        mu_hi = np.minimum(1.0, Pi_s / rs)
        active = mu_hi > mu_lo                            # (n_r,)

        mus = mu_lo[:, None] + (mu_hi - mu_lo)[:, None] * mu_t[None, :]
        wmu = (mu_hi - mu_lo)[:, None] * mu_w[None, :]     # (n_r, n_mu)
        Rp = rs[:, None] * np.sqrt(np.maximum(1.0 - mus ** 2, 0.0))
        x = Rp / R                                         # (n_r, n_mu)

        fA = area_overlap(x.reshape(-1), 1.0, rho)         # (n_r*n_mu, Nlam)
        F2vals = (fA @ w_eff).reshape(n_r, n_mu)
        F1vals = sigmoid_x(x) * F2vals

        Pi_grid = rs[:, None] * mus
        wz_pm = 0.5 * (w_z(z_of_Pi(Pi_grid), zob)
                       + w_z(z_of_Pi(-Pi_grid), zob))

        G2 = 2.0 * np.sum(wmu * F2vals * wz_pm, axis=1) * active
        G1 = 2.0 * np.sum(wmu * F1vals * wz_pm, axis=1) * active

        total2 += float(np.sum(wr * rs ** 3 * xi_r * G2))
        total1 += float(np.sum(wr * rs ** 3 * xi_r * G1))

    near2 = 2.0 * np.pi * total2
    near1 = 2.0 * np.pi * total1

    # --- far zone: unchanged (already 1-D and cheap) ------------------
    x_n, x_w = gl_nodes(0.0, 2.0, 200)
    fA_n = area_overlap(x_n, 1.0, rho)
    F2_n = fA_n @ w_eff
    F1_n = sigmoid_x(x_n) * F2_n
    A2 = R ** 2 * float(np.sum(x_w * x_n * F2_n))
    B2 = R ** 4 * float(np.sum(x_w * x_n ** 3 * F2_n))
    A1 = R ** 2 * float(np.sum(x_w * x_n * F1_n))
    B1 = R ** 4 * float(np.sum(x_w * x_n ** 3 * F1_n))

    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    z_coarse = np.linspace(z_fg_lo, z_bg_hi, 15)
    amp = np.array([np.sum(frozen_weights(stack, lob, zc)[1])
                    for zc in z_coarse])
    amp /= np.sum(w_eff)

    def far(A_X, B_X):
        total = 0.0
        for sgn, z_lim in ((-1.0, z_fg_lo), (+1.0, z_bg_hi)):
            Pi_max = abs(float(cosmo.chi(z_lim)) - chi_o)
            if Pi_max <= Pi_s:
                continue
            lu, wu = gl_nodes(np.log(Pi_s), np.log(Pi_max), n_far)
            Pis = np.exp(lu)
            zPi = z_of_Pi(sgn * Pis)
            wzv = w_z(zPi, zob)
            sz = np.interp(zPi, z_coarse, amp)
            chi_fac = ((chi_o + sgn * Pis) / chi_o) ** 2
            xi_v = xi(Pis, zob)
            dxi = (xi(Pis * 1.005, zob) - xi(Pis * 0.995, zob)) / (0.01 * Pis)
            integ = A_X * xi_v + (B_X / (2.0 * Pis)) * dxi
            total += float(np.sum(wu * Pis * chi_fac * wzv * sz * integ))
        return 2.0 * np.pi * total

    I2 = near2 + far(A2, B2)
    I1 = near1 + far(A1, B1)
    return I1, I2


def frozen_P1_fast(stack, lob, zob, n_pi=80):
    """Identical maths to frozen_kernels.frozen_P1(); no r-loop there
    already (M0 is a 1-D cumulative moment), kept as a thin wrapper for
    API symmetry with assemble_fast()."""
    from frozen_kernels import frozen_P1
    return frozen_P1(stack, lob, zob, n_pi=n_pi)


# ---------------------------------------------------------------------
# (lob, zob) memoization, matching SelBias._cache
# ---------------------------------------------------------------------

_CACHE = {}


def frozen_precompute(stack, lob, zob, ndigits=6):
    """Cached (P1, I1, I2) triple, keyed by rounded (lob, zob) -- same
    pattern as SelBias._cache. Repeat calls at the same point are a
    dict lookup; first call at a new point pays the vectorized cost."""
    key = (round(lob, ndigits), round(zob, ndigits), id(stack))
    if key not in _CACHE:
        I1, I2 = assemble_fast(stack, lob, zob)
        P1 = frozen_P1_fast(stack, lob, zob)
        _CACHE[key] = (P1, I1, I2)
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()
