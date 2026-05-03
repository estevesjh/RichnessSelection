"""Expected number counts N_ij per (richness, redshift) bin.

Eq. (37) of ``docs/richness_selection_function.tex``:

    <N_ij> = int dM int dz  Omega(z) * (dV / dz dOmega) * n(M, z)
                             * S_i(M, z) * K_j(z),

where ``S_i`` is the richness selection function and ``K_j`` the
Gaussian-CDF redshift kernel.  The outer (ln M, z) integral is a
fast 2-D Gauss-Legendre rule.
"""
from __future__ import annotations
import numpy as np

from ..gl import gl_nodes
from .kernels import K_i, K_j


def N_ij(bin_lam, bin_z, cosmo, hmf, mor, sigma_z,
         ln_M_min=np.log(1.0e13), ln_M_max=np.log(10.0 ** 15.5),
         N_M=32, N_z=24, N_q=32, L=6.0,
         solid_angle=None):
    """Expected number counts in the (i, j) bin (Eq. 37).

    Parameters
    ----------
    bin_lam : (float, float)
        Observed-richness bin edges (lambda_min, lambda_max).
    bin_z : (float, float)
        Observed-redshift bin edges (z_min, z_max).
    cosmo : Cosmology
        Provides ``dV_dzdOm(z)``.
    hmf : HMF
        ``hmf(M, z)`` returns n(M, z) in h^-1 Msun^-1 h^3 Mpc^-3.
    mor : MOR-like
        Mass-observable relation.
    sigma_z : float
        Photo-z scatter for the i-th richness bin.
    ln_M_min, ln_M_max : float
        Integration limits in ln M (default: 1e13 to 10^15.5 h^-1 Msun).
    N_M, N_z : int
        Gauss-Legendre node counts for the outer (ln M, z) integration.
    N_q : int
        Gauss-Legendre nodes for the inner ltr integral inside S_i.
    L : float
        Half-width of the ltr bracket in units of sigma_eff.
    solid_angle : float or callable or None
        Effective survey solid angle Omega(z) in steradians.  Accepts
        a scalar (redshift-independent), a callable taking z, or None
        (for a per-steradian number, Omega = 1).

        Physics note: at fixed richness cut, N_ij typically *rises*
        with z because dV/dzdOmega grows ~3x between z=0.2 and 0.6
        while the mass-integral int dM n(M,z) S_i(M,z) drops only
        ~30%.  The reason real redMaPPer N(z) falls at high z is
        that the catalogue's effective Omega(z) drops sharply as the
        red sequence loses photometric contrast -- pass a callable
        solid_angle to model that.

    Returns
    -------
    N : float
        Expected number of clusters in the (i, j) bin.
    """
    lam_min, lam_max = bin_lam
    zmin, zmax = bin_z

    # --- outer (lnM, z) grids, log in M ------------------------------------
    lnMs, wM = gl_nodes(ln_M_min, ln_M_max, N_M)
    Ms = np.exp(lnMs)                                   # (NM,)

    # Pad the z-grid by L photo-z sigmas on each side so K_j's Gaussian CDF
    # difference has fully truncated by the endpoints.
    z_pad = L * float(sigma_z)
    z_lo = max(1e-3, zmin - z_pad)
    z_hi = zmax + z_pad
    zs, wz = gl_nodes(z_lo, z_hi, N_z)                  # (Nz,)

    # --- solid angle, dV/dz, K_j vectors ----------------------------------
    if solid_angle is None:
        Omega_vec = np.ones_like(zs)
    elif callable(solid_angle):
        Omega_vec = np.asarray([float(solid_angle(z)) for z in zs])
    else:
        Omega_vec = float(solid_angle) * np.ones_like(zs)

    dVdz_vec = cosmo.dV_dzdOm(zs)                       # (Nz,)
    Kj_vec   = K_j(zs, zmin, zmax, sigma_z)             # (Nz,)

    # --- inner ltr grid (shared across all M, per z) ----------------------
    # Bracket [a, b] from the global (max over M) MOR moments so a single
    # GL grid covers the mass range.  ltr_mean / ltr_sigma are vectorised
    # over M.
    z_mid = zs[zs.size // 2]
    ltr_mean_M = mor.ltr_mean(Ms, z_mid)                # use midpoint z for bracket
    ltr_sig_M  = mor.ltr_sigma(Ms, z_mid)
    # N_q=32 GL on [a, b] captures the log-normal / Poisson cleanly; here we
    # widen the bracket slightly over what S_i uses so one grid fits all M.
    a_global = 0.0
    b_global = float(np.max(ltr_mean_M + L * ltr_sig_M))
    ltr_nodes, w_ltr = gl_nodes(a_global, b_global, N_q)   # (Nq,)

    # --- per-z accumulation ------------------------------------------------
    # K_i depends only on (ltr, z), not M.  Precompute once per z.
    # P(ltr | M, z) is an (Nq, NM) tensor per z; contract to (NM,) once.
    total = 0.0
    M_weight = wM * Ms                                   # (NM,) -- d lnM Jacobian
    for iz, z in enumerate(zs):
        Ki_z = K_i(ltr_nodes, float(z), lam_min, lam_max)          # (Nq,)
        P_lmz = mor.pdf(ltr_nodes[:, None], Ms[None, :], float(z)) # (Nq, NM)
        # S_i(M, z) = sum_k w_ltr[k] K_i(ltr_k, z) P(ltr_k | M, z)
        Si_m = np.einsum('q,q,qM->M', w_ltr, Ki_z, P_lmz)          # (NM,)
        n_mz = hmf(Ms, float(z))                                   # (NM,)
        inner = float(np.sum(M_weight * n_mz * Si_m))
        total += wz[iz] * Omega_vec[iz] * dVdz_vec[iz] * Kj_vec[iz] * inner
    return float(total)
