"""Richness-binned mass selection function S_i(M, z).

Eq. (22) of the doc:

    S_i(M, z) ≈ sum_k (b - a) / 2 * w_k * K_i(lam_k, z) * P(lam_k | M, z),

with Gauss-Legendre nodes (t_k, w_k) on [-1, 1] mapped to
[a, b] set by the MOR moments (mu_eff, sigma_eff).
"""
from __future__ import annotations
import numpy as np

from ..gl import gl_nodes
from .kernels import K_i, K_j


def _ab_from_moments(mor, M, z, L=6.0):
    """Quadrature interval [a, b] from the MOR's linear-space moments."""
    mu_eff = float(mor.ltr_mean(M, z))
    sig_eff = float(mor.ltr_sigma(M, z))
    a = max(0.0, mu_eff - L * sig_eff)
    b = mu_eff + L * sig_eff
    return a, b


def S_i(M, z, lam_min, lam_max, mor, N_q=32, L=6.0):
    """Richness selection function S_i(M, z, [lam_min, lam_max]).

    Parameters
    ----------
    M : float or array
        Halo mass in h^-1 M_sun.  Broadcasts: ``z`` is a scalar (one
        redshift per call), ``M`` may be an array.
    z : float
        Halo true redshift.
    lam_min, lam_max : float
        Observed-richness bin edges.
    mor : MOR-like
        Object exposing ``pdf(ltr, M, z)``, ``ltr_mean(M, z)``, and
        ``ltr_sigma(M, z)``.  Either ``MOR`` (HOD) or ``LogNormalMOR``.
    N_q : int
        Gauss-Legendre node count for the ``ltr`` integral.
    L : float
        Half-width, in units of sigma_eff, of the quadrature interval.

    Returns
    -------
    S_i : float or array
        Probability that a halo of mass ``M`` at true redshift ``z``
        is observed with a richness in ``[lam_min, lam_max]``.
    """
    M_arr = np.atleast_1d(np.asarray(M, dtype=float))
    out = np.empty_like(M_arr)
    for i, Mi in enumerate(M_arr):
        a, b = _ab_from_moments(mor, float(Mi), float(z), L=L)
        if b <= a:
            out[i] = 0.0
            continue
        lam_nodes, w_nodes = gl_nodes(a, b, N_q)
        # K_i at each node
        Ki = K_i(lam_nodes, z, lam_min, lam_max)
        # P(ltr | M, z) at each node
        P = mor.pdf(lam_nodes, float(Mi), float(z))
        out[i] = float(np.sum(w_nodes * Ki * P))
    if np.ndim(M) == 0:
        return float(out[0])
    return out


def S_threshold(M, z, lam_min, mor, N_q=32, L=6.0, lam_max=1e4):
    """S(M, z; lambda^ob > lam_min) = int_{lam_min}^infty P(lob | M, z) dlob.

    Useful as the cumulative ``above-threshold'' selection for a single
    richness cut (as opposed to a bin).  Internally we call ``S_i`` with
    an upper edge ``lam_max`` that is effectively infinite for the
    Costanzi kernel.
    """
    return S_i(M, z, lam_min, lam_max, mor, N_q=N_q, L=L)


def S_ij(M, z, bin_lam, bin_z, sigma_z, mor, N_q=32, L=6.0):
    """Joint richness+redshift selection function S_ij(M, z).

    ``bin_lam`` and ``bin_z`` are ``(min, max)`` tuples for the i-th
    richness bin and j-th redshift bin, respectively. ``sigma_z`` is
    the photo-z scatter for that richness bin.
    """
    lam_min, lam_max = bin_lam
    zob_min, zob_max = bin_z
    Si = S_i(M, z, lam_min, lam_max, mor, N_q=N_q, L=L)
    Kj = K_j(z, zob_min, zob_max, sigma_z)
    return Si * Kj
