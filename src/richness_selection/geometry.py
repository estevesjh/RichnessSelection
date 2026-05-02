"""Projection-disk geometry (Costanzi 2026).

Ported from Matteo's cell 14 + `area_overlap` (end of cell 16) of
Costanzi2026_SelectionBias.ipynb, so the projection kernel f_area,
solid-angle normalisation Omega_halos, and the richness-radius
conversion R_lambda all match his numerics.

    R_lambda(lam) = (lam / 100)^0.2                [cMpc/h]
    theta_lambda(lam, z) = R_lambda(lam) * (1+z) / chi(z)   [rad]
    f_area(lob, zob, ltr, z) = (1 + theta_ltr / theta_lob)^(-2)
    Omega_halos(lob, zob, ltr, z) = 2 pi (1 - cos(theta_ltr + theta_lob))
    area_overlap(theta, theta_lob, theta_ltr) = |overlap|/(pi theta_ltr^2)

Note: f_area is Matteo's closed form, *not* the two-disk-overlap.  The
closed form is slightly different physics (cap-based, not disk-disk),
and is what Matteo's Sigma_prj pipeline actually uses.
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology


def R_lambda(lam):
    """Cluster radius proxy from richness [cMpc/h]."""
    lam = np.asarray(lam, dtype=float)
    return (lam / 100.0) ** 0.2


def theta_lambda(lam, z, cosmo: Cosmology):
    """Angular size of the lambda-disk at redshift z [rad]."""
    return R_lambda(lam) * (1.0 + z) / cosmo.chi(z)


def f_area(lob, zob, ltr, z, cosmo: Cosmology):
    """Matteo cell-14 closed form: (1 + theta_ltr / theta_lob)^(-2)."""
    theta_ltr = theta_lambda(ltr, z, cosmo)
    theta_lob = theta_lambda(lob, zob, cosmo)
    return (1.0 + theta_ltr / theta_lob) ** (-2.0)


def Omega_halos(lob, zob, ltr, z, cosmo: Cosmology):
    """Solid angle 2 pi (1 - cos(theta_ltr + theta_lob))."""
    theta_ltr = theta_lambda(ltr, z, cosmo)
    theta_lob = theta_lambda(lob, zob, cosmo)
    return 2.0 * np.pi * (1.0 - np.cos(theta_ltr + theta_lob))


def area_overlap(theta, theta_lob, theta_ltr):
    """Fractional overlap area between two circular apertures (Matteo's form).

    Returns A_overlap of shape broadcast(theta, theta_ltr) / (pi theta_ltr^2).

    Parameters
    ----------
    theta : array
        Angular separation between disk centres [rad].  Shape (..., N_ltr) or
        (...,).  If a 1-D array of length N_theta is passed together with a
        1-D theta_ltr of length N_ltr, caller is expected to broadcast to
        (N_theta, N_ltr) before the call.
    theta_lob : float
        Angular radius of the target (primary) disk.
    theta_ltr : array
        Angular radius of the projector (secondary) disk, length N_ltr.
    """
    theta = np.asarray(theta, dtype=float)
    theta_ltr = np.atleast_1d(np.asarray(theta_ltr, dtype=float))
    # If the caller already shaped theta to match theta_ltr along its last axis
    # (Matteo's convention), broadcast directly; otherwise add a new axis so
    # theta has shape (*theta.shape, N_ltr).
    if theta.ndim >= 1 and theta.shape[-1] == theta_ltr.shape[-1]:
        theta_b = np.asarray(theta, dtype=float)
        ltr_b = np.broadcast_to(theta_ltr, theta_b.shape)
    else:
        theta_b, ltr_b = np.broadcast_arrays(theta[..., None], theta_ltr)
    theta_b = np.array(theta_b, dtype=float)
    ltr_b = np.array(ltr_b, dtype=float)
    A = np.ones_like(theta_b)

    # No overlap
    A[theta_b > theta_lob + ltr_b] = 0.0

    # Full containment: projector bigger than target, target sits inside it
    mask_full = ltr_b > theta_lob
    A[mask_full] = (theta_lob ** 2) / ltr_b[mask_full] ** 2

    # Partial overlap (lens formula)
    cond = theta_b > np.abs(theta_lob - ltr_b)
    if np.any(cond):
        tt = theta_b[cond]
        ll = ltr_b[cond]
        argcos1 = np.clip((tt ** 2 + ll ** 2 - theta_lob ** 2)
                          / (2.0 * tt * ll), -1.0, 1.0)
        argcos2 = np.clip((tt ** 2 + theta_lob ** 2 - ll ** 2)
                          / (2.0 * tt * theta_lob), -1.0, 1.0)
        argsqrt = ((-tt + ll + theta_lob)
                   * (tt + ll - theta_lob)
                   * (tt - ll + theta_lob)
                   * (tt + ll + theta_lob))
        argsqrt = np.clip(argsqrt, 0.0, None)
        A[cond] = (ll ** 2 * np.arccos(argcos1)
                   + theta_lob ** 2 * np.arccos(argcos2)
                   - 0.5 * np.sqrt(argsqrt)) / (np.pi * ll ** 2)
    return A


def sigma_theta(theta, lob, zob, cosmo: Cosmology,
                damping: float = 2.5,
                n_th_inf: float = 0.0, n_th_sup: float = 1.0):
    """Smooth sigmoid transition in theta (Matteo cell 16).

        x0 = (n_th_inf + n_th_sup) * theta_lob / 2
        k  = damping / ((n_th_sup - n_th_inf) * theta_lob)
        return 1 / (1 + exp(-k (theta - x0)))
    """
    theta_lob = theta_lambda(lob, zob, cosmo)
    x0 = 0.5 * (n_th_inf + n_th_sup) * theta_lob
    k = damping / ((n_th_sup - n_th_inf) * theta_lob)
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(theta, dtype=float) - x0)))
