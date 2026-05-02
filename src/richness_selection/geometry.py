"""Cylinder / disk geometry helpers for the projection kernel.

    R_lambda(lam) = (lam / 100)^0.2   [cMpc/h]    (richness-radius relation)
    theta_lambda(lam, z) = R_lambda(lam) * (1+z) / chi(z)    [rad]

Closed-form S1: integrating f_A(theta) = A(theta, r1, r2) / (pi r2^2) over
the projected disk yields pi r1^2 (the overlap fraction, area-weighted,
integrates to the target-disk area, independent of the projector radius).
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology


def R_lambda(lam):
    """Cluster radius proxy from richness [cMpc/h]."""
    lam = np.asarray(lam, dtype=float)
    return (lam / 100.0) ** 0.2


def theta_lambda(lam, z, cosmo: Cosmology):
    """Angular radius of the lambda-disk at redshift z [rad]."""
    return R_lambda(lam) * (1.0 + z) / cosmo.chi(z)


def two_disk_overlap(d, r1, r2):
    """Overlap area of two disks with radii r1, r2 at centre-to-centre separation d.

    Vectorised over d; r1, r2 scalar.
    """
    d = np.asarray(d, dtype=float)
    A = np.zeros_like(d)
    r_min = min(r1, r2)
    r_max = max(r1, r2)
    full = d + r_min <= r_max
    empty = d >= r1 + r2
    partial = ~(full | empty)
    A[full] = np.pi * r_min * r_min
    dp = d[partial]
    r1_2, r2_2 = r1 * r1, r2 * r2
    alpha = np.arccos((dp * dp + r1_2 - r2_2) / (2.0 * dp * r1))
    beta = np.arccos((dp * dp + r2_2 - r1_2) / (2.0 * dp * r2))
    A[partial] = (r1_2 * alpha + r2_2 * beta
                  - 0.5 * np.sqrt(
                      (-dp + r1 + r2) * (dp + r1 - r2)
                      * (dp - r1 + r2) * (dp + r1 + r2)))
    return A


def f_A(theta, lob, zob, lam, z, cosmo: Cosmology):
    """Overlap fraction A(theta) / (pi r2^2), r1 = target disk, r2 = projector."""
    r1 = theta_lambda(lob, zob, cosmo)
    r2 = theta_lambda(lam, z, cosmo)
    return two_disk_overlap(theta, r1, r2) / (np.pi * r2 * r2)


def sigma_theta(theta, lob, zob, cosmo: Cosmology):
    """Sigmoid ansatz with Costanzi 2026 fixed constants (k=2.5, theta0=0.5)."""
    th_lam = theta_lambda(lob, zob, cosmo)
    k = 2.5 / th_lam
    theta0 = 0.5 * th_lam
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(theta, dtype=float) - theta0)))
