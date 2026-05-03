"""Richness-binned selection function S_i(M, z) for number counts.

Implements Eqs. (13)-(37) of
``docs/richness_selection_function.tex``.

Public entries:

- ``K_i(ltr, z, lam_min, lam_max)`` -- closed-form bin-integrated
  observed-richness kernel (Eq. 16 + 17).
- ``K_j(ztr, zob_min, zob_max, sigma_z)`` -- Gaussian-CDF
  observed-redshift kernel (Eq. 12).
- ``S_i(M, z, lam_min, lam_max, mor, N_q=32)`` -- richness selection
  function per mass bin (Eq. 22).
- ``S_threshold(M, z, lam_min, mor, N_q=32)`` -- above-threshold
  selection S(M, z; lambda^ob > lam_min), i.e. the CDF complement
  at a single cut (same as ``S_i`` with lam_max = +inf).
- ``S_ij(M, z, bin_lam, bin_z, mor, ...)`` -- S_i * K_j per (i, j) bin.
- ``N_ij(bin_lam, bin_z, cosmo, hmf, mor, ...)`` -- expected cluster
  number counts (Eq. 37).
"""
from .kernels import K_i, K_j, F_EMG
from .selection import S_i, S_ij, S_threshold
from .number_counts import N_ij
from .survey import (
    omega_z_sdss, omega_z_sdss_table, omega_z_des, omega_z_const,
    SDSS_AREA_DEG2, DES_AREA_DEG2, DEG2_TO_STER,
)

__all__ = [
    "K_i", "K_j", "F_EMG",
    "S_i", "S_ij", "S_threshold",
    "N_ij",
    "omega_z_sdss", "omega_z_sdss_table", "omega_z_des", "omega_z_const",
    "SDSS_AREA_DEG2", "DES_AREA_DEG2", "DEG2_TO_STER",
]
