"""Parabolic photo-z kernel w_z(z | z_ob).

    w_z(z) = max(1 - ((z - z_ob)/sigma_z(z_ob))^2, 0)

Toy placeholder: sigma_z(z) is linear in z.  For production pipelines,
replace with the z_kernel table / polynomial used by lc_lt_projection_y3.
"""
from __future__ import annotations
import numpy as np


def sigma_z_of_z(z):
    z = np.asarray(z, dtype=float)
    return 0.01 + 0.003 * (z - 0.4)


def w_z(z, zob):
    z = np.asarray(z, dtype=float)
    sig = sigma_z_of_z(zob)
    u = (z - zob) / sig
    return np.where(np.abs(u) < 1.0, 1.0 - u * u, 0.0)
