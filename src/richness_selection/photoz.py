"""Photo-z kernel from the DES Y3 z-kernel table.

Loads Matteo's `z_kernel_5perc_ext_z01.txt` (95%-percentile lambda-bin
probability width) and defines:

    sigma_z(z) = 1 / 100 / sqrt(sig_z_red(z))
    w_z(z, zob) = max(1 - ((z - zob) / sigma_z(zob))^2, 0)

The table path is resolved either via `RICHNESS_SELECTION_DATA_DIR`
env var or the package's `data/` directory alongside the install root.

Also exposes `zmin4zkernel` / `zmax4zkernel` used by Matteo's bisect
loops to find the redshift range over which w_z is non-zero.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DEFAULT = os.path.abspath(os.path.join(_HERE, "..", "..", "data"))


def _data_dir():
    return os.environ.get("RICHNESS_SELECTION_DATA_DIR", _DATA_DEFAULT)


_Z_KERNEL_FILE = "z_kernel_5perc_ext_z01.txt"


_cache = {}


def _kernel_spline():
    if "spl" in _cache:
        return _cache["spl"]
    path = os.path.join(_data_dir(), _Z_KERNEL_FILE)
    z_red, sig_z_red = np.loadtxt(path, unpack=True)
    spl = ius(z_red, 1.0 / 100.0 / np.sqrt(sig_z_red), k=1, ext=3)
    _cache["spl"] = spl
    _cache["z_min"] = float(z_red[0])
    _cache["z_max"] = float(z_red[-1])
    return spl


def sigma_z(z):
    """Photo-z kernel half-width (z-space) from Matteo's table."""
    spl = _kernel_spline()
    return spl(np.asarray(z, dtype=float))


# Back-compat alias (toy API name kept by old callers)
sigma_z_of_z = sigma_z


def w_z(z, zob):
    """Parabolic kernel w_z(z, zob) with support |z - zob| < sigma_z(z)."""
    z = np.asarray(z, dtype=float)
    u = (z - zob) / sigma_z(z)
    return np.where(np.abs(u) < 1.0, 1.0 - u * u, 0.0)


def zmin4zkernel(zmin, zcl):
    """For bisect: zmin + sigma_z(zmin) - zcl."""
    return zmin + sigma_z(zmin) - zcl


def zmax4zkernel(zmax, zcl):
    """For bisect: zmax - sigma_z(zmax) - zcl."""
    return zmax - sigma_z(zmax) - zcl
