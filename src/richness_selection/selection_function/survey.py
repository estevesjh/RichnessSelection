"""Survey-footprint models Omega(z) in steradians.

The effective solid angle an optical cluster catalogue covers drops
at high z because the red-sequence photometric contrast degrades.
For a realistic Omega(z), we port two polynomial fits from the C++
pipeline (``y3_cluster_cpp/src/models/omega_z_{sdss,des}.hh``):

- ``omega_z_sdss(z)``: three-piece polynomial fit calibrated on
  the SDSS redMaPPer-v5.10 volume-limited catalogue (Costanzi 2019b
  / Costanzi 2021).  Peaks around 3.13 sr (~ 10,260 deg^2) between
  z = 0.1 and z = 0.4, drops slightly by z = 0.6.
- ``omega_z_des(z)``: piecewise polynomial fit for the DES Y1
  volume-limited catalogue.  Plateau of ~ 0.45 sr (~ 1480 deg^2)
  below z ~ 0.5, sharp cutoff above z ~ 0.7.

Both return the effective solid angle in steradians and take a
single z (scalar or ndarray).  Drop-in callables for the
``solid_angle=`` argument of ``N_ij``.

Also exposed:

- ``omega_z_sdss_table(z)``: interpolant over the
  ``data/omega_z_sdss.txt`` lookup table (549 nodes over
  z in [0.05, 0.60]); same catalogue as ``omega_z_sdss`` but
  uses the tabulated Omega(z) directly instead of the polynomial
  fit.  Prefer this for precision work.
- ``omega_z_const(z, A_ster)``: a constant-Omega sanity baseline.
"""
from __future__ import annotations
import os
import numpy as np


DEG2_TO_STER = (np.pi / 180.0) ** 2

# Catalogue-area headline values (for docs / plots only).
SDSS_AREA_DEG2 = 10260.0           # ~ peak of omega_z_sdss
DES_AREA_DEG2  = 1480.0            # ~ plateau of omega_z_des


# ------------------------------------------------------------------
# SDSS polynomial fit (Costanzi 2019 / 2021)
# ------------------------------------------------------------------
# Coefficients mirror y3_cluster_cpp/src/models/omega_z_sdss.hh
# polynomial<12> SDSS_fit{{c11, c10, ..., c1, c0}} applied to (zt - 0.2).
# The y3_cluster polynomial convention stores c_{n-1} ... c_0 so we
# reverse once to use np.polyval-friendly order (highest degree first).
_SDSS_COEFFS = np.array([
    -1.14293122e+05,  5.96846869e+04,  9.24239180e+03, -2.23118813e+03,
    -4.52580713e+03,  1.18404878e+03,  1.27951911e+02, -5.05716847e+01,
     1.01744577e+00, -3.11253383e-01,  5.48481084e-03,  3.12629987e+00,
])  # highest degree first (same order as the C++ array).


def omega_z_sdss(z):
    """SDSS polynomial fit, Omega(z) in steradians.

    Mirrors ``OMEGA_Z_SDSS::operator()`` from y3_cluster_cpp.
    Applied as ``polynomial(z - 0.2)``.  Valid approximately for
    z in [0.05, 0.60]; extrapolates outside.
    """
    z = np.asarray(z, dtype=float)
    return np.polyval(_SDSS_COEFFS, z - 0.2)


# ------------------------------------------------------------------
# DES polynomial fit (three-piece, from omega_z_des.hh)
# ------------------------------------------------------------------
_DES_FIT1 = np.array([0.0, 0.0, 0.0,
                      -0.00262353, 0.01940118, 0.45133063])   # z < 0.504
_DES_FIT2 = np.array([1.33647377e+4, 1.35291046e+3,
                      -1.26204891e+2, -2.83454918e+1,
                      -2.26465905, 3.84958753e-1])             # 0.504 <= z < 0.7, input (z - 0.6)
_DES_FIT3 = np.array([0.0, 0.0,
                      -1.88101967, 4.8071839,
                      -4.11424324, 1.18196785])                # z >= 0.7, input z


def omega_z_des(z):
    """DES Y1 piecewise polynomial fit, Omega(z) in steradians.

    Mirrors ``OMEGA_Z_DES::operator()`` from y3_cluster_cpp.
    Three-piece: z<0.504 rising plateau, 0.504<=z<0.7 smooth transition
    (input shifted by 0.6), z>=0.7 cubic rolloff to ~0.
    """
    z = np.asarray(z, dtype=float)
    v1 = np.polyval(_DES_FIT1, z)
    v2 = np.polyval(_DES_FIT2, z - 0.6)
    v3 = np.polyval(_DES_FIT3, z)
    return np.where(z < 0.504, v1,
                    np.where(z < 0.7, v2, v3))


# ------------------------------------------------------------------
# SDSS tabulated Omega(z) (lookup, 549 nodes)
# ------------------------------------------------------------------
_SDSS_TABLE = None


def _load_sdss_table():
    global _SDSS_TABLE
    if _SDSS_TABLE is not None:
        return _SDSS_TABLE
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.environ.get(
        "RICHNESS_SELECTION_OMEGA_Z_SDSS",
        os.path.abspath(os.path.join(here, "..", "..", "..", "data", "omega_z_sdss.txt")),
    )
    data = np.loadtxt(path)
    _SDSS_TABLE = (data[:, 0], data[:, 1])   # (z, Omega)
    return _SDSS_TABLE


def omega_z_sdss_table(z):
    """Tabulated SDSS Omega(z) (lookup over 549 nodes in [0.05, 0.60])."""
    z_grid, O_grid = _load_sdss_table()
    return np.interp(np.asarray(z, dtype=float), z_grid, O_grid,
                     left=O_grid[0], right=O_grid[-1])


# ------------------------------------------------------------------
# Constant-Omega baseline
# ------------------------------------------------------------------

def omega_z_const(z, A_ster):
    """Constant Omega (no z-dependence). ``A_ster`` is in steradians."""
    z = np.asarray(z, dtype=float)
    return np.full_like(z, float(A_ster))
