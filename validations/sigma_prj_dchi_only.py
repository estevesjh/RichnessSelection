"""Quantify the Delta-chi-only approximation for |r(z) - r(z_ob)|.

Exact     |Dr|^2 = chi(z)^2 + chi(z_ob)^2 - 2 chi(z) chi(z_ob) cos(theta)
Approx    |Dr|^2 ~ (chi(z) - chi(z_ob))^2

The approximation drops the 2 chi chi_o (1 - cos theta) ~ chi^2 theta^2
term, which formally gives an O(theta^4/12) correction after the
small-angle expansion. This script reports the relative error on the
z-integrated xi_NL weighted by dV(z) w_z(z, z_ob), at a handful of
theta values chosen as multiples of theta_lambda, across the 12
DES Y3 bin centres.

Output
------
validations/cache/dchi_only_error.csv
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np
from scipy.optimize import bisect

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _common import build_stack, CACHE_DIR  # noqa: E402

from richness_selection import Y3_LAM_MEAN, Y3_Z_MEAN
from richness_selection.geometry import theta_lambda, R_lambda
from richness_selection.photoz import (
    w_z, sigma_z, zmin4zkernel, zmax4zkernel,
)
from richness_selection.config import DEFAULT_GRID


OUT_CSV = os.path.join(CACHE_DIR, "dchi_only_error.csv")
THETA_FRACS = (0.1, 0.5, 1.0, 2.0)


def z_grid(sb, cosmo, lob, zob, Nz=None):
    """Reuse SelBias._z_grid for fidelity with the SigmaPrj integration."""
    if Nz is None:
        Nz = DEFAULT_GRID.Nz
    chi_o = float(cosmo.chi(zob))
    R_excl = R_lambda(lob) * (1.0 + zob)
    try:
        z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
        z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    except ValueError:
        sig = float(sigma_z(zob))
        z_fg_lo, z_bg_hi = max(0.01, zob - sig), zob + sig
    zs_ref = np.linspace(0.0, 2.0, 2000)
    chi_ref = cosmo.chi(zs_ref)
    dchi_dz_ref = np.gradient(chi_ref, zs_ref)
    zs, wzs = sb._z_grid(lob, zob, Nz, chi_o, R_excl,
                         z_fg_lo, z_bg_hi, zs_ref, chi_ref, dchi_dz_ref)
    return zs, wzs, chi_o


def integrals_for_theta(theta, cosmo, xi, zob, zs, wzs, chi_o):
    """Return (I_exact, I_dchi) with weights dV(z) * w_z(z, zob)."""
    chi_z = cosmo.chi(zs)
    dV = cosmo.dV_dzdOm(zs)
    wz = w_z(zs, zob)
    weight = wzs * dV * wz
    # Exact 3-D separation
    cos_th = np.cos(theta)
    r_exact = np.sqrt(np.maximum(
        chi_z ** 2 + chi_o ** 2 - 2.0 * chi_z * chi_o * cos_th, 0.0))
    # Delta-chi-only approximation
    r_dchi = np.abs(chi_z - chi_o)
    r_dchi = np.maximum(r_dchi, 1e-6)
    xi_exact = xi(r_exact, zob)
    xi_dchi = xi(r_dchi, zob)
    I_exact = float(np.sum(weight * xi_exact))
    I_dchi = float(np.sum(weight * xi_dchi))
    return I_exact, I_dchi


def main():
    stack = build_stack()
    cosmo = stack["cosmo"]; sb = stack["sb"]; xi = stack["xi"]

    rows = []
    for lam_bar in Y3_LAM_MEAN:
        for z_bar in Y3_Z_MEAN:
            th_lam = float(theta_lambda(lam_bar, z_bar, cosmo))
            zs, wzs, chi_o = z_grid(sb, cosmo, lam_bar, z_bar)
            for frac in THETA_FRACS:
                theta = frac * th_lam
                I_exact, I_dchi = integrals_for_theta(
                    theta, cosmo, xi, z_bar, zs, wzs, chi_o)
                rel_err = (I_dchi - I_exact) / I_exact if I_exact != 0 else np.nan
                rows.append({
                    "lam_bar": float(lam_bar),
                    "z_bar": float(z_bar),
                    "theta_lambda_rad": th_lam,
                    "theta_frac_of_lam": frac,
                    "theta_rad": theta,
                    "I_exact": I_exact,
                    "I_dchi_only": I_dchi,
                    "rel_error": rel_err,
                })

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[dchi_only] wrote {OUT_CSV}  ({len(rows)} rows)")

    # Summary: max |rel_error| across all 12 bins, per theta/theta_lam.
    print(f"{'theta/theta_lam':>15s}  "
          f"{'max |rel_err|':>14s}  {'median |rel_err|':>18s}")
    for frac in THETA_FRACS:
        errs = np.array([abs(r["rel_error"]) for r in rows
                         if r["theta_frac_of_lam"] == frac])
        print(f"{frac:>15.2f}  {errs.max():>14.3e}  "
              f"{np.median(errs):>18.3e}")


if __name__ == "__main__":
    main()
