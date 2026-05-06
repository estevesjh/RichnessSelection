"""Study the symmetry of xi(|Delta chi|) around z_ob and the
asymmetry of the remaining kernel factors.

The integrand of the z-axis in <Sigma^prj> is
    W(z; M_bar, z_ob) = dV/dzdOmega(z) * w_z(z, z_ob)
                        * n_bar(M_bar, z) * xi_NL(|Delta chi|, z_ob)
where xi_NL depends on |Delta chi| = |chi(z) - chi(z_ob)| through the
Delta-chi-only approximation, which is exactly symmetric around z_ob.
The dV, n_bar and (weakly) w_z factors are not.

This script:
  1. Tabulates W(z) on a dense z-grid per DES Y3 bin centre.
  2. Reports the local asymmetry A(Delta z) for a handful of offsets.
  3. Compares the full integral to a symmetrised "one-sided times 2"
     version that drops the asymmetry.

Outputs
-------
validations/cache/z_symmetry.csv     (per-bin scalars)
validations/cache/xi_weights_z.npz   (dense curves for the audit notebook)
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _common import build_stack, CACHE_DIR  # noqa: E402

from richness_selection import Y3_LAM_MEAN, Y3_Z_MEAN
from richness_selection.geometry import R_lambda
from richness_selection.photoz import w_z, sigma_z


OUT_CSV = os.path.join(CACHE_DIR, "z_symmetry.csv")
OUT_NPZ = os.path.join(CACHE_DIR, "xi_weights_z.npz")

M_BAR = 3.0e14          # representative mass for n_bar(M, z)
DELTA_Z_GRID = 0.08     # half-width of the dense z-grid around z_ob
NZ = 401                # dense z-grid (odd so z_ob is a node)
OFFSETS = (0.005, 0.01, 0.02, 0.04)   # Delta z values for the A-metric


def integrand_pieces(cosmo, hmf, xi, lam_bar, z_bar, zs):
    """Return a dict of the z-dependent factors that build W(z)."""
    chi_o = float(cosmo.chi(z_bar))
    chi_z = cosmo.chi(zs)
    dchi = np.abs(chi_z - chi_o)
    dchi_eff = np.maximum(dchi, 1e-6)
    dV = cosmo.dV_dzdOm(zs)
    wz = w_z(zs, z_bar)
    n_bar = hmf(M_BAR, zs)
    xi_vals = xi(dchi_eff, z_bar)
    # Mask out LoS exclusion per P1/I1/I2 convention.
    R_excl = R_lambda(lam_bar) * (1.0 + z_bar)
    excl_mask = dchi < R_excl
    xi_vals_excl = xi_vals.copy()
    xi_vals_excl[excl_mask] = 0.0
    W = dV * wz * n_bar * xi_vals_excl
    return dict(chi_o=chi_o, chi_z=chi_z, dchi=dchi, dV=dV, wz=wz,
                n_bar=n_bar, xi=xi_vals, xi_excl=xi_vals_excl,
                W=W, R_excl=R_excl)


def asymmetry_scalars(zs, W, z_ob, offsets):
    """A(dz) = [W(z_ob+dz) - W(z_ob-dz)] / [W(z_ob+dz) + W(z_ob-dz)]."""
    out = {}
    for dz in offsets:
        w_plus = float(np.interp(z_ob + dz, zs, W))
        w_minus = float(np.interp(z_ob - dz, zs, W))
        tot = w_plus + w_minus
        out[dz] = (w_plus - w_minus) / tot if tot != 0 else np.nan
    return out


def main():
    stack = build_stack()
    cosmo = stack["cosmo"]; hmf = stack["hmf"]; xi = stack["xi"]

    per_bin = {}   # key: (lam_bar, z_bar) -> dict of arrays for NPZ
    rows = []

    for lam_bar in Y3_LAM_MEAN:
        for z_bar in Y3_Z_MEAN:
            zs = np.linspace(z_bar - DELTA_Z_GRID, z_bar + DELTA_Z_GRID, NZ)
            pieces = integrand_pieces(cosmo, hmf, xi, lam_bar, z_bar, zs)

            W_full = pieces["W"]

            # Symmetrised-around-zob approximation: average the two sides
            # of the z-grid and treat as if |Delta chi| is enough to fix
            # the shape. Equivalent to replacing the asymmetric kernel
            # factors with their zob value; we do this exactly here.
            # "Symmetrised" = dV(zob)*wz(zob)*n_bar(zob)*xi_symm(dchi).
            dV_o = float(cosmo.dV_dzdOm(z_bar))
            nbar_o = float(hmf(M_BAR, z_bar))
            wz_o = float(w_z(np.array([z_bar]), z_bar)[0])  # = 1.0 by definition
            W_sym = dV_o * nbar_o * wz_o * pieces["xi_excl"]

            # One-sided-times-two proxy on the asymmetric W: 2 * int_{z>=zob} W dz
            mask_pos = zs >= z_bar
            I_full = float(np.trapz(W_full, zs))
            I_onesided = 2.0 * float(np.trapz(W_full[mask_pos], zs[mask_pos]))
            I_sym = float(np.trapz(W_sym, zs))

            A = asymmetry_scalars(zs, W_full, z_bar, OFFSETS)
            rel_err_onesided = (I_onesided - I_full) / I_full if I_full != 0 else np.nan
            rel_err_symm = (I_sym - I_full) / I_full if I_full != 0 else np.nan

            rows.append({
                "lam_bar": float(lam_bar),
                "z_bar": float(z_bar),
                "sigma_z": float(sigma_z(z_bar)),
                "R_excl_cMpch": pieces["R_excl"],
                "I_full": I_full,
                "I_onesided_x2": I_onesided,
                "rel_err_onesided": rel_err_onesided,
                "I_symmetrised": I_sym,
                "rel_err_symmetrised": rel_err_symm,
                **{f"A_dz_{dz:.3f}": A[dz] for dz in OFFSETS},
            })
            per_bin[f"lam_{lam_bar:g}_z_{z_bar:g}"] = dict(
                zs=zs,
                W=W_full,
                W_sym=W_sym,
                dV=pieces["dV"],
                wz=pieces["wz"],
                n_bar=pieces["n_bar"],
                xi=pieces["xi"],
                xi_excl=pieces["xi_excl"],
                dchi=pieces["dchi"],
                R_excl=np.array(pieces["R_excl"]),
                z_ob=np.array(z_bar),
                lam_bar=np.array(lam_bar),
            )

    with open(OUT_CSV, "w", newline="") as fh:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[z_symmetry] wrote {OUT_CSV}  ({len(rows)} bins)")

    flat = {}
    for key, data in per_bin.items():
        for field, arr in data.items():
            flat[f"{key}__{field}"] = arr
    np.savez_compressed(OUT_NPZ, **flat)
    print(f"[z_symmetry] wrote {OUT_NPZ}  ({len(flat)} arrays)")

    print()
    print(f"{'lam':>6s} {'z':>5s}  {'A(0.005)':>9s} {'A(0.02)':>9s} "
          f"{'A(0.04)':>9s}  {'err_onesided':>13s}  {'err_symm':>10s}")
    for r in rows:
        print(f"{r['lam_bar']:>6.1f} {r['z_bar']:>5.3f}  "
              f"{r['A_dz_0.005']:>+9.3e} {r['A_dz_0.020']:>+9.3e} "
              f"{r['A_dz_0.040']:>+9.3e}  "
              f"{r['rel_err_onesided']:>+13.3e}  "
              f"{r['rel_err_symmetrised']:>+10.3e}")


if __name__ == "__main__":
    main()
