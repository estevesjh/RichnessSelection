"""Tabulate R_lambda, theta_lambda, 2*theta_lambda across the 12 DES Y3 bins.

Physics context
---------------
The projection-bias model uses the aperture-overlap fraction f_A which
has support within theta < theta_lambda + theta_lob ~ 2*theta_lambda.
Once b_sel(theta) is folded in, the cl+LSS piece of <Sigma^prj> is
essentially zero beyond 2*theta_lambda. This script tabulates those
angular scales across the 12 (lambda, z) bin centres and contrasts
them with theta_R = R/D_A(z_ob) at R = 30 cMpc/h, which is where the
Sigma_mis peak of the RND piece moves as R grows.

Output
------
validations/cache/theta_lambda_12bins.csv
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from richness_selection import Cosmology, Y3_LAM_MEAN, Y3_Z_MEAN
from richness_selection.geometry import R_lambda, theta_lambda


RAD_TO_ARCMIN = (180.0 / np.pi) * 60.0
R_REFERENCE_CMPCH = 30.0
OUT_CSV = os.path.join(_HERE, "cache", "theta_lambda_12bins.csv")


def main():
    cosmo = Cosmology()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = []
    for lam_bar in Y3_LAM_MEAN:
        for z_bar in Y3_Z_MEAN:
            chi = float(cosmo.chi(z_bar))
            D_A = float(cosmo.D_A(z_bar))
            R_lam = float(R_lambda(lam_bar))
            th_lam = float(theta_lambda(lam_bar, z_bar, cosmo))
            two_th_lam = 2.0 * th_lam
            th_R30 = R_REFERENCE_CMPCH / D_A
            rows.append({
                "lam_bar": float(lam_bar),
                "z_bar": float(z_bar),
                "R_lambda_cMpch": R_lam,
                "chi_cMpch": chi,
                "D_A_cMpch": D_A,
                "theta_lambda_rad": th_lam,
                "theta_lambda_arcmin": th_lam * RAD_TO_ARCMIN,
                "two_theta_lambda_rad": two_th_lam,
                "two_theta_lambda_arcmin": two_th_lam * RAD_TO_ARCMIN,
                "theta_R_at_30Mpch_rad": th_R30,
                "ratio_thetaR30_over_2thetalam": th_R30 / two_th_lam,
            })

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[theta_lambda] wrote {OUT_CSV}  ({len(rows)} bins)")
    header = (f"{'lam':>6s} {'z':>5s}  {'R_lam':>7s}  {'D_A':>8s}  "
              f"{'theta_lam [arcmin]':>20s}  {'2*theta_lam [arcmin]':>22s}  "
              f"{'thetaR30/2theta_lam':>21s}")
    print(header)
    for r in rows:
        print(f"{r['lam_bar']:>6.1f} {r['z_bar']:>5.3f}  "
              f"{r['R_lambda_cMpch']:>7.3f}  {r['D_A_cMpch']:>8.1f}  "
              f"{r['theta_lambda_arcmin']:>20.3f}  "
              f"{r['two_theta_lambda_arcmin']:>22.3f}  "
              f"{r['ratio_thetaR30_over_2thetalam']:>21.1f}")


if __name__ == "__main__":
    main()
