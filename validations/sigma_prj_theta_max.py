"""Sensitivity of <Sigma^prj(R)> to the R_max_cMpch configuration.

For each of the 12 DES Y3 bin centres and R in {0.3, 1, 3, 10}:
evaluate the refactored SigmaPrj with R_max_cMpch in
{15, 30, 60, 120} and report the absolute / relative change of
each piece (total, rnd, cl).

Expectation:
  - cl:    near-converged at R_max = 30 (xi_NL falls off physically).
  - rnd:   grows unboundedly with R_max (integrand ~ constant in the
           NFW-extrapolation regime).  This is why the DEFAULT return
           of SigmaPrj is now the cl piece alone.
  - total: inherits the rnd growth; dominated by the choice of R_max.

Output
------
validations/cache/theta_max_compare.csv
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

from richness_selection import Y3_LAM_MEAN, Y3_Z_MEAN, SigmaPrj


OUT_CSV = os.path.join(CACHE_DIR, "theta_max_compare.csv")
R_VALS = np.array([0.3, 1.0, 3.0, 10.0])
R_MAX_GRID = (15.0, 30.0, 60.0, 120.0)    # cMpc/h


def main():
    stack = build_stack()
    cosmo = stack["cosmo"]; sb = stack["sb"]; nfw = stack["nfw"]

    # Reference instance: default R_max = 30 cMpc/h
    sp_ref = SigmaPrj(cosmo, sb, nfw, R_max_cMpch=30.0)

    rows = []
    for lam_bar in Y3_LAM_MEAN:
        for z_bar in Y3_Z_MEAN:
            dec_ref = sp_ref(R_VALS, lam_bar, z_bar,
                              return_decomposition=True)
            for iR, R in enumerate(R_VALS):
                row = {
                    "lam_bar": float(lam_bar),
                    "z_bar": float(z_bar),
                    "R_cMpch": float(R),
                    "total_30Mpc": float(dec_ref["total"][iR]),
                    "rnd_30Mpc":   float(dec_ref["rnd"][iR]),
                    "cl_30Mpc":    float(dec_ref["cl"][iR]),
                    "cl_over_total_30Mpc":
                        float(dec_ref["cl"][iR] / dec_ref["total"][iR]),
                }
                rows.append(row)
            # add sweeps
            for R_max in R_MAX_GRID:
                sp_s = SigmaPrj(cosmo, sb, nfw, R_max_cMpch=R_max)
                dec = sp_s(R_VALS, lam_bar, z_bar,
                            return_decomposition=True)
                for iR, R in enumerate(R_VALS):
                    row = rows[-R_VALS.size + iR]
                    row[f"total_Rmax{int(R_max)}"] = float(dec["total"][iR])
                    row[f"rnd_Rmax{int(R_max)}"] = float(dec["rnd"][iR])
                    row[f"cl_Rmax{int(R_max)}"] = float(dec["cl"][iR])

    with open(OUT_CSV, "w", newline="") as fh:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[theta_max] wrote {OUT_CSV}  ({len(rows)} rows)")

    print()
    print(f"{'lam':>6s} {'z':>5s} {'R':>5s}  "
          f"{'rnd(15)/rnd(30)':>15s} {'rnd(60)/rnd(30)':>15s} "
          f"{'rnd(120)/rnd(30)':>16s}  {'cl(60)/cl(30)':>14s}")
    for r in rows:
        ref_rnd = r["rnd_Rmax30"]
        ref_cl = r["cl_Rmax30"]
        rnd15 = r["rnd_Rmax15"] / ref_rnd
        rnd60 = r["rnd_Rmax60"] / ref_rnd
        rnd120 = r["rnd_Rmax120"] / ref_rnd
        cl60 = r["cl_Rmax60"] / ref_cl
        print(f"{r['lam_bar']:>6.1f} {r['z_bar']:>5.3f} "
              f"{r['R_cMpch']:>5.2f}  "
              f"{rnd15:>15.3f} {rnd60:>15.3f} "
              f"{rnd120:>16.3f}  {cl60:>14.4f}")


if __name__ == "__main__":
    main()
