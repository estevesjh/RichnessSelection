"""Scan Sigma_prj(R) decomposition: rnd vs cl+LSS, across the 12 Y3 bins.

For each (lam_bar, z_bar) in DES Y3, evaluate the refactored SigmaPrj
on a dense R grid and save the three components (total, rnd, cl+LSS)
so the audit notebook can plot the R-dependence of each piece.

Output
------
validations/cache/sigma_prj_rnd_cl.npz
    R [NR]                        dense R grid in cMpc/h
    lam_bar [Nbin]                richness means
    z_bar [Nbin]                  redshift means
    total [Nbin, NR]              Sigma_prj(R)
    rnd   [Nbin, NR]              Sigma_prj_rnd(R)
    cl    [Nbin, NR]              Sigma_prj_cl+LSS(R)
"""
from __future__ import annotations
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _common import build_stack, CACHE_DIR  # noqa: E402

from richness_selection import Y3_LAM_MEAN, Y3_Z_MEAN, SigmaPrj


OUT_NPZ = os.path.join(CACHE_DIR, "sigma_prj_rnd_cl.npz")
R_DENSE = np.logspace(-1, 1.3, 30)   # 0.1 to ~20 cMpc/h


def main():
    stack = build_stack()
    cosmo = stack["cosmo"]; sb = stack["sb"]; nfw = stack["nfw"]
    sp = SigmaPrj(cosmo, sb, nfw, n_theta_per_seg=30)

    lam_bars = np.asarray(Y3_LAM_MEAN, dtype=float)
    z_bars = np.asarray(Y3_Z_MEAN, dtype=float)

    pairs = [(lam, z) for lam in lam_bars for z in z_bars]
    Nbin = len(pairs); NR = R_DENSE.size
    total = np.empty((Nbin, NR))
    rnd = np.empty((Nbin, NR))
    cl = np.empty((Nbin, NR))
    lam_col = np.empty(Nbin); z_col = np.empty(Nbin)

    for i, (lam, z) in enumerate(pairs):
        dec = sp(R_DENSE, lam, z, return_decomposition=True)
        total[i] = dec["total"]; rnd[i] = dec["rnd"]; cl[i] = dec["cl"]
        lam_col[i] = lam; z_col[i] = z
        print(f"  bin {i:2d}: lam={lam:5.1f}, z={z:.3f}  "
              f"Sigma[R=1]={dec['total'][np.argmin(np.abs(R_DENSE - 1.0))]:.3e}  "
              f"(rnd={dec['rnd'][np.argmin(np.abs(R_DENSE - 1.0))]:.3e}, "
              f"cl={dec['cl'][np.argmin(np.abs(R_DENSE - 1.0))]:.3e})",
              flush=True)

    np.savez_compressed(
        OUT_NPZ,
        R=R_DENSE, lam_bar=lam_col, z_bar=z_col,
        total=total, rnd=rnd, cl=cl,
    )
    print(f"[rnd_cl_vs_R] wrote {OUT_NPZ}  ({Nbin} bins x {NR} R-points)")


if __name__ == "__main__":
    main()
