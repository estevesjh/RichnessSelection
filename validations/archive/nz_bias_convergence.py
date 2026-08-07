"""Nz_bias convergence study for SelBias._P_operator.

Companion to the z-axis analytic-exclusion review (docs/richness_selection.tex
Sec. "z-axis"). Original plan proposed replacing the ring+outer GL scheme
with a polynomial-fit + closed-form integral; that was dropped after this
script's sibling check found xi_NL goes negative over ~20-25% of the outer
decay range at higher zob (BAO-scale separations), which breaks a log-space
fit. Instead: the existing (already-validated) ring+outer GL code has slack
-- hard floors (n_ring>=9, n_outer>=15 per side) mean Nz below ~40 was
already "free" before this change. This script pins the chosen default
(Nz_bias=48) against a high-precision (Nz=200, Nth=30) reference across all
12 DES Y3 (lob, zob) bins, and reports the wall-clock speedup vs Nz=80.

SigmaPrj/DeltaSigmaPrj are unaffected: they read GridConfig.Nz (unchanged,
still 80) for their own z-integration via SelBias._z_grid; only
SelBias._P_operator's P[1]/I1/I2 use the new, separate Nz_bias field.
"""
import csv
import os
import time

from richness_selection import Cosmology, PkGrid, HMF, Bias, MOR, XiNL, SelBias
from richness_selection.sigma_m import SigmaM
from richness_selection.config import GridConfig
from richness_selection.des_y3 import iter_bins

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def main():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo)

    points = [(lam_bar, z_bar) for (_, _, (lam_bar, z_bar)) in iter_bins()]

    print("Building high-precision reference (Nz=200, Nth=30) ...")
    sb_ref = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi,
                     grid=GridConfig(Nz_bias=200, Nth=30))
    sb_ref._P_operator(*points[0])  # warm (lazy XiNL FFTlog build)
    ref = {pt: sb_ref._P_operator(*pt) for pt in points}

    rows = []
    print(f'\n{"Nz_bias":>8} {"nring":>6} {"nouter":>6} {"tot":>4} '
          f'{"maxErrP1":>9} {"maxErrI2":>9} {"maxErrI1":>9}  {"avg ms":>8}')
    for Nz_bias in (80, 70, 60, 56, 52, 48, 44, 40, 36, 32):
        n_ring = max(9, Nz_bias // 4)
        n_outer = max(15, (Nz_bias - n_ring) // 2)
        tot = n_ring + 2 * n_outer
        sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi,
                    grid=GridConfig(Nz_bias=Nz_bias))
        sb._P_operator(*points[0])  # warm
        errs = {"P1": [], "I2": [], "I1": []}
        times = []
        for pt in points:
            t0 = time.perf_counter()
            P1, I1, I2 = sb._P_operator(*pt)
            times.append(time.perf_counter() - t0)
            rP1, rI1, rI2 = ref[pt]
            errs["P1"].append(abs(P1 / rP1 - 1) * 100)
            errs["I1"].append(abs(I1 / rI1 - 1) * 100)
            errs["I2"].append(abs(I2 / rI2 - 1) * 100)
        avg_ms = sum(times) / len(times) * 1e3
        print(f'{Nz_bias:>8d} {n_ring:>6d} {n_outer:>6d} {tot:>4d} '
              f'{max(errs["P1"]):>8.4f}% {max(errs["I2"]):>8.4f}% '
              f'{max(errs["I1"]):>8.4f}%  {avg_ms:>7.2f}')
        rows.append(dict(Nz_bias=Nz_bias, n_ring=n_ring, n_outer=n_outer,
                         n_total_nodes=tot,
                         max_err_P1_pct=max(errs["P1"]),
                         max_err_I2_pct=max(errs["I2"]),
                         max_err_I1_pct=max(errs["I1"]),
                         avg_ms=avg_ms))

    out_path = os.path.join(CACHE_DIR, "nz_bias_convergence.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_path}")
    print("Chosen default: Nz_bias=48 (comfortable margin under the 0.1%"
          " quad-matched test tolerance; see GridConfig docstring).")


if __name__ == "__main__":
    main()
