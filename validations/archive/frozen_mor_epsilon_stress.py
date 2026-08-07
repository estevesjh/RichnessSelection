"""Stress test: frozen-physics precision vs the MOR redshift-evolution
slope epsilon (l_sat(M,z) propto ((1+z)/(1+z_pivot))^epsilon).

Freezing the MOR at z_ob (docs/richness_selection_frozen.tex Sec.
"Frozen physics") assumes P(lambda|M,z) drifts slowly across the
exclusion-affected zone. epsilon directly controls that drift rate --
this is the one MOR parameter that speaks straight to the frozen
approximation's core assumption, so it is the right dial to stress.

epsilon in [-2, 2] at z_ob = 0.60 (fiducial default is epsilon=0.0);
compares the frozen reformulation (validations/frozen_kernels.py) to
the scipy.quad matched-inner truth, for two richness bins.
"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import CACHE_DIR
from frozen_kernels import assemble, frozen_P1, quad_truth

from richness_selection import Cosmology, PkGrid, HMF, Bias, MOR, XiNL, SelBias
from richness_selection.sigma_m import SigmaM

ZOB = 0.60
LOB_POINTS = (20.0, 52.5, 130.0)
EPSILONS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


def build_stack_eps(cosmo, pk, sm, hmf, bias, xi, epsilon):
    """Same stack as frozen_kernels.build_stack(), but with MOR(epsilon=...)."""
    mor = MOR(epsilon=epsilon)
    return dict(cosmo=cosmo, pk=pk, sm=sm, hmf=hmf, bias=bias, mor=mor, xi=xi)


def main():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    xi = XiNL(cosmo)
    xi.build()

    rows = []
    print(f"z_ob = {ZOB}\n")
    print(f"{'lob':>6} {'eps':>5} {'src':>6}   "
          f"{'P1 err%':>9} {'I1 err%':>9} {'I2 err%':>9}")

    for lob in LOB_POINTS:
        for eps in EPSILONS:
            stack = build_stack_eps(cosmo, pk, sm, hmf, bias, xi, eps)
            truth = quad_truth(stack, lob, ZOB)
            I1, I2, aux = assemble(stack, lob, ZOB)
            P1 = frozen_P1(stack, lob, ZOB)

            e_P1 = abs(P1 / truth['P1'] - 1) * 100
            e_I1 = abs(I1 / truth['I1'] - 1) * 100
            e_I2 = abs(I2 / truth['I2'] - 1) * 100

            # production: the actual SelBias grid pipeline (Nz_bias=48
            # ring+outer GL), same stack (same epsilon), same quad truth
            sb = SelBias(stack['cosmo'], stack['pk'], stack['hmf'],
                        stack['bias'], stack['mor'], xi_nl=stack['xi'])
            pre_prod = sb.bias_precompute(lob, ZOB)
            e_P1_prod = abs(pre_prod['P1'] / truth['P1'] - 1) * 100
            e_I1_prod = abs(pre_prod['I1'] / truth['I1'] - 1) * 100
            e_I2_prod = abs(pre_prod['I2'] / truth['I2'] - 1) * 100

            # how strongly the MOR itself now drifts across the exclusion
            # ring (+-dz_excl around z_ob), for context on the dial's effect
            dz_excl = aux['R'] / float(np.gradient(
                cosmo.chi(np.linspace(ZOB - 0.01, ZOB + 0.01, 5)),
                np.linspace(ZOB - 0.01, ZOB + 0.01, 5))[2])
            drift = ((1 + ZOB + dz_excl) / (1 + ZOB)) ** eps - 1.0

            print(f"{lob:>6.1f} {eps:>5.1f} {'prod':>6}   "
                  f"{e_P1_prod:>8.4f}% {e_I1_prod:>8.4f}% {e_I2_prod:>8.4f}%")
            print(f"{'':>6} {'':>5} {'frozen':>6}   "
                  f"{e_P1:>8.4f}% {e_I1:>8.4f}% {e_I2:>8.4f}%")
            rows.append(dict(lob=lob, zob=ZOB, epsilon=eps,
                             P1=P1, I1=I1, I2=I2,
                             P1_quad=truth['P1'], I1_quad=truth['I1'],
                             I2_quad=truth['I2'],
                             P1_prod=pre_prod['P1'], I1_prod=pre_prod['I1'],
                             I2_prod=pre_prod['I2'],
                             err_P1_pct=e_P1, err_I1_pct=e_I1, err_I2_pct=e_I2,
                             err_P1_prod_pct=e_P1_prod,
                             err_I1_prod_pct=e_I1_prod,
                             err_I2_prod_pct=e_I2_prod,
                             mor_ring_drift_pct=drift * 100))
        print()

    out = os.path.join(CACHE_DIR, "frozen_mor_epsilon_stress.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")

    worst = max(rows, key=lambda r: max(r['err_P1_pct'], r['err_I1_pct'], r['err_I2_pct']))
    print(f"\nWorst case (frozen): lob={worst['lob']}, eps={worst['epsilon']}: "
          f"P1={worst['err_P1_pct']:.4f}%, I1={worst['err_I1_pct']:.4f}%, "
          f"I2={worst['err_I2_pct']:.4f}%")
    worst_p = max(rows, key=lambda r: max(r['err_P1_prod_pct'], r['err_I1_prod_pct'], r['err_I2_prod_pct']))
    print(f"Worst case (production): lob={worst_p['lob']}, eps={worst_p['epsilon']}: "
          f"P1={worst_p['err_P1_prod_pct']:.4f}%, I1={worst_p['err_I1_prod_pct']:.4f}%, "
          f"I2={worst_p['err_I2_prod_pct']:.4f}%")


if __name__ == "__main__":
    main()
