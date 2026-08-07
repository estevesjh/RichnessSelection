"""Benchmark: FrozenSelBias vs production SelBias bias_precompute.

Times the fresh compute (cache cleared) and the cached repeat for both
methods across the DES-Y3-like reference points.

Run from the repo root:
    python validations/benchmark_frozen.py
"""
import time

import numpy as np

from richness_selection import (Cosmology, PkGrid, HMF, Bias, MOR, XiNL,
                                SelBias, FrozenSelBias)
from richness_selection.sigma_m import SigmaM

POINTS = [(20.0, 0.500), (52.5, 0.425), (130.0, 0.575)]
N_REPEATS = 5


def build():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo); xi.build()
    args = (cosmo, pk, hmf, bias, mor)
    return (SelBias(*args, xi_nl=xi), FrozenSelBias(*args, xi_nl=xi))


def time_method(obj, lob, zob):
    fresh = []
    for _ in range(N_REPEATS):
        obj._cache.clear()
        t0 = time.perf_counter()
        obj.bias_precompute(lob, zob)
        fresh.append(1e3 * (time.perf_counter() - t0))
    cached = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        obj.bias_precompute(lob, zob)
        cached.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(fresh)), float(np.median(cached))


def main():
    print("Building stack (one-time CAMB/halofit cost)...")
    sel, fsel = build()

    hdr = (f"{'lob':>6} {'zob':>6} {'prod fresh':>12} {'prod cached':>12} "
           f"{'frozen fresh':>13} {'frozen cached':>14}")
    print("\nmedian of "
          f"{N_REPEATS} runs, ms")
    print(hdr)
    print("-" * len(hdr))
    for lob, zob in POINTS:
        pf, pc = time_method(sel, lob, zob)
        ff, fc = time_method(fsel, lob, zob)
        print(f"{lob:6.1f} {zob:6.3f} {pf:11.3f} {pc:12.5f} "
              f"{ff:12.3f} {fc:13.5f}")


if __name__ == "__main__":
    main()
