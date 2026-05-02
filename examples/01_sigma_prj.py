"""Evaluate Sigma_prj(R | lob=20, zob=0.5) at 20 R-points."""
import numpy as np
import time

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, SelBias, SigmaPrj,
)
from richness_selection.sigma_m import SigmaM


def main():
    t0 = time.perf_counter()
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sigma_m = SigmaM(pk)
    hmf = HMF(sigma_m)
    bias = Bias(sigma_m)
    mor = MOR()
    sel_bias = SelBias(cosmo, pk, hmf, bias, mor)
    nfw = NFWMiscentered(cosmo)
    sp = SigmaPrj(cosmo, sel_bias, nfw)
    print(f"Setup (CAMB + sigma(M) tab): {(time.perf_counter() - t0) * 1e3:.1f} ms")

    pipe = sel_bias.bias_pipeline(20.0, 0.5, 18.0)
    print("\nbias_pipeline(lob=20, zob=0.5, ltr=18):")
    for k, v in pipe.items():
        if isinstance(v, float):
            print(f"  {k:>8s} = {v:+.5e}")

    R = np.logspace(-1, 1.3, 20)
    t0 = time.perf_counter()
    prof = sp(R, 20.0, 0.5)
    print(f"\nSigma_prj(20 R, lob=20, zob=0.5): "
          f"{(time.perf_counter() - t0) * 1e3:.2f} ms")
    for r, s in zip(R, prof):
        print(f"  R = {r:7.4f} cMpc/h   Sigma = {s:.4e}")


if __name__ == "__main__":
    main()
