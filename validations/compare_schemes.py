"""Compare z-integration Option E vs Option D against scipy.quad truth.

Run:  python validations/compare_schemes.py
"""
import numpy as np
import time

from richness_selection import Cosmology, PkGrid, HMF, Bias, MOR, XiNL, SelBias
from richness_selection.sigma_m import SigmaM
from richness_selection.config import GridConfig


# Truth values at (lob, zob) = (20, 0.5), from scipy.integrate.quad (rtol=1e-5)
# computed by validations/quad_validate.py.
TRUTH_P1 = 1.997627
TRUTH_I2 = 3.696605e-1
TRUTH_I1 = 2.493808e-1


def _bench(sb, lob, zob):
    """Time one bias_precompute + b_sel(Delta_prj=10, theta->0)."""
    t0 = time.perf_counter()
    pre = sb.bias_precompute(lob, zob)
    b0 = float(sb.b_lob_theta(np.array([1e-9]),
                              lob - 10.0, zob, lob)[0])
    dt = (time.perf_counter() - t0) * 1e3
    return pre, b0, dt


def main():
    cosmo = Cosmology()
    pk = PkGrid(cosmo); sm = SigmaM(pk)
    hmf = HMF(sm); bias = Bias(sm); mor = MOR(); xi = XiNL(cosmo)

    print('=' * 90)
    print('Convergence vs scipy.quad truth at (lob=20, zob=0.5)')
    print('Truth: P1=%.4e  I2=%.4e  I1=%.4e' % (TRUTH_P1, TRUTH_I2, TRUTH_I1))
    print('=' * 90)
    print(f'{"Nz":>4s} {"scheme":>7s}   {"P1 err%":>8s} {"I2 err%":>8s} '
          f'{"I1 err%":>8s}  {"b(Dprj=10)":>11s}  {"time":>8s}')
    for Nz in (40, 80, 160, 320):
        for scheme in ('E', 'D'):
            g = GridConfig(Nz=Nz)
            sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi,
                         grid=g, z_scheme=scheme)
            pre, b0, dt = _bench(sb, 20.0, 0.5)
            eP1 = (pre['P1'] - TRUTH_P1) / TRUTH_P1 * 100
            eI2 = (pre['I2'] - TRUTH_I2) / TRUTH_I2 * 100
            eI1 = (pre['I1'] - TRUTH_I1) / TRUTH_I1 * 100
            print(f'{Nz:>4d} {scheme:>7s}   {eP1:>+7.3f}% {eI2:>+7.3f}% '
                  f'{eI1:>+7.3f}%  {b0:>11.4f}  {dt:>7.1f}ms')
        print()

    print('=' * 90)
    print('D vs E relative difference at Nz=80 across (lob, zob) bins')
    print('=' * 90)
    print(f'{"lob":>4s} {"zob":>4s}  {"P1 D/E":>8s}  {"I2 D/E":>8s}  '
          f'{"I1 D/E":>8s}  {"b(D=10)_E":>10s}  {"b(D=10)_D":>10s}')
    for lob, zob in [(20, 0.3), (20, 0.5), (20, 0.7),
                     (50, 0.3), (50, 0.5), (50, 0.7)]:
        g = GridConfig(Nz=80)
        sb_E = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi, grid=g, z_scheme='E')
        sb_D = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi, grid=g, z_scheme='D')
        pE, b_E, _ = _bench(sb_E, lob, zob)
        pD, b_D, _ = _bench(sb_D, lob, zob)
        print(f'{lob:>4.0f} {zob:>4.2f}  {pD["P1"]/pE["P1"]:>7.4f}  '
              f'{pD["I2"]/pE["I2"]:>7.4f}  {pD["I1"]/pE["I1"]:>7.4f}  '
              f'{b_E:>10.3f}  {b_D:>10.3f}')


if __name__ == '__main__':
    main()
