"""Cross-check S_i(M, z, [lam_min, lam_max]) against scipy.integrate.nquad.

Reference integral (2-D):

    S_i(M, z) = int_{lam_min}^{lam_max} dlob
                int_0^inf           dltr
                P(lob | ltr, z) P(ltr | M, z).

This is the 2-D form of Eq. (8)/(15) of docs/richness_selection_function.tex,
before the analytic lob reduction. If our closed-form S_i matches this
quad truth, the Gaussian/EMG CDFs are correct.

The zob integral (Eq. 12 K_j) is a Gaussian CDF difference and does not
need a quad check.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import nquad

from richness_selection import MOR, LogNormalMOR
from richness_selection.selection_function import S_i
from richness_selection.plob_ltr import P_lob_given_ltr


def S_i_nquad(M, z, lam_min, lam_max, mor, rtol=1e-5):
    """Brute 2D quadrature reference."""
    def f(ltr, lob):
        return (P_lob_given_ltr(lob, ltr, z)
                * mor.pdf(np.asarray(ltr), M, z))
    # ltr support: use the MOR moments to bracket
    mu = float(mor.ltr_mean(M, z))
    sig = float(mor.ltr_sigma(M, z))
    ltr_lo = max(0.0, mu - 10.0 * sig)
    ltr_hi = mu + 10.0 * sig
    val, _ = nquad(f, [[ltr_lo, ltr_hi], [lam_min, lam_max]],
                    opts=[{"epsrel": rtol, "limit": 200},
                          {"epsrel": rtol, "limit": 200}])
    return float(val)


def main():
    hod = MOR()
    logn = LogNormalMOR()

    bins = [(20., 30.), (30., 45.), (45., 60.)]
    z_test = [0.3, 0.5, 0.7]
    M_test = [5e13, 1e14, 3e14, 1e15]

    print('== HOD MOR ==')
    print(f'{"z":>5} {"M":>10} {"bin":>14} '
          f'{"ours":>10} {"quad":>10} {"rel.err":>10}')
    for z in z_test:
        for M in M_test:
            for (lo, hi) in bins:
                ours = S_i(M, z, lo, hi, hod, N_q=64)
                quad = S_i_nquad(M, z, lo, hi, hod)
                if abs(quad) < 1e-10:
                    rel = 0.0 if abs(ours) < 1e-10 else float('nan')
                else:
                    rel = abs(ours / quad - 1.0)
                print(f'{z:>5.2f} {np.log10(M):>10.2f} '
                      f'[{lo:>3g},{hi:>3g}]    '
                      f'{ours:>10.4e} {quad:>10.4e} {rel:>10.2e}')

    print()
    print('== LogNormal MOR ==')
    for z in z_test:
        for M in M_test:
            for (lo, hi) in bins:
                ours = S_i(M, z, lo, hi, logn, N_q=64)
                quad = S_i_nquad(M, z, lo, hi, logn)
                if abs(quad) < 1e-10:
                    rel = 0.0 if abs(ours) < 1e-10 else float('nan')
                else:
                    rel = abs(ours / quad - 1.0)
                print(f'{z:>5.2f} {np.log10(M):>10.2f} '
                      f'[{lo:>3g},{hi:>3g}]    '
                      f'{ours:>10.4e} {quad:>10.4e} {rel:>10.2e}')


if __name__ == '__main__':
    main()
