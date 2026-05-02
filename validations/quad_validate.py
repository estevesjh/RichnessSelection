"""Independent scipy.quad validation of P[1], I1, I2 at (lob=20, zob=0.5).

Builds a scalar z-integrand that, at each z, carries out the
(M, lambda, theta) integrations with Matteo-style fixed grids (so
the only variable integration method is the outermost z-axis).
Then compares scipy.integrate.quad (adaptive, rtol=1e-6) to our
fast Nz=80 Simpson+pinned path.

If quad ≈ our Nz=80 to a few parts in 1e4, the z-axis is converged
and the remaining paper gap is not in z-integration.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, XiNL, SelBias,
)
from richness_selection.sigma_m import SigmaM
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, sigma_z, zmin4zkernel, zmax4zkernel

cosmo = Cosmology()
pk = PkGrid(cosmo); sm = SigmaM(pk)
hmf = HMF(sm); bias = Bias(sm); mor = MOR(); xi = XiNL(cosmo)
sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)

lob, zob = 20.0, 0.5
theta_lob = sb._theta_lob(lob, zob)
chi_o = float(cosmo.chi(zob))
R_excl = R_lambda(lob) * (1.0 + zob)

# Fixed inner grids (Matteo-style)
m_grid = 10.0 ** np.linspace(13, 15.5, 100)          # NM = 100 trapz
ln_m = np.log(m_grid)


def _inner_at_z(z, weight_type):
    """Compute the inner (M, lambda, theta) triple integral at one z.

    weight_type: 'P1' -> P[1] integrand (no b, no xi)
                 'I2' -> P[b xi]
                 'I1' -> P[b xi sigmoid]

    Returns f(z) such that outer integral = int dz * f(z).
    That is: f(z) = (dV/dzdOm)(z) * w_z(z, zob) * [int dlnM M n(M,z) b(M,z)
                                                   * int dlam P(lam|M,z) lam
                                                     * 2pi int dtheta sin(theta) f_A(...) xi(...)]
    """
    chi_z = float(cosmo.chi(z))
    dV = float(cosmo.dV_dzdOm(z))
    wz_val = float(w_z(np.array([z]), zob)[0])
    if wz_val <= 0:
        return 0.0

    # Matteo-style theta grid per-ltr: geomspace(1e-6, theta_lob + theta_ltr(lam), 50)
    # We'll evaluate for each lam in ltr_grid
    ltr_grid = np.linspace(1e-10, lob, 100)
    theta_ltr = R_lambda(ltr_grid) * (1.0 + z) / chi_z                     # (Nltr,)
    theta_max = theta_lob + theta_ltr                                       # (Nltr,)
    theta_grid = np.geomspace(1e-6, theta_max, 50)                          # (Nth, Nltr)
    # 3-D separation
    dis = np.sqrt(chi_z**2 + chi_o**2
                  - 2.0 * chi_z * chi_o * np.cos(theta_grid))               # (Nth, Nltr)
    xi_vals = xi(dis.ravel(), zob).reshape(dis.shape)                        # (Nth, Nltr)
    xi_vals = np.where(dis < R_excl, 0.0, xi_vals)

    # Sigmoid (per-theta)
    x0 = 0.5 * theta_lob
    k = 2.5 / theta_lob
    sigmoid = 1.0 / (1.0 + np.exp(-k * (theta_grid - x0)))                   # (Nth, Nltr)

    # Angular f_A(theta, theta_lob, theta_ltr)  -- same function as Matteo's area_overlap
    fA = area_overlap(theta_grid, theta_lob, theta_ltr)                      # (Nth, Nltr)
    sin_theta = np.sin(theta_grid)                                           # (Nth, Nltr)

    if weight_type == 'P1':
        # Angular integral:  2 pi int dtheta sin theta f_A
        ang = np.trapz(2.0 * np.pi * sin_theta * fA, theta_grid, axis=0)    # (Nltr,)
    elif weight_type == 'I2':
        ang = np.trapz(2.0 * np.pi * sin_theta * fA * xi_vals,
                       theta_grid, axis=0)                                   # (Nltr,)
    elif weight_type == 'I1':
        ang = np.trapz(2.0 * np.pi * sin_theta * fA * xi_vals * sigmoid,
                       theta_grid, axis=0)                                   # (Nltr,)
    else:
        raise ValueError(weight_type)

    # Lambda integral: int dlam P(lam|M,z) lam * ang(lam)
    # P(lam|M,z) shape (Nltr, NM)
    P_lmz = mor.pdf(ltr_grid[:, None], m_grid[None, :], z)
    lam_integrand = (ltr_grid[:, None] * ang[:, None]) * P_lmz              # (Nltr, NM)
    lam_int = np.trapz(lam_integrand, ltr_grid, axis=0)                     # (NM,)

    # M integral: int dlnM M n(M,z) [b(M,z)] lam_int(M)
    n_m = hmf(m_grid, z)
    if weight_type == 'P1':
        M_integrand = m_grid * n_m * lam_int
    else:
        b_m = bias(m_grid, z)
        M_integrand = m_grid * n_m * b_m * lam_int
    M_int = np.trapz(M_integrand, ln_m)

    return dV * wz_val * M_int


# scipy.quad over z, with adaptive subdivision
# Split at zob to handle the spike cleanly
z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))

print(f'z support: [{z_fg_lo:.4f}, {z_bg_hi:.4f}]')
print(f'R_excl    = {R_excl:.4f} cMpc/h')
print(f'dchi/dz at zob = {float((cosmo.chi(zob+1e-4)-cosmo.chi(zob-1e-4))/2e-4):.1f} cMpc/h')
print()

import time

for weight in ('P1', 'I2', 'I1'):
    t0 = time.perf_counter()
    # Split integration at zob to handle the spike cleanly
    # quad works best with finite subranges around the singularity
    val_fg, err_fg = quad(_inner_at_z, z_fg_lo, zob,
                          args=(weight,), epsrel=1e-5, limit=200)
    val_bg, err_bg = quad(_inner_at_z, zob, z_bg_hi,
                          args=(weight,), epsrel=1e-5, limit=200)
    val = val_fg + val_bg
    err = np.sqrt(err_fg**2 + err_bg**2)
    dt = time.perf_counter() - t0
    print(f'{weight:>3s} (scipy.quad, eps=1e-5): {val:+.6e}  +/- {err:.2e}   [{dt:.1f} s]')

# Compare to our implementation
print()
print('Our Nz=80 Simpson+pinned result:')
pre = sb.bias_precompute(lob, zob)
print(f'  P1 = {pre["P1"]:+.6e}')
print(f'  I2 = {pre["I2"]:+.6e}')
print(f'  I1 = {pre["I1"]:+.6e}')
