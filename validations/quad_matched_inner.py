"""scipy.quad with the SAME inner (M, lam, theta) grid as _P_operator.

If E/D match this quad result to sub-%, then the z-axis integration
is converged and the oscillation is purely the inner-grid convention
difference with the original quad_validate.py.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect
import time

from richness_selection import Cosmology, PkGrid, HMF, Bias, MOR, XiNL, SelBias
from richness_selection.sigma_m import SigmaM
from richness_selection.config import GridConfig
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel
from richness_selection.gl import gl_nodes

cosmo = Cosmology()
pk = PkGrid(cosmo); sm = SigmaM(pk)
hmf = HMF(sm); bias = Bias(sm); mor = MOR(); xi = XiNL(cosmo)

lob, zob = 20.0, 0.5
theta_lob = R_lambda(lob) * (1.0 + zob) / float(cosmo.chi(zob))
chi_o = float(cosmo.chi(zob))
R_excl = R_lambda(lob) * (1.0 + zob)

# Use _P_operator's EXACT inner grids
g = GridConfig(Nz=80)
lnMs, wM = gl_nodes(np.log(1e13), np.log(10**15.5), g.NM)
Ms = np.exp(lnMs)
M_weight = wM * Ms
lam_grid, wlam = gl_nodes(1e-6, float(lob), 60)
ths, wth = gl_nodes(1e-6, 2.0 * theta_lob, g.Nth)
sin_th = np.sin(ths)
th_weight = wth * 2.0 * np.pi * sin_th
sigmoid_th = 1.0 / (1.0 + np.exp(-(2.5 / theta_lob) *
                                  (ths - 0.5 * theta_lob)))


def f_inner_matched(z, which):
    """Inner integrand at scalar z, using _P_operator's GL grids.
    Returns f(z) such that outer integral = int dz f(z)."""
    chi_z = float(cosmo.chi(z))
    dV = float(cosmo.dV_dzdOm(z))
    wz_val = float(w_z(np.array([z]), zob)[0])
    if wz_val <= 0:
        return 0.0

    cos_th = np.cos(ths)
    dchi = np.sqrt(np.maximum(
        chi_z**2 + chi_o**2 - 2.0 * chi_z * chi_o * cos_th, 0.0))
    xi_th = xi(dchi, zob)
    xi_th = np.where(dchi < R_excl, 0.0, xi_th)
    theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
    fA = area_overlap(ths, theta_lob, theta_lam_l)   # (Nth, Nlam)

    if which == 'P1':
        ang = np.einsum('t,tL->L', th_weight, fA)
    elif which == 'I2':
        ang = np.einsum('t,tL,t->L', th_weight, fA, xi_th)
    elif which == 'I1':
        ang = np.einsum('t,t,tL,t->L', th_weight, sigmoid_th, fA, xi_th)

    P_lmz = mor.pdf(lam_grid[:, None], Ms[None, :], z)
    rho_prefac = wz_val * lam_grid
    lam_I = rho_prefac * ang
    lam_int = np.einsum('L,LM,L->M', wlam, P_lmz, lam_I)

    if which == 'P1':
        return dV * np.sum(M_weight * hmf(Ms, z) * lam_int)
    else:
        return dV * np.sum(M_weight * hmf(Ms, z) * bias(Ms, z) * lam_int)


z_fg_lo = float(bisect(zmin4zkernel, -2., 2., args=(zob,)))
z_bg_hi = float(bisect(zmax4zkernel, -2., 2., args=(zob,)))

print(f'z support [{z_fg_lo:.4f}, {z_bg_hi:.4f}], zob={zob}')
print()

truth_matched = {}
for which in ('P1', 'I2', 'I1'):
    t0 = time.perf_counter()
    v_fg, _ = quad(f_inner_matched, z_fg_lo, zob,
                   args=(which,), epsrel=1e-5, limit=200)
    v_bg, _ = quad(f_inner_matched, zob, z_bg_hi,
                   args=(which,), epsrel=1e-5, limit=200)
    dt = time.perf_counter() - t0
    truth_matched[which] = v_fg + v_bg
    print(f'{which:>4s}  quad-matched-inner = {truth_matched[which]:+.6e}  [{dt:.1f}s]')

# Compare to the original quad_validate.py truth (trapz-based inner grids)
print()
print('Prior quad truth (trapz inner grids, linspace M/lam, geomspace theta):')
print(f'  P1 = 1.997627e+00')
print(f'  I2 = 3.696605e-01')
print(f'  I1 = 2.493808e-01')
print()
print('% diff (matched - trapz) / trapz:')
print(f'  P1: {(truth_matched["P1"]/1.997627 - 1)*100:+.3f}%')
print(f'  I2: {(truth_matched["I2"]/3.696605e-1 - 1)*100:+.3f}%')
print(f'  I1: {(truth_matched["I1"]/2.493808e-1 - 1)*100:+.3f}%')

# Now compare our SelBias to the *matched* quad
sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi, grid=g)
pE = sb.bias_precompute(lob, zob)

print()
print('Our SelBias at Nz=80 vs matched-inner quad (should be <0.5% z-axis error):')
print(f'  P1: {(pE["P1"]/truth_matched["P1"] - 1)*100:+.3f}%')
print(f'  I2: {(pE["I2"]/truth_matched["I2"] - 1)*100:+.3f}%')
print(f'  I1: {(pE["I1"]/truth_matched["I1"] - 1)*100:+.3f}%')
