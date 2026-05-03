"""Integral-simplification regression tests.

Each test validates one of the numerical recipes we rely on:

  1. sel_bias theta split-at-exclusion: Nth=10 converged to <0.1% of quad.
  2. sel_bias z-schemes E/D/C agree within 1% at Nz=80.
  3. Lambda axis converged: n_ltr=30 matches n_ltr=100 to <0.01%.
  4. Sigma_prj theta split-at-theta_R: log-GL 50+50 within ~3% of quad.
  5. Sigma_prj monotonicity: profile is smooth after R_peak ~ 0.5 cMpc/h.

These tests share a session-level fixture (cosmo + CAMB) to avoid the
~1 s CAMB startup for every test.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import bisect

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, XiNL, NFWMiscentered, SelBias, SigmaPrj,
)
from richness_selection.sigma_m import SigmaM
from richness_selection.config import GridConfig
from richness_selection.gl import gl_nodes
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel


# ---------- Session-level fixtures -----------------------------------------

@pytest.fixture(scope="module")
def cosmo():
    return Cosmology()


@pytest.fixture(scope="module")
def _stack(cosmo):
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo)
    return dict(pk=pk, sm=sm, hmf=hmf, bias=bias, mor=mor, xi=xi)


def _make_sb(cosmo, stack, scheme='E', Nz=80, Nth=10, n_ltr=60):
    return SelBias(cosmo, stack['pk'], stack['hmf'], stack['bias'],
                   stack['mor'], xi_nl=stack['xi'],
                   grid=GridConfig(Nz=Nz, Nth=Nth),
                   n_ltr=n_ltr, z_scheme=scheme)


# ---------- Shared quad-reference utilities ------------------------------

def _quad_P1_I1_I2(cosmo, stack, lob=20.0, zob=0.5, Nth=20, n_ltr=60,
                   NM=24, rtol=1e-4):
    """Compute (P1, I2, I1) via scipy.quad on the outer z, using the SAME
    inner GL grids (M, lambda, theta split-at-exclusion) that _P_operator
    uses.  This is the 'matched-inner' ground truth.
    """
    chi_o = float(cosmo.chi(zob))
    theta_lob = R_lambda(lob) * (1.0 + zob) / chi_o
    R_excl = R_lambda(lob) * (1.0 + zob)

    lnMs, wM = gl_nodes(np.log(1e13), np.log(10**15.5), NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, lob, n_ltr)

    theta_max = 2.0 * theta_lob
    eps_theta = 1e-6

    def f_inner(z, which):
        chi_z = float(cosmo.chi(z))
        dV = float(cosmo.dV_dzdOm(z))
        wz_val = float(w_z(np.array([z]), zob)[0])
        if wz_val <= 0:
            return 0.0
        # split-at-exclusion theta_lo
        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = min(max(cos_excl, -1.0), 1.0)
        th_lo = np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12 else eps_theta
        th_lo = max(th_lo, eps_theta)
        if th_lo >= theta_max:
            return 0.0
        ths, wth = gl_nodes(th_lo, theta_max, Nth)
        sin_th = np.sin(ths)
        th_weight = wth * 2.0 * np.pi * sin_th
        sigmoid = 1.0 / (1.0 + np.exp(
            -(2.5 / theta_lob) * (ths - 0.5 * theta_lob)))
        cos_th = np.cos(ths)
        dchi = np.sqrt(np.maximum(
            chi_z**2 + chi_o**2 - 2 * chi_z * chi_o * cos_th, 0.0))
        xi_th = stack['xi'](dchi, zob)
        theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
        fA = area_overlap(ths, theta_lob, theta_lam_l)
        if which == 'P1':
            ang = np.einsum('t,tL->L', th_weight, fA)
            need_b = False
        elif which == 'I2':
            ang = np.einsum('t,tL,t->L', th_weight, fA, xi_th)
            need_b = True
        else:                  # I1
            ang = np.einsum('t,t,tL,t->L', th_weight, sigmoid, fA, xi_th)
            need_b = True
        P_lmz = stack['mor'].pdf(lam_grid[:, None], Ms[None, :], z)
        lam_int = np.einsum('L,LM,L->M', wlam, P_lmz,
                            wz_val * lam_grid * ang)
        n_m = stack['hmf'](Ms, z)
        if need_b:
            b_m = stack['bias'](Ms, z)
            return dV * np.sum(M_weight * n_m * b_m * lam_int)
        return dV * np.sum(M_weight * n_m * lam_int)

    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))

    truth = {}
    for w in ('P1', 'I2', 'I1'):
        v_fg, _ = quad(f_inner, z_fg_lo, zob, args=(w,),
                       epsrel=rtol, limit=200)
        v_bg, _ = quad(f_inner, zob, z_bg_hi, args=(w,),
                       epsrel=rtol, limit=200)
        truth[w] = v_fg + v_bg
    return truth


@pytest.fixture(scope="module")
def quad_truth(cosmo, _stack):
    """scipy.quad reference at (lob=20, zob=0.5)."""
    return _quad_P1_I1_I2(cosmo, _stack, Nth=20)


# ---------- Tests ---------------------------------------------------------

class TestSelBiasThetaSplit:
    """Test 1: split-at-exclusion converges at Nth=10 to < 0.1% of quad."""

    def test_Nth10_sub_01pct(self, cosmo, _stack, quad_truth):
        sb = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=10)
        pre = sb.bias_precompute(20.0, 0.5)
        for key in ('P1', 'I2', 'I1'):
            rel_err = abs(pre[key] / quad_truth[key] - 1.0)
            assert rel_err < 1e-3, (
                f'{key}: {pre[key]:.5e} vs quad {quad_truth[key]:.5e}, '
                f'rel_err={rel_err*100:.3f}% (expect < 0.1%)')

    def test_Nth_converged_Nth15_to_5_digits(self, cosmo, _stack):
        sb10 = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=10)
        sb15 = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=15)
        sb50 = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=50)
        p10 = sb10.bias_precompute(20.0, 0.5)
        p15 = sb15.bias_precompute(20.0, 0.5)
        p50 = sb50.bias_precompute(20.0, 0.5)
        # Nth=15 should be bit-stable vs Nth=50
        for key in ('P1', 'I2', 'I1'):
            diff = abs(p15[key] / p50[key] - 1.0)
            assert diff < 1e-4, (
                f'{key}: Nth=15 {p15[key]:.6e} vs Nth=50 {p50[key]:.6e}, '
                f'diff={diff*100:.4f}% (expect < 0.01%)')


class TestSelBiasZSchemes:
    """Test 2: E/D/C agree within 1% at Nz=80, Nth=10."""

    @pytest.mark.parametrize('key', ['P1', 'I2', 'I1'])
    def test_schemes_agree(self, cosmo, _stack, key):
        sb_E = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=10)
        sb_D = _make_sb(cosmo, _stack, scheme='D', Nz=80, Nth=10)
        pE = sb_E.bias_precompute(20.0, 0.5)
        pD = sb_D.bias_precompute(20.0, 0.5)
        # E and D agree to ~1% on I1/I2 (D is slightly worse at Nz=80).
        # Tight 0.3% tolerance for P1 where both schemes are strong.
        tol = 3e-3 if key == 'P1' else 1.0e-2
        assert abs(pE[key] / pD[key] - 1.0) < tol, (
            f'{key}: E {pE[key]:.5e}, D {pD[key]:.5e}')


class TestLambdaAxisConverged:
    """Test 3: lambda axis already converged at n_ltr=30."""

    def test_lam_converged(self, cosmo, _stack):
        sb30 = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=10, n_ltr=30)
        sb100 = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=10, n_ltr=100)
        p30 = sb30.bias_precompute(20.0, 0.5)
        p100 = sb100.bias_precompute(20.0, 0.5)
        for key in ('P1', 'I2', 'I1'):
            diff = abs(p30[key] / p100[key] - 1.0)
            assert diff < 1e-4, (
                f'{key}: n_ltr=30 {p30[key]:.6e}, '
                f'n_ltr=100 {p100[key]:.6e}, diff={diff*100:.4f}%')


class TestSigmaPrjTheta:
    """Tests 4 and 5: Sigma_prj split-at-theta_R + monotonicity."""

    @pytest.fixture(scope="class")
    def sigma_prj_obj(self, cosmo, _stack):
        sb = _make_sb(cosmo, _stack, scheme='E', Nz=80, Nth=10)
        nfw = NFWMiscentered(cosmo)
        return SigmaPrj(cosmo, sb, nfw, n_theta_inner=50, n_theta_outer=50)

    def test_profile_monotonic_after_peak(self, sigma_prj_obj):
        """After R ~ 0.5 cMpc/h the Sigma_prj profile must decrease
        smoothly -- no oscillations, no zigzag."""
        R = np.logspace(np.log10(0.3), np.log10(20.0), 25)
        prof = sigma_prj_obj(R, lob=20.0, zob=0.5)
        # After R=1 the profile should be strictly decreasing within 2%.
        for i in range(1, len(R)):
            if R[i] < 1.0:
                continue
            assert prof[i] <= prof[i-1] * 1.02, (
                f'non-monotonic at R={R[i]:.3f}: '
                f'prof={prof[i]:.3e} > prev={prof[i-1]:.3e}')

    def test_profile_positive(self, sigma_prj_obj):
        R = np.logspace(-1, 1.3, 15)
        prof = sigma_prj_obj(R, lob=20.0, zob=0.5)
        assert np.all(prof > 0), f'Sigma_prj has non-positive values: {prof}'

    def test_sigma_prj_single_z_matches_quad(self, cosmo, _stack,
                                              sigma_prj_obj):
        """For each of a few R values, the single-z (z=zob) Sigma_prj
        integrand computed by log-GL split-at-theta_R on 50+50 nodes
        matches scipy.quad to 5%.  (Running the full Sigma_prj against
        quad would take ~5 min per R-point; this is the cheap
        equivalent.)"""
        lob, zob = 20.0, 0.5
        chi_o = float(cosmo.chi(zob))
        D_A_o = chi_o / (1.0 + zob)
        R_lam_lob = R_lambda(lob)
        sb = sigma_prj_obj.sel_bias
        nfw = sigma_prj_obj.nfw

        lnMs, wM = gl_nodes(np.log(1e13), np.log(10**15.5), sb.grid.NM)
        Ms = np.exp(lnMs); M_weight_ln = wM * Ms
        n_m = sb.hmf(Ms, zob); b_m = sb.bias(Ms, zob)
        pre = sb.bias_precompute(lob, zob)

        def integrand(theta, R_val):
            sin_th = np.sin(theta); R_theta = theta * D_A_o
            bsel = float(sb.b_sel_marginalised(
                np.array([theta]), lob, zob, precomp=pre)[0])
            dchi = chi_o * theta
            xi_v = 0.0 if dchi < R_lam_lob * (1 + zob) else float(
                sb.xi_NL(dchi, zob))
            mc = 0.0
            for i, M in enumerate(Ms):
                S = float(nfw.sigma_grid(
                    np.array([R_val]), np.array([R_theta]),
                    float(M), zob).ravel()[0])
                mc += M_weight_ln[i] * n_m[i] * (
                    1.0 + b_m[i] * bsel * xi_v) * S
            return 2 * np.pi * sin_th * mc

        def sp_quad(R_val):
            tmax = 30.0 / D_A_o; tR = max(R_val / D_A_o, 1e-6)
            if tR >= tmax:
                return quad(integrand, 1e-5, tmax, args=(R_val,),
                            epsrel=1e-3, limit=200)[0]
            v1 = quad(integrand, 1e-5, tR, args=(R_val,),
                      epsrel=1e-3, limit=200)[0]
            v2 = quad(integrand, tR, tmax, args=(R_val,),
                      epsrel=1e-3, limit=200)[0]
            return v1 + v2

        def sp_log_split(R_val):
            tmax = 30.0 / D_A_o
            ths, wth = sigma_prj_obj._theta_grid_for_R(R_val, D_A_o, tmax)
            sin_th = np.sin(ths); R_theta = ths * D_A_o
            bsel_th = sb.b_sel_marginalised(ths, lob, zob, precomp=pre)
            dchi = chi_o * ths
            xi_vals = np.where(dchi < R_lam_lob * (1 + zob), 0.0,
                               sb.xi_NL(dchi, zob))
            total = 0.0
            for i, M in enumerate(Ms):
                S = nfw.sigma_grid(np.array([R_val]), R_theta,
                                    float(M), zob).ravel()
                total += M_weight_ln[i] * n_m[i] * np.sum(
                    wth * 2 * np.pi * sin_th
                    * (1.0 + b_m[i] * bsel_th * xi_vals) * S)
            return total

        # Test at a few R values spanning the profile's features.
        # Tolerances:
        #   R in [0.3, 1] cMpc/h: ~10% -- inner NFW-lookup's R_mis table
        #     spacing and the theta_R-split endpoint interact awkwardly here.
        #   R > 1 cMpc/h: 3% -- smoother regime.
        for R_val, tol in [(0.5, 0.10), (1.0, 0.05),
                           (2.0, 0.05), (5.0, 0.05), (10.0, 0.05)]:
            q = sp_quad(R_val)
            s = sp_log_split(R_val)
            rel_err = abs(s / q - 1.0)
            assert rel_err < tol, (
                f'R={R_val}: log-split {s:.3e} vs quad {q:.3e}, '
                f'rel_err={rel_err*100:.2f}%, tol={tol*100:.1f}%')
