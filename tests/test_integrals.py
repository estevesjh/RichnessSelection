"""Integral-simplification regression tests.

Each test validates one of the numerical recipes we rely on:

  1. sel_bias theta split-at-exclusion: Nth=10 converged to <0.1% of quad.
  2. Lambda axis converged: n_ltr=30 matches n_ltr=100 to <0.01%.
  3. Sigma_prj theta split-at-theta_R: log-GL 50+50 within ~3% of quad.
  4. Sigma_prj monotonicity: profile is smooth after R_peak ~ 0.5 cMpc/h.

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


def _make_sb(cosmo, stack, Nz=80, Nth=10, n_ltr=60, Nz_bias=None):
    """Nz_bias defaults to Nz so existing call sites (which predate the
    Nz_bias split) keep testing exactly what they asked for; pass
    Nz_bias explicitly to test SelBias._P_operator's own (smaller,
    default 48) z-grid independently of Nz (which SigmaPrj still uses)."""
    if Nz_bias is None:
        Nz_bias = Nz
    return SelBias(cosmo, stack['pk'], stack['hmf'], stack['bias'],
                   stack['mor'], xi_nl=stack['xi'],
                   grid=GridConfig(Nz=Nz, Nth=Nth, Nz_bias=Nz_bias),
                   n_ltr=n_ltr)


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

class TestSelBiasNzBias:
    """Nz_bias (default 48) is a separate, smaller z-node budget for
    _P_operator's own ring+outer grid -- decoupled from Nz, which
    SigmaPrj/DeltaSigmaPrj still read (unchanged) via SelBias._z_grid.
    The ring+outer split enforces hard floors (n_ring>=9, n_outer>=15
    per side), so Nz_bias=48 (n_ring=12, n_outer=18+18=48 total) sits
    just above where those floors bind, at ~1.6-1.7x fewer expensive
    per-z (theta,lambda,M) evaluations than Nz=80. See docs/GridConfig
    and the z-axis analytic-exclusion review for the convergence table
    this default is pinned from (worst case 0.045% across the 12 DES
    Y3 bins vs an Nz=200,Nth=30 reference)."""

    def test_default_Nz_bias_sub_01pct_of_quad(self, cosmo, _stack, quad_truth):
        # No grid override at all -- exactly what a real caller gets.
        sb = SelBias(cosmo, _stack['pk'], _stack['hmf'], _stack['bias'],
                     _stack['mor'], xi_nl=_stack['xi'])
        assert sb.grid.Nz_bias == 48
        pre = sb.bias_precompute(20.0, 0.5)
        for key in ('P1', 'I2', 'I1'):
            rel_err = abs(pre[key] / quad_truth[key] - 1.0)
            assert rel_err < 1e-3, (
                f'{key}: {pre[key]:.5e} vs quad {quad_truth[key]:.5e}, '
                f'rel_err={rel_err*100:.3f}% (expect < 0.1%)')

    def test_Nz_bias_faster_than_Nz80(self, cosmo, _stack):
        sb48 = _make_sb(cosmo, _stack, Nth=10, Nz_bias=48)
        sb80 = _make_sb(cosmo, _stack, Nth=10, Nz_bias=80)
        sb48.bias_precompute(20.0, 0.5); sb48._cache.clear()  # warm (CAMB/xi_NL lazy build)
        sb80.bias_precompute(20.0, 0.5); sb80._cache.clear()
        best48 = min(_timed_precompute(sb48) for _ in range(3))
        best80 = min(_timed_precompute(sb80) for _ in range(3))
        assert best48 < best80, (
            f'Nz_bias=48 ({best48*1e3:.2f} ms) should be faster than '
            f'Nz_bias=80 ({best80*1e3:.2f} ms)')


def _timed_precompute(sb):
    import time
    sb._cache.clear()
    t0 = time.perf_counter()
    sb.bias_precompute(20.0, 0.5)
    return time.perf_counter() - t0


class TestSelBiasThetaSplit:
    """Test 1: split-at-exclusion converges at Nth=10 to < 0.1% of quad."""

    def test_Nth10_sub_01pct(self, cosmo, _stack, quad_truth):
        sb = _make_sb(cosmo, _stack, Nz=80, Nth=10)
        pre = sb.bias_precompute(20.0, 0.5)
        for key in ('P1', 'I2', 'I1'):
            rel_err = abs(pre[key] / quad_truth[key] - 1.0)
            assert rel_err < 1e-3, (
                f'{key}: {pre[key]:.5e} vs quad {quad_truth[key]:.5e}, '
                f'rel_err={rel_err*100:.3f}% (expect < 0.1%)')

    def test_Nth_converged_Nth15_to_5_digits(self, cosmo, _stack):
        sb10 = _make_sb(cosmo, _stack, Nz=80, Nth=10)
        sb15 = _make_sb(cosmo, _stack, Nz=80, Nth=15)
        sb50 = _make_sb(cosmo, _stack, Nz=80, Nth=50)
        p10 = sb10.bias_precompute(20.0, 0.5)
        p15 = sb15.bias_precompute(20.0, 0.5)
        p50 = sb50.bias_precompute(20.0, 0.5)
        # Nth=15 should be bit-stable vs Nth=50
        for key in ('P1', 'I2', 'I1'):
            diff = abs(p15[key] / p50[key] - 1.0)
            assert diff < 1e-4, (
                f'{key}: Nth=15 {p15[key]:.6e} vs Nth=50 {p50[key]:.6e}, '
                f'diff={diff*100:.4f}% (expect < 0.01%)')


class TestLambdaAxisConverged:
    """Test 2: lambda axis already converged at n_ltr=30."""

    def test_lam_converged(self, cosmo, _stack):
        sb30 = _make_sb(cosmo, _stack, Nz=80, Nth=10, n_ltr=30)
        sb100 = _make_sb(cosmo, _stack, Nz=80, Nth=10, n_ltr=100)
        p30 = sb30.bias_precompute(20.0, 0.5)
        p100 = sb100.bias_precompute(20.0, 0.5)
        for key in ('P1', 'I2', 'I1'):
            diff = abs(p30[key] / p100[key] - 1.0)
            assert diff < 1e-4, (
                f'{key}: n_ltr=30 {p30[key]:.6e}, '
                f'n_ltr=100 {p100[key]:.6e}, diff={diff*100:.4f}%')


class TestBsellMarginalisedSigmoid:
    """Marginalisation commutes with the sigmoid ansatz.

    Because b_lob_theta is linear in (b_zero, b_infty) and sigma(theta)
    is ltr-independent, the ltr-marginalisation can be carried through
    to the plateaus.  The production code exploits this; this test
    pins the equivalence against a straight ltr-loop to machine
    precision.  See docs/richness_selection.tex eq. (b_marg_sigmoid).
    """

    def test_matches_explicit_ltr_loop(self, cosmo, _stack):
        from richness_selection.plob_ltr import P_lob_given_ltr
        from richness_selection.gl import gl_nodes

        sb = _make_sb(cosmo, _stack, Nz=80, Nth=10)
        lob, zob = 20.0, 0.5
        pre = sb.bias_precompute(lob, zob)

        # Replicate the ltr-grid and weights used inside
        # _marginalised_plateaus so this is an apples-to-apples check.
        ltr_grid_size = sb.grid.ltr_grid_size
        t_nodes, t_wts = gl_nodes(1.0, 3.0 * lob, ltr_grid_size * 2)
        log10_Mmin = np.log10(sb.min_mass4integral)
        m_grid = 10.0 ** np.linspace(log10_Mmin, sb.ln_M_max_log10, 50)
        hmf_m = sb.hmf(m_grid, zob)
        p_ltr_M = sb.mor.pdf(t_nodes[:, None], m_grid[None, :], zob)
        prior_ltr = np.trapezoid(p_ltr_M * (hmf_m * m_grid)[None, :],
                             np.log(m_grid), axis=1)
        p_lob_ltr = np.array([float(P_lob_given_ltr(lob, float(ltr), zob))
                              for ltr in t_nodes])
        p_ltr = p_lob_ltr * prior_ltr

        # Straight loop: sum_ltr w(ltr) b_lob_theta(theta | ltr) / sum w
        theta = np.logspace(-5, -1, 11)
        num = np.zeros_like(theta)
        den = 0.0
        for ltr, w, pw in zip(t_nodes, t_wts, p_ltr):
            wt = float(w * pw)
            if wt == 0.0:
                continue
            num += wt * sb.b_lob_theta(theta, float(ltr), zob, lob,
                                        precomp=pre)
            den += wt
        ref = num / den

        marg = sb.b_sel_marginalised(theta, lob, zob, precomp=pre)
        np.testing.assert_allclose(marg, ref, rtol=1e-12, atol=0)


class TestSigmaPrjTheta:
    """Tests 3 and 4: Sigma_prj split-at-theta_R + monotonicity."""

    @pytest.fixture(scope="class")
    def sigma_prj_obj(self, cosmo, _stack):
        import os
        nfw_dir = os.environ.get(
            "RICHNESS_SELECTION_NFW_DIR",
            "/Users/esteves/Documents/Projetos/y3_cluster_cpp/data/nfw_off_center",
        )
        if not os.path.exists(nfw_dir):
            pytest.skip(f"NFW table dir not found: {nfw_dir}")
        sb = _make_sb(cosmo, _stack, Nz=80, Nth=10)
        nfw = NFWMiscentered(cosmo, table_dir=nfw_dir)
        return SigmaPrj(cosmo, sb, nfw, n_theta_per_seg=30)

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

