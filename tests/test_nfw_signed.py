"""Tests for the signed miscentered DeltaSigma reconstruction in
``NFWMiscentered`` (``_build_signed_dsigma_spline``).

Physics pins:
1. UNITY -- mass conservation: miscentering only redistributes the
   halo's projected mass azimuthally, so the enclosed 2D mass around
   the cluster center converges to the centered one,
       bar-Sigma_mis(<R | R_mis) / bar-Sigma_cen(<R)  ->  1
   for R >> R_mis.
2. UNITY -- offset forgetting: the excess itself forgets the offset,
       DeltaSigma_mis(R | R_mis) / DeltaSigma_cen(R)  ->  1
   for R >> R_mis.
3. Centered limit (R_mis -> 0) matches the analytic Wright & Brainerd
   excess.
4. The negative branch exists: DeltaSigma_mis(R < R_mis) < 0 (the ring
   average rises toward R_mis, so the interior mean lies below it) --
   the branch the shipped ln-table destroyed.
5. Exact zero crossing consistency: DeltaSigma_mis integrates the same
   Sigma_mis the Sigma spline provides (excess-of-Sigma identity on a
   fine grid).
"""
from __future__ import annotations
import numpy as np
import pytest

from richness_selection.nfw import wright_brainerd_g


M_TEST = 1e14
Z_TEST = 0.5


def _dimless(nfw, lnxmis, lnx):
    """Signed dimensionless excess gs(x | x_mis) from the spline.

    Scalar inputs give a python float; array lnx gives a 1-D array.
    """
    out = nfw._dsig_spl(np.atleast_1d(lnxmis), np.atleast_1d(lnx))
    return float(out[0, 0]) if out.size == 1 else out.ravel()


def test_unity_mass_conservation(nfw):
    """bar-Sigma_mis(<R)/bar-Sigma_cen(<R) -> 1 for R >> R_mis.

    bar-Sigma = DeltaSigma + Sigma, both from the module's splines.
    """
    x_mis = 5.0
    for x_ratio, tol in ((10.0, 2e-2), (30.0, 5e-3), (80.0, 2e-3)):
        x = x_ratio * x_mis
        lnx = np.log(x)
        gs_mis = _dimless(nfw, np.log(x_mis), lnx)
        f_mis = float(np.exp(nfw._spl(np.log(x_mis), lnx))[0, 0])
        gs_cen = _dimless(nfw, nfw._lnxmis_lo, lnx)
        f_cen = float(np.exp(nfw._spl(nfw._lnxmis_lo, lnx))[0, 0])
        ratio = (gs_mis + f_mis) / (gs_cen + f_cen)
        assert abs(ratio - 1.0) < tol, (
            f"mass conservation: bar-Sigma ratio {ratio:.5f} != 1 "
            f"at x/x_mis = {x_ratio}")


def test_unity_offset_forgetting(nfw):
    """DeltaSigma_mis(R | R_mis)/DeltaSigma_cen(R) -> 1 for R >> R_mis."""
    x_mis = 5.0
    for x_ratio, tol in ((10.0, 5e-2), (30.0, 1e-2), (80.0, 5e-3)):
        x = x_ratio * x_mis
        lnx = np.log(x)
        gs_mis = _dimless(nfw, np.log(x_mis), lnx)
        gs_cen = _dimless(nfw, nfw._lnxmis_lo, lnx)
        ratio = gs_mis / gs_cen
        assert abs(ratio - 1.0) < tol, (
            f"offset forgetting: DeltaSigma ratio {ratio:.5f} != 1 "
            f"at x/x_mis = {x_ratio}")


def test_centered_limit_matches_wright_brainerd(nfw):
    """R_mis -> 0 row equals the analytic W&B excess, g_WB(x)/2 in the
    2 r_s rho_eff normalisation, over the working x range."""
    x = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 200.0])
    gs = _dimless(nfw, nfw._lnxmis_lo, np.log(x)).ravel()
    ref = 0.5 * wright_brainerd_g(x)
    np.testing.assert_allclose(gs, ref, rtol=2e-2)


def test_negative_branch_exists(nfw):
    """DeltaSigma_mis(R | R_mis) < 0 for R somewhat below R_mis."""
    x_mis = 10.0
    for x in (3.0, 6.0, 9.0):
        gs = _dimless(nfw, np.log(x_mis), np.log(x))
        assert gs < 0.0, (
            f"negative branch missing: gs({x} | x_mis={x_mis}) = {gs:+.3e}")


def test_excess_identity_on_fine_grid(nfw):
    """gs(x | x_mis) == (2/x^2) int x' f dx' - f, recomputed here with
    an independent (finer, Simpson-free trapezoid) cumulative."""
    x_mis = 2.0
    lnx_f = np.linspace(nfw._lnx_lo, nfw._lnx_hi, 12000)
    f = np.exp(nfw._spl(np.log(x_mis), lnx_f)).ravel()
    x_f = np.exp(lnx_f)
    integ = f * x_f ** 2
    K = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(lnx_f))])
    K += f[0] * x_f[0] ** 2 / 2.0
    gs_ref = 2.0 * K / x_f ** 2 - f
    probe = np.array([0.5, 1.9, 2.1, 5.0, 40.0])
    gs_spl = _dimless(nfw, np.log(x_mis), np.log(probe)).ravel()
    gs_ref_p = np.interp(np.log(probe), lnx_f, gs_ref)
    scale = np.maximum(np.abs(gs_ref_p), 1e-3)
    assert (np.abs(gs_spl - gs_ref_p) / scale < 5e-3).all()


def test_delta_sigma_grid_signed_units(nfw):
    """delta_sigma_grid returns signed values with the same prefactor
    convention as sigma_grid (W&B factor-2 check at R_mis -> 0)."""
    R = np.array([0.5, 2.0])
    ds = nfw.delta_sigma_grid(R, np.array([1e-9]), M_TEST, Z_TEST)[0]
    rs, rho_eff = nfw._rs_and_rhos(M_TEST, Z_TEST)
    ref = 2.0 * rs * rho_eff * 1.0e-12 * 0.5 * wright_brainerd_g(R / rs)
    np.testing.assert_allclose(ds, ref, rtol=2e-2)
    # signed: a strongly offset halo gives negative excess at small R
    ds_neg = nfw.delta_sigma_grid(
        np.array([0.1 * rs]), np.array([5.0 * rs]), M_TEST, Z_TEST)[0]
    assert ds_neg[0] < 0.0
