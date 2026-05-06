"""Regression tests for the refactored SigmaPrj.

Pins:
1. Default ``__call__`` returns the cl+LSS correlation-excess piece
   (not the full ``1 + b xi`` total).
2. RND+CL algebra identity in the decomposition (total == rnd + cl) to
   machine precision.
3. n_theta_per_seg convergence plateau (sub-0.2% agreement between
   default 30 and a gold 120-per-segment run) on the full ``total``.
4. Full ``total`` at default (n_per_seg=30) matches the legacy
   scipy.quad reference to 0.3%.
5. R_max_cMpch hook changes the theta grid upper bound.
"""
from __future__ import annotations
import numpy as np
import pytest

from richness_selection import SigmaPrj


REF_R = np.array([0.3, 1.0, 3.0, 10.0])
# scipy.quad reference from validations/sigma_prj_diag_results.md
# ("ref_code_conv": xi at zob, R_mis = theta * D_A(zob); matches the
# refactor's conventions).  Applies to the FULL total (1 + b xi).
REF_QUAD_TOTAL = np.array([3.6756e14, 3.1926e14, 2.6149e14, 2.2048e14])


@pytest.fixture(scope="module")
def sp(cosmo, sel_bias, nfw):
    return SigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30)


def test_default_returns_cl(sp):
    """Default call returns the cl+LSS piece; decomposition sums match."""
    cl_default = sp(REF_R, 20.0, 0.5)
    dec = sp(REF_R, 20.0, 0.5, return_decomposition=True)
    np.testing.assert_allclose(cl_default, dec["cl"], rtol=0, atol=0)
    # rnd + cl = total
    np.testing.assert_allclose(dec["rnd"] + dec["cl"], dec["total"],
                               rtol=0, atol=0)
    # For lob=20 at zob=0.5 the cl piece is a substantial but non-dominant
    # fraction of the total -- positive and smaller than total.
    assert (dec["cl"] > 0).all()
    assert (dec["cl"] < dec["total"]).all()


def test_total_vs_scipy_quad(sp):
    """total (from return_decomposition) must match the legacy quad to 0.3%."""
    dec = sp(REF_R, 20.0, 0.5, return_decomposition=True)
    rel = np.abs(dec["total"] - REF_QUAD_TOTAL) / REF_QUAD_TOTAL
    assert rel.max() < 3e-3, (
        f"Max relative error {rel.max():.3%} exceeds 0.3% target.\n"
        f"total = {dec['total']}\nref   = {REF_QUAD_TOTAL}\nrel   = {rel}"
    )


def test_n_per_seg_convergence(cosmo, sel_bias, nfw):
    sp30 = SigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30)
    sp120 = SigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=120)
    v30 = sp30(REF_R, 20.0, 0.5, return_decomposition=True)["total"]
    v120 = sp120(REF_R, 20.0, 0.5, return_decomposition=True)["total"]
    rel = np.abs(v30 - v120) / v120
    assert rel.max() < 2e-3, (
        f"n_per_seg=30 vs 120 disagree by {rel.max():.3%}; "
        f"default grid is not converged."
    )


def test_R_max_cMpch_controls_theta_grid(cosmo, sel_bias, nfw):
    """The R_max_cMpch kwarg sets the upper bound of the theta grid."""
    sp30 = SigmaPrj(cosmo, sel_bias, nfw, R_max_cMpch=30.0)
    sp60 = SigmaPrj(cosmo, sel_bias, nfw, R_max_cMpch=60.0)
    d30 = sp30(REF_R, 20.0, 0.5, return_decomposition=True)
    d60 = sp60(REF_R, 20.0, 0.5, return_decomposition=True)
    # Breakpoint upper bound scales with R_max_cMpch.
    assert d60["theta_info"]["theta_max"] > d30["theta_info"]["theta_max"]
    # The cl piece barely changes (xi_NL falls fast); the rnd piece grows.
    cl_rel = np.abs(d60["cl"] - d30["cl"]).max() / np.abs(d30["cl"]).max()
    rnd_rel = (d60["rnd"] - d30["rnd"]).max() / d30["rnd"].max()
    assert cl_rel < 1e-2, f"cl should be ~converged at R_max=30 (got {cl_rel:.3%})"
    assert rnd_rel > 0.05, (
        f"rnd should grow when R_max goes 30 -> 60 (got {rnd_rel:.3%})")


def test_theta_grid_breakpoints_include_each_R(sp):
    """Each requested R must appear as a theta_R breakpoint.

    This is the key fix that brought n_per_seg=30 to 0.1% convergence
    on R=3 (previously +1.3% off).
    """
    dec = sp(REF_R, 20.0, 0.5, return_decomposition=True)
    brk = dec["theta_info"]["breakpoints"]
    D_A_o = sp.cosmo.D_A(0.5)
    theta_Rs = REF_R / D_A_o
    for tR in theta_Rs:
        assert np.any(np.abs(brk - tR) / tR < 1e-6), (
            f"theta_R = {tR:.3e} missing from breakpoints {brk}"
        )
