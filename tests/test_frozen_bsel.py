"""Tests for the frozen-physics selection bias (frozen_bsel.py).

Anchored to the equations of docs/richness_selection_frozen.tex:
eq. (Ilam_zero), the total-donation moment identity, eq. (budget),
and the dataclass-vs-legacy-dict path agreement.
"""
import numpy as np
import pytest

from richness_selection import FrozenSelBias, FrozenOperators
from richness_selection.geometry import R_lambda
from richness_selection.gl import gl_nodes

LOB, ZOB = 20.0, 0.5


@pytest.fixture(scope="session")
def frozen(cosmo, pk, hmf, bias, mor, xi_nl):
    return FrozenSelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi_nl)


class TestIdentities:
    def test_Ilam_zero(self, frozen):
        """eq. (Ilam_zero): I_lambda(0) = <Dlam>_b(zob), exact."""
        I = frozen.I_lambda(LOB, ZOB)
        assert float(I(0.0)[0]) == pytest.approx(
            frozen.Dlam_b(LOB, ZOB), rel=1e-12)

    def test_total_donation_moment(self, frozen):
        """A_ss + A_ls = Rex^2 <Dlam>_b(zob) / 2 (eq. moments +
        the total-donation identity)."""
        Rex = R_lambda(LOB) * (1.0 + ZOB)
        I = frozen.I_lambda(LOB, ZOB)
        x, w = gl_nodes(0.0, 2.0, 400)
        A_tot = Rex ** 2 * float(np.sum(w * x * I(x)))
        assert A_tot == pytest.approx(
            Rex ** 2 * frozen.Dlam_b(LOB, ZOB) / 2.0, rel=1e-6)

    def test_budget_closure(self, frozen):
        """eq. (budget): lob - ltr = rnd + b^ls ls + b^ss ss, exact
        by construction of eq. (bss)."""
        pre = frozen.bias_precompute(LOB, ZOB)
        for ltr in (18.0, 15.0, 10.0):
            pr = frozen.bias_from_precomp(pre, ltr)
            budget = (pre["P1"] + pr["b_infty"] * pre["I1"]
                      + pr["b_zero"] * (pre["I2"] - pre["I1"]))
            assert budget == pytest.approx(LOB - ltr, abs=1e-9)

    def test_costanzi_delta_rnd_convention(self, frozen):
        """eq. (bls), fiducial: delta_prj denominator is
        <Dprj>_halo = <Dprj>_rnd + b_halo (ss + ls)."""
        pre = frozen.bias_precompute(LOB, ZOB)
        assert pre["Delta_RND"] == pytest.approx(
            pre["P1"] + pre["b_eff"] * pre["I2"], rel=1e-14)
        pr = frozen.bias_from_precomp(pre, 15.0)
        assert pr["delta_prj"] == pytest.approx(
            (LOB - 15.0) / pre["Delta_RND"] - 1.0, rel=1e-12)


class TestDataclassPaths:
    def test_operators_vs_precomp(self, frozen):
        ops = frozen.operators(LOB, ZOB)
        pre = frozen.bias_precompute(LOB, ZOB)
        assert isinstance(ops, FrozenOperators)
        assert pre["P1"] == ops.Dprj_rnd
        assert pre["I1"] == ops.tDprj_ls
        assert pre["I2"] == pytest.approx(ops.tDprj_ss + ops.tDprj_ls,
                                          rel=1e-14)
        assert pre["denom"] == ops.tDprj_ss

    def test_marginalised_bias_equals_legacy(self, frozen):
        """The MarginalisedBias hook == b_sel_marginalised."""
        theta = np.geomspace(1e-5, 1e-2, 40)
        mb = frozen.marginalised_bias(LOB, ZOB)
        np.testing.assert_allclose(
            mb(theta), frozen.b_sel_marginalised(theta, LOB, ZOB),
            rtol=1e-12)

    def test_plateaus_vectors(self, frozen):
        p = frozen.plateaus(LOB, ZOB)
        assert p.ltr.shape == p.w_ltr.shape == p.b_rm_ss_ltr_vec.shape
        assert p.w_ltr.sum() == pytest.approx(1.0, rel=1e-12)
        assert p.b_rm_ss == pytest.approx(
            float(np.sum(p.w_ltr * p.b_rm_ss_ltr_vec)), rel=1e-12)
        # per-ltr vector agrees with the scalar legacy closure
        pre = frozen.bias_precompute(LOB, ZOB)
        i = len(p.ltr) // 2
        pr = frozen.bias_from_precomp(pre, float(p.ltr[i]))
        assert p.b_rm_ss_ltr_vec[i] == pytest.approx(pr["b_zero"], rel=1e-12)
        assert p.b_rm_ls_ltr_vec[i] == pytest.approx(pr["b_infty"], rel=1e-12)


class TestOperatorsRegression:
    """Frozen operators at the reference point (values validated
    against scipy.quad in validations/frozen_bsel_validation.py)."""

    def test_reference_point(self, frozen):
        ops = frozen.operators(LOB, ZOB)
        assert ops.tDprj_ls == pytest.approx(0.22595848441, rel=1e-6)
        assert ops.tDprj_ss + ops.tDprj_ls == pytest.approx(
            0.33475094192, rel=1e-6)
        # carve-out (fiducial) random channel
        assert ops.Dprj_rnd == pytest.approx(1.8014304871, rel=1e-6)

    def test_production_marginalised_api(self, sel_bias):
        """The shared plateau API also works on the production class."""
        p = sel_bias.plateaus(LOB, ZOB)
        theta = np.array([1e-4, 1e-3])
        np.testing.assert_allclose(
            sel_bias.b_rm(theta, LOB, ZOB),
            sel_bias.b_sel_marginalised(theta, LOB, ZOB), rtol=1e-12)
        assert np.isfinite(p.b_rm_ss) and np.isfinite(p.b_rm_ls)


class TestFrozenDeltaSigmaPrj:
    """Sec. 'Extension' of the frozen note: FrozenDeltaSigmaPrj vs
    production DeltaSigmaPrj (same theta grid, same NFW lookup, same
    b_rm; residual = the (z,M) factorisation only)."""

    def test_vs_production(self, cosmo, sel_bias, nfw):
        from richness_selection import DeltaSigmaPrj, FrozenDeltaSigmaPrj
        R = np.array([0.3, 1.0, 3.0, 10.0])
        dsp = DeltaSigmaPrj(cosmo, sel_bias, nfw)
        fdsp = FrozenDeltaSigmaPrj(cosmo, sel_bias, nfw)
        d_p = dsp(R, LOB, ZOB, return_decomposition=True)
        d_f = fdsp(R, LOB, ZOB, return_decomposition=True)
        # rnd channel is exact (tilde-n hoist commutes past z-free kernel)
        np.testing.assert_allclose(d_f["rnd"], d_p["rnd"], rtol=1e-10)
        # cl channel carries only the drift-shape residual (eq. nb_drift)
        np.testing.assert_allclose(d_f["cl"], d_p["cl"], rtol=5e-4)
