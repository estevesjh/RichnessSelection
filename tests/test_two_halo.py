"""Tests for the traditional two-halo term (two_halo.TwoHalo).

Pins:
1. Cylinder projection against the closed form for a power-law xi:
   xi = (r/r0)^-2  =>  C_xi(R) = pi r0^2 / R  (L -> infinity).
2. DeltaSigma algebra in the same power-law regime: Sigma ~ 1/R gives
   Sigmabar(<R) = 2 Sigma(R), hence DeltaSigma = Sigma exactly.
3. b_sel_ls is the large-scale plateau of the marginalised bias
   (selection-weighted mean cluster bias, Tinker-level amplitude).
4. The rnd piece is identically zero -- the two-halo term is defined
   as the excess over the mean field.
5. Exclusion only affects small R.

None of these need the NFW lookup table: TwoHalo has no NFW dependence
by construction.
"""
from __future__ import annotations
import numpy as np
import pytest

from richness_selection import TwoHalo


class _PowerLawXi:
    """xi(r) = (r / r0)^-2, the closed-form projection test profile."""

    def __init__(self, r0=3.0):
        self.r0 = float(r0)

    def __call__(self, r, z):
        r = np.asarray(r, dtype=float)
        return (np.clip(r, 1e-30, None) / self.r0) ** -2.0


@pytest.fixture(scope="module")
def th(cosmo, sel_bias):
    return TwoHalo(cosmo, sel_bias)


@pytest.fixture(scope="module")
def th_power(cosmo, sel_bias):
    t = TwoHalo(cosmo, sel_bias, L_los=2000.0, n_los=200001,
                exclusion=False)
    t.xi_NL = _PowerLawXi(r0=3.0)
    return t


def test_cylinder_projection_power_law(th_power):
    R = np.array([1.0, 2.0, 5.0, 10.0])
    C = th_power.C_xi(R, 20.0, 0.5)
    exact = np.pi * th_power.xi_NL.r0 ** 2 / R
    np.testing.assert_allclose(C, exact, rtol=5e-3)


def test_delta_sigma_equals_sigma_power_law(th_power):
    """Sigma ~ 1/R  =>  Sigmabar = 2 Sigma  =>  DeltaSigma = Sigma."""
    R = np.array([2.0, 5.0, 10.0])
    S = th_power.sigma(R, 20.0, 0.5)
    DS = th_power.delta_sigma(R, 20.0, 0.5, n_grid=400, R_min_grid=0.005)
    np.testing.assert_allclose(DS, S, rtol=2e-2)


def test_b_sel_ls_plateau(th, sel_bias, cosmo):
    """b_sel_ls equals the marginalised bias at large theta and is
    Tinker-amplitude (full cluster bias, not a ratio ~ 1)."""
    b_ls = th.b_sel_ls(20.0, 0.5)
    pre = sel_bias.bias_precompute(20.0, 0.5)
    fn = sel_bias.marginalised_bias(20.0, 0.5, precomp=pre)
    chi_o = float(cosmo.chi(0.5))
    plateau = float(fn(np.array([50.0 / chi_o]))[0])
    assert abs(b_ls - plateau) / plateau < 1e-3
    assert b_ls > 2.0  # full bias amplitude, not a ~1 ratio


def test_rnd_identically_zero(th):
    dec = th(np.array([1.0, 5.0, 10.0]), 20.0, 0.5,
             return_decomposition=True)
    assert np.all(dec["rnd"] == 0.0)
    np.testing.assert_allclose(dec["cl"], dec["total"], rtol=0, atol=0)


def test_exclusion_only_small_R(cosmo, sel_bias):
    t_ex = TwoHalo(cosmo, sel_bias, exclusion=True)
    t_no = TwoHalo(cosmo, sel_bias, exclusion=False)
    R = np.array([0.5, 1.0, 10.0, 20.0])
    S_ex = t_ex.sigma(R, 20.0, 0.5)
    S_no = t_no.sigma(R, 20.0, 0.5)
    # carved core at R <= 1 cMpc/h
    assert S_ex[0] < 0.9 * S_no[0]
    # untouched at 2h scales
    np.testing.assert_allclose(S_ex[2:], S_no[2:], rtol=1e-4)


def test_sigma_positive_and_decreasing(th):
    R = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 20.0])
    S = th.sigma(R, 20.0, 0.5)
    assert (S > 0).all()
    assert (np.diff(S) < 0).all()
