"""Tests for SurveyArea and its wiring into DeltaSigmaPrj's z-integral.

Pins:
1. Default (kind="unity") reproduces DeltaSigmaPrj's historical
   Omega(z)-free behaviour bit-for-bit.
2. kind="constant" is a pure overall rescaling of rnd/cl/total by the
   same factor (it multiplies the shared outer_weight before rnd/cl
   channels diverge).
3. kind="polynomial" (des/sdss) produces finite, non-trivial output that
   differs from the unity case.
4. Invalid kind/model raise ValueError at construction time.
"""
from __future__ import annotations
import numpy as np
import pytest

from richness_selection import DeltaSigmaPrj, SurveyArea


REF_R = np.array([0.3, 1.0, 3.0, 10.0])


def test_unity_is_default():
    sa = SurveyArea()
    assert sa.kind == "unity"
    z = np.array([0.1, 0.3, 0.5, 0.9])
    np.testing.assert_array_equal(sa(z), np.ones_like(z))


def test_constant_broadcasts():
    sa = SurveyArea(kind="constant", value=2.5)
    z = np.array([0.1, 0.3, 0.5])
    np.testing.assert_allclose(sa(z), np.full_like(z, 2.5))


def test_polynomial_matches_survey_module():
    from richness_selection.selection_function.survey import omega_z_des, omega_z_sdss
    z = np.linspace(0.1, 0.9, 9)
    sa_des = SurveyArea(kind="polynomial", model="des")
    sa_sdss = SurveyArea(kind="polynomial", model="sdss")
    np.testing.assert_allclose(sa_des(z), omega_z_des(z))
    np.testing.assert_allclose(sa_sdss(z), omega_z_sdss(z))


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        SurveyArea(kind="bogus")


def test_invalid_polynomial_model_raises():
    with pytest.raises(ValueError):
        SurveyArea(kind="polynomial", model="bogus")


# ---------------------------------------------------------------------
# Wiring into DeltaSigmaPrj
# ---------------------------------------------------------------------

def test_default_survey_area_matches_no_arg(cosmo, sel_bias, nfw):
    """Passing survey_area=SurveyArea() explicitly must be bit-identical
    to omitting the argument entirely -- confirms the default wiring
    changes nothing for existing callers."""
    dsp_implicit = DeltaSigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30)
    dsp_explicit = DeltaSigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30,
                                 survey_area=SurveyArea())
    dec_implicit = dsp_implicit(REF_R, 20.0, 0.5, return_decomposition=True)
    dec_explicit = dsp_explicit(REF_R, 20.0, 0.5, return_decomposition=True)
    np.testing.assert_array_equal(dec_implicit["total"], dec_explicit["total"])
    np.testing.assert_array_equal(dec_implicit["rnd"], dec_explicit["rnd"])
    np.testing.assert_array_equal(dec_implicit["cl"], dec_explicit["cl"])


def test_constant_survey_area_rescales_linearly(cosmo, sel_bias, nfw):
    """A constant Omega(z) multiplies outer_weight uniformly across z, so
    it must rescale rnd, cl, and total by exactly that same factor."""
    dsp_unity = DeltaSigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30)
    factor = 3.0
    dsp_scaled = DeltaSigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30,
                              survey_area=SurveyArea(kind="constant",
                                                     value=factor))
    dec_unity = dsp_unity(REF_R, 20.0, 0.5, return_decomposition=True)
    dec_scaled = dsp_scaled(REF_R, 20.0, 0.5, return_decomposition=True)
    for key in ("rnd", "cl", "total"):
        np.testing.assert_allclose(dec_scaled[key], factor * dec_unity[key],
                                   rtol=1e-10)


def test_polynomial_survey_area_differs_from_unity(cosmo, sel_bias, nfw):
    dsp_unity = DeltaSigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30)
    dsp_des = DeltaSigmaPrj(cosmo, sel_bias, nfw, n_theta_per_seg=30,
                            survey_area=SurveyArea(kind="polynomial",
                                                   model="des"))
    dec_unity = dsp_unity(REF_R, 20.0, 0.5, return_decomposition=True)
    dec_des = dsp_des(REF_R, 20.0, 0.5, return_decomposition=True)
    assert np.all(np.isfinite(dec_des["total"]))
    assert not np.allclose(dec_des["total"], dec_unity["total"])
    # rnd/cl need not scale by the same factor under a z-varying Omega(z)
    # (unlike the constant case), since rnd and cl weight different
    # z-ranges differently -- just check the ratio actually moved.
    ratio_unity = dec_unity["rnd"] / dec_unity["cl"]
    ratio_des = dec_des["rnd"] / dec_des["cl"]
    assert not np.allclose(ratio_unity, ratio_des)
