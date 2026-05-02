"""End-to-end smoke: every module imports and evaluates at the notebook
reference point (lob=20, zob=0.5).
"""
import numpy as np


def test_cosmology_chi(cosmo):
    assert abs(cosmo.chi(0.5) - 1328.513) / 1328.513 < 1e-3


def test_pk_shape(pk):
    assert pk.k.shape[0] > 100
    assert pk.z.shape[0] >= 4
    assert pk.P.shape == (pk.z.size, pk.k.size)


def test_sigma_m_sanity(sigma_m):
    s = sigma_m(1e14, 0.0)
    assert 0.1 < s < 1.5


def test_hmf_positive(hmf):
    n = hmf(1e14, 0.0)
    assert n > 0 and np.isfinite(n)


def test_bias_monotone(bias):
    b_low = bias(1e13, 0.0)
    b_hi = bias(1e15, 0.0)
    assert b_hi > b_low > 0


def test_mor_closed_form(mor):
    big = mor.lambda_mean_below(1e14, 0.5, 1e6)
    full = np.exp(mor.mu_ln_lambda(1e14, 0.5) + 0.5 * mor.sigma ** 2)
    assert abs(big / full - 1.0) < 1e-8


def test_sel_bias_precompute_invariance(sel_bias):
    pre = sel_bias.bias_precompute(20.0, 0.5)
    out_a = sel_bias.bias_from_precomp(pre, 18.0)
    out_b = sel_bias.bias_from_precomp(pre, 18.0)
    assert out_a["b_large"] == out_b["b_large"]
    assert out_a["b_small"] == out_b["b_small"]


def test_b_halo_reasonable(sel_bias):
    b = sel_bias.b_halo(20.0, 0.5)
    assert 1.0 < b < 10.0


def test_sigma_prj_one_R(sigma_prj):
    R = np.array([1.0])
    s = sigma_prj(R, 20.0, 0.5)
    assert s.shape == (1,)
    assert np.isfinite(s[0])
