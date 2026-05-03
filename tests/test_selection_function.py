"""Tests for the richness selection function package."""
from __future__ import annotations
import numpy as np
import pytest
from scipy.integrate import nquad

from richness_selection import MOR, LogNormalMOR
from richness_selection.selection_function import K_i, K_j, S_i, S_ij, N_ij
from richness_selection.plob_ltr import P_lob_given_ltr


# ---- fixtures -----------------------------------------------------------

@pytest.fixture(scope="module")
def hod():
    return MOR()


@pytest.fixture(scope="module")
def logn():
    return LogNormalMOR()


# ---- K_i sanity ---------------------------------------------------------

class TestKi:
    """K_i(ltr, z) is a probability over lob bins."""

    def test_full_range_is_one(self):
        # Full (-inf, +inf) must give ~ 1 by construction.
        # (The Costanzi EMG kernel is not normalised on [0, +inf) alone
        # because the Gaussian core can leak into negative lob for small
        # ltr -- that's physical and documented in the tech note.)
        for z in (0.3, 0.5, 0.7):
            for lt in (10.0, 30.0, 80.0):
                val = K_i(lt, z, -1e4, 1e4)
                assert abs(val - 1.0) < 1e-6, (
                    f'K_i(ltr={lt}, z={z}, [-1e4, 1e4]) = {val} != 1')

    def test_disjoint_bins_sum_to_one(self):
        edges = [-1e4, 0.0, 10.0, 25.0, 50.0, 120.0, 1e4]
        for z in (0.3, 0.5):
            for lt in (20.0, 50.0):
                tot = sum(K_i(lt, z, a, b)
                          for a, b in zip(edges[:-1], edges[1:]))
                assert abs(tot - 1.0) < 1e-6, (
                    f'ltr={lt}, z={z}: sum={tot}')

    def test_empty_bin_is_zero(self):
        assert abs(K_i(20.0, 0.5, 1000.0, 1000.0 + 1e-8)) < 1e-10

    def test_monotone_in_lam_max(self):
        # K_i([0, x]) is a CDF: increasing in x at fixed ltr
        xs = np.array([10., 20., 40., 80., 200.])
        z = 0.5
        ltr = 40.0
        vals = np.array([K_i(ltr, z, 0.0, x) for x in xs])
        assert np.all(np.diff(vals) >= -1e-10)


# ---- K_j (Gaussian CDF difference) --------------------------------------

class TestKj:
    """K_j reduces to the standard normal CDF difference."""

    def test_symmetric_bin(self):
        # +/- 1 sigma: Phi(1) - Phi(-1) = 0.6827
        val = K_j(0.5, 0.5 - 0.02, 0.5 + 0.02, 0.02)
        assert abs(val - 0.6827) < 1e-3

    def test_two_sigma(self):
        val = K_j(0.5, 0.5 - 0.04, 0.5 + 0.04, 0.02)
        assert abs(val - 0.9545) < 1e-3


# ---- S_i against nquad reference -----------------------------------------

def _S_i_nquad(M, z, lam_min, lam_max, mor, rtol=1e-5):
    def f(ltr, lob):
        return (P_lob_given_ltr(lob, ltr, z)
                * mor.pdf(np.asarray(ltr), M, z))
    mu = float(mor.ltr_mean(M, z))
    sig = float(mor.ltr_sigma(M, z))
    ltr_lo = max(0.0, mu - 10.0 * sig)
    ltr_hi = mu + 10.0 * sig
    val, _ = nquad(f, [[ltr_lo, ltr_hi], [lam_min, lam_max]],
                    opts=[{'epsrel': rtol, 'limit': 200},
                          {'epsrel': rtol, 'limit': 200}])
    return float(val)


class TestSiVsQuad:
    """S_i (closed-form + GL) vs nquad reference at the peak of the
    distribution (where both integrators give reliable answers)."""

    @pytest.mark.parametrize('M,z,bin_lam', [
        (1e14, 0.30, (20., 30.)),
        (1e14, 0.50, (20., 30.)),
        (3e14, 0.30, (30., 45.)),
        (3e14, 0.50, (45., 60.)),
        (3e14, 0.70, (30., 45.)),
        (1e15, 0.50, (45., 60.)),
    ])
    def test_hod(self, hod, M, z, bin_lam):
        ours = S_i(M, z, bin_lam[0], bin_lam[1], hod, N_q=64)
        quad = _S_i_nquad(M, z, bin_lam[0], bin_lam[1], hod)
        # Only enforce 1e-3 where S_i > 1e-4 (tails are unreliable
        # for nquad itself).
        if quad > 1e-4:
            rel = abs(ours / quad - 1.0)
            assert rel < 1e-3, (
                f'HOD M={M:.2e} z={z} bin={bin_lam}: '
                f'ours={ours:.4e} quad={quad:.4e} rel={rel:.2e}')

    @pytest.mark.parametrize('M,z,bin_lam', [
        (1e14, 0.30, (20., 30.)),
        (1e14, 0.50, (30., 45.)),
        (3e14, 0.30, (30., 45.)),
        (3e14, 0.50, (45., 60.)),
    ])
    def test_lognormal(self, logn, M, z, bin_lam):
        ours = S_i(M, z, bin_lam[0], bin_lam[1], logn, N_q=64)
        quad = _S_i_nquad(M, z, bin_lam[0], bin_lam[1], logn)
        if quad > 1e-4:
            rel = abs(ours / quad - 1.0)
            assert rel < 1e-2, (
                f'LogN M={M:.2e} z={z} bin={bin_lam}: '
                f'ours={ours:.4e} quad={quad:.4e} rel={rel:.2e}')


# ---- S_i internal consistency: disjoint bins sum to ~1 -------------------

class TestSiSumRule:
    """sum_i S_i(M, z) over disjoint lob bins covering [0, inf] ~ 1
    (the probability that the halo lands in _some_ bin is 1)."""

    @pytest.mark.parametrize('M,z', [
        (5e13, 0.3), (1e14, 0.3), (3e14, 0.5), (1e15, 0.7),
    ])
    def test_hod(self, hod, M, z):
        # Include a negative-lob bin to capture the Gaussian tail that
        # leaks below zero for small ltr (Costanzi kernel is normalised
        # on R, not [0, inf]).
        edges = [-1e4, 0., 5., 15., 30., 60., 150., 1e4]
        tot = sum(S_i(M, z, a, b, hod, N_q=64)
                   for a, b in zip(edges[:-1], edges[1:]))
        assert abs(tot - 1.0) < 1e-3, f'HOD M={M} z={z}: sum={tot}'

    @pytest.mark.parametrize('M,z', [
        (5e13, 0.3), (1e14, 0.3), (3e14, 0.5),
    ])
    def test_lognormal(self, logn, M, z):
        edges = [-1e4, 0., 5., 15., 30., 60., 150., 1e4]
        tot = sum(S_i(M, z, a, b, logn, N_q=64)
                   for a, b in zip(edges[:-1], edges[1:]))
        assert abs(tot - 1.0) < 2e-3, f'LogN M={M} z={z}: sum={tot}'


# ---- S_ij equals S_i * K_j -----------------------------------------------

class TestSij:
    def test_product(self, hod):
        M = 1e14
        z = 0.4
        bin_lam = (20., 30.)
        bin_z = (0.3, 0.5)
        sigma_z = 0.02
        Si = S_i(M, z, *bin_lam, hod)
        Kj = K_j(z, *bin_z, sigma_z)
        Sij = S_ij(M, z, bin_lam, bin_z, sigma_z, hod)
        assert abs(Sij - Si * Kj) < 1e-12


# ---- N_ij basic sanity ---------------------------------------------------

class TestNij:
    def test_positive_and_sensible(self):
        from richness_selection import Cosmology, PkGrid, HMF
        from richness_selection.sigma_m import SigmaM
        cosmo = Cosmology()
        pk = PkGrid(cosmo)
        sm = SigmaM(pk)
        hmf = HMF(sm)
        mor = MOR()
        N = N_ij(bin_lam=(20., 40.), bin_z=(0.3, 0.5),
                 cosmo=cosmo, hmf=hmf, mor=mor, sigma_z=0.02,
                 N_M=32, N_z=24, N_q=32)
        # ~few thousand per sr for DES-like richness / redshift is order of
        # magnitude correct (Y3 ~6k-7k in [20-30] at [0.3, 0.5]).
        assert N > 100.0 and N < 1e5
