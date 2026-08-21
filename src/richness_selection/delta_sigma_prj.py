"""DeltaSigma_prj(R | lob, zob): two-halo excess surface density.

Evaluates the cluster--lensing observable

    < DeltaSigma^prj(R | lob, zob) > =
        int dz dV/dOmega int dM n(M, z)
        * int dtheta sin(theta)
          [1 + b(M, z) b_sel(theta) xi_NL(|Delta r|, zob)]
        * DeltaSigma_mis(R | M, z, R_mis = theta * D_A(zob))

obtained from the ``SigmaPrj`` Eq. 13 kernel by swapping ``Sigma_mis``
for its radial excess ``DeltaSigma_mis = bar-Sigma_mis(<R) -
Sigma_mis(R)``: the excess functional is linear in R only, so it
commutes with the (theta, z, M) integrals, and the only change from
the ``SigmaPrj`` pipeline is the underlying kernel lookup.

The kernel is the *signed* miscentered excess reconstructed by
``NFWMiscentered`` from the positive-definite Sigma table (see
``nfw.py``: the shipped ``log_deltasigma_single`` table stored
``ln g`` and floored the negative branch ``DeltaSigma_mis(R < R_mis)``
to ~0, which inflated the two-halo cl piece by ~1.5x and broke the
uniform-field rnd cancellation).

Defaults
--------

- Default return is the ``cl+LSS`` piece, matching ``SigmaPrj``.  The
  rnd (uniform mean-field) piece does not belong in the two-halo
  observable -- a uniform sheet has DeltaSigma = 0 exactly -- and with
  the signed kernel the computed rnd is numerically consistent with
  zero (finite-aperture residual only).

- theta-upper bound: ``max(R_MAX_CMPCH, R_max_factor * max(R))``.
  Unlike the old (sign-broken) kernel, the signed excess has a
  negative branch extending to all ``R_mis > R``, so the aperture must
  cover the xi_NL support, not just ``theta_R``; the legacy 30 cMpc/h
  floor does that, and the ``R_max_factor * max(R)`` term extends it
  when large radii are requested.  Pass ``R_max_cMpch`` to override.
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology
from .nfw import NFWMiscentered
from .sel_bias import SelBias
from .sigma_prj import SigmaPrj
from .survey_area import SurveyArea
from .config import R_MAX_CMPCH


# Multiplier on max(R) for the theta-grid upper bound.
R_MAX_FACTOR = 3.0


class DeltaSigmaPrj(SigmaPrj):
    """Two-halo excess surface density around a richness-selected cluster.

    Subclasses ``SigmaPrj`` and overrides two hooks:

    - ``_kernel_closure`` substitutes the signed ``DeltaSigma_mis``
      lookup (``NFWMiscentered._dsig_spl``, linear-space values) for
      the ``Sigma_mis`` one.  Same C++ ``2 * r_s * rho_eff * 1e-12``
      prefactor.

    - ``R_max_cMpch`` defaults to ``max(R_MAX_CMPCH,
      R_max_factor * max(R))`` at call time (see module docstring).

    The theta-grid, z- and M-integration, b_sel evaluation, xi_NL
    exclusion, and outer-loop structure are inherited unchanged.
    """

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 nfw: NFWMiscentered,
                 n_theta_per_seg: int = 48,
                 R_max_cMpch: float | None = None,
                 R_max_factor: float = R_MAX_FACTOR,
                 survey_area: SurveyArea = SurveyArea(),
                 tmap: str = "DA"):
        # n_theta_per_seg default is 48 (vs SigmaPrj's 30): the signed
        # kernel's zero crossing at theta ~ theta_R is a C0 kink
        # mid-segment, and 48 nodes hold the smallest-R (theta_R near
        # theta_excl) segment to < 0.3% vs a 120-node reference.
        super().__init__(
            cosmo=cosmo, sel_bias=sel_bias, nfw=nfw,
            n_theta_per_seg=n_theta_per_seg,
            R_max_cMpch=(R_MAX_CMPCH if R_max_cMpch is None else R_max_cMpch),
            survey_area=survey_area, tmap=tmap,
        )
        self._R_max_cMpch_user = (
            None if R_max_cMpch is None else float(R_max_cMpch))
        self.R_max_factor = float(R_max_factor)

    # ---------------- kernel override ---------------------------------------

    def _kernel_closure(self, R, ctx):
        """Signed DeltaSigma_mis lookup [Msun/h / pc^2].

        ``_dsig_spl`` holds linear-space signed values (no exp); both
        axes clipped to the stored ranges.
        """
        rs_M = ctx["rs_M"]; rho_eff = ctx["rho_s"]; D_A_o = ctx["D_A_o"]
        _dsig_spl = self.nfw._dsig_spl
        _lnx_lo = self.nfw._lnx_lo; _lnx_hi = self.nfw._lnx_hi
        _lnxmis_lo = self.nfw._lnxmis_lo; _lnxmis_hi = self.nfw._lnxmis_hi
        ln_R = np.log(R)[None, :] - np.log(rs_M)[:, None]   # (NM, NR)
        ln_R = np.clip(ln_R, _lnx_lo, _lnx_hi)
        prefac_M = 2.0 * rs_M * rho_eff * 1.0e-12           # (NM,)  Msun/h/pc^2

        def kernel(theta):
            R_theta = theta * D_A_o
            lnxmis = np.log(R_theta / rs_M)
            lnxmis = np.clip(lnxmis, _lnxmis_lo, _lnxmis_hi)
            out = np.empty_like(ln_R)
            for iM in range(rs_M.size):
                out[iM] = prefac_M[iM] * _dsig_spl(
                    lnxmis[iM:iM + 1], ln_R[iM]).ravel()
            return out
        return kernel

    # ---------------- adaptive theta_max ------------------------------------

    def _theta_grid(self, lob, zob, R_vec, ctx):
        """Inherit the parent grid recipe with the floored-adaptive
        ``R_max_cMpch`` when the user did not pin it."""
        if self._R_max_cMpch_user is not None:
            self.R_max_cMpch = self._R_max_cMpch_user
        else:
            self.R_max_cMpch = max(
                R_MAX_CMPCH, self.R_max_factor * float(np.max(R_vec)))
        return super()._theta_grid(lob, zob, R_vec, ctx)
