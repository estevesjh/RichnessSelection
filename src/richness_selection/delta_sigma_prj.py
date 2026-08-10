"""DeltaSigma_prj(R | lob, zob): two-halo excess surface density.

Evaluates the cluster--lensing observable

    < DeltaSigma^prj(R | lob, zob) > =
        int dz dV/dOmega int dM n(M, z)
        * int dtheta sin(theta)
          [1 + b(M, z) b_sel(theta) xi_NL(|Delta r|, zob)]
        * DeltaSigma_mis(R | M, z, R_mis = theta * D_A(zob))

which is obtained from the ``SigmaPrj`` Eq. 13 kernel by swapping
``Sigma_mis`` for its radial excess ``DeltaSigma_mis = bar_Sigma(<R) -
Sigma(R)``.  The argument is developed in
``docs/delta_sigma_prj_derivation.tex``: the radial-average-and-
subtraction functional is linear in R only, so it commutes with the
(theta, z, M) integrals, and the only change from the ``SigmaPrj``
pipeline is the underlying NFW lookup (``_dsig_spl`` vs ``_spl`` in
``NFWMiscentered``).

Defaults
--------

- By default the class returns the ``cl+LSS`` piece, matching the
  convention of ``SigmaPrj`` (the ``rnd`` / uniform-background term
  vanishes in the full-aperture limit and is a boundary-truncation
  artefact; see ``docs/delta_sigma_prj_derivation.tex`` Section 2).

- The theta-upper limit is set adaptively from the requested R-grid:
  ``R_max_cMpch = R_max_factor * max(R)``, with ``R_max_factor = 3``
  (``docs/delta_sigma_prj_derivation.tex`` Section 3).  The rationale
  is that DeltaSigma_mis(R | R_mis) has compact support around
  R_mis ~ R and dies for R_mis >> R, unlike Sigma_mis which plateaus
  at a constant in R.  The ``SigmaPrj`` legacy ``R_max = 30 cMpc/h``
  is therefore unnecessarily generous here.

  Pass ``R_max_cMpch`` explicitly to override the adaptive rule.
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology
from .nfw import NFWMiscentered
from .sel_bias import SelBias
from .sigma_prj import SigmaPrj
from .survey_area import SurveyArea


# Default safety factor for the adaptive theta_max rule
# (recipe in docs/delta_sigma_prj_derivation.tex Section 3).
R_MAX_FACTOR = 3.0


class DeltaSigmaPrj(SigmaPrj):
    """Two-halo excess surface density around a richness-selected cluster.

    Subclasses ``SigmaPrj`` and overrides two hooks:

    - ``_kernel_closure`` substitutes ``NFWMiscentered._dsig_spl``
      for ``_spl`` inside the per-theta (NM, NR) lookup.  Same C++
      ``2 * r_s * rho_eff * 1e-12`` prefactor (both tables are in
      the C++ kernel's natural units).

    - ``R_max_cMpch`` defaults to ``R_max_factor * max(R_grid)`` when
      ``None`` is passed; otherwise forwards to ``SigmaPrj``.  The
      default ``R_max_factor`` is ``3.0`` (see module docstring).

    The theta-grid, z- and M-integration, b_sel evaluation, xi_NL
    exclusion, and outer-loop structure are inherited unchanged.

    Parameters
    ----------
    R_max_cMpch : float or None
        If ``None`` (default), use the adaptive rule
        ``R_max_factor * max(R)`` at call time.  Otherwise, forward
        to ``SigmaPrj`` as a fixed upper bound.
    R_max_factor : float
        Multiplier on ``max(R)`` for the adaptive rule.  Default 3.0.
    """

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 nfw: NFWMiscentered,
                 n_theta_per_seg: int = 30,
                 R_max_cMpch: float | None = None,
                 R_max_factor: float = R_MAX_FACTOR,
                 survey_area: SurveyArea = SurveyArea()):
        # Placeholder ``R_max_cMpch`` for the parent; we override
        # ``_theta_grid`` below to apply the adaptive rule when
        # ``self._R_max_cMpch_user is None``.
        super().__init__(
            cosmo=cosmo, sel_bias=sel_bias, nfw=nfw,
            n_theta_per_seg=n_theta_per_seg,
            R_max_cMpch=(30.0 if R_max_cMpch is None else R_max_cMpch),
            survey_area=survey_area,
        )
        self._R_max_cMpch_user = (
            None if R_max_cMpch is None else float(R_max_cMpch))
        self.R_max_factor = float(R_max_factor)

    # ---------------- kernel override ---------------------------------------

    def _kernel_closure(self, R, ctx):
        """DeltaSigma_mis lookup in the C++ convention [Msun/h / pc^2]."""
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
                out[iM] = prefac_M[iM] * np.exp(
                    _dsig_spl(lnxmis[iM:iM + 1], ln_R[iM])).ravel()
            return out
        return kernel

    # ---------------- adaptive theta_max ------------------------------------

    def _theta_grid(self, lob, zob, R_vec, ctx):
        """Inherit the parent grid recipe with an adaptive ``R_max_cMpch``
        when the user did not pin it.
        """
        if self._R_max_cMpch_user is None:
            self.R_max_cMpch = self.R_max_factor * float(np.max(R_vec))
        else:
            self.R_max_cMpch = self._R_max_cMpch_user
        return super()._theta_grid(lob, zob, R_vec, ctx)
