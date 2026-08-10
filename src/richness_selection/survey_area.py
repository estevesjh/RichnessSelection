"""Survey-footprint solid angle Omega(z) for the Sigma_prj/DeltaSigma_prj
pipeline.

``SigmaPrj``/``DeltaSigmaPrj``/``FrozenDeltaSigmaPrj`` historically have no
survey-area weighting at all in their z-integral (``outer_weight = wzs * dV
* w_z(z)``) -- this matches production ``ShearPrjEvaluator``'s own
convention, which has an explicit comment that Omega(z) must NOT appear for
a surface-density observable (it cancels between numerator and
normalisation, unlike number counts -- see
``y3_cluster_cpp/src/models/sigma_prj_t.hh``). This was confirmed
empirically while benchmarking the frozen-physics ports of this pipeline
(see ``REPORT_frozen_physics_pipeline_port.md``): a C++ variant that
included Omega(z) broke the fiducial mock-data-vector closure
(Likelihood -151.68 instead of ~0), while omitting it restored closure.

``SurveyArea`` makes that choice an explicit, pluggable parameter instead of
a hardcoded absence, so the question can be revisited later (e.g. once a
mock data vector is regenerated with a deliberate survey-area convention)
without editing the integration code itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .selection_function.survey import omega_z_des, omega_z_sdss

__all__ = ["SurveyArea"]

_POLY_MODELS = {"des": omega_z_des, "sdss": omega_z_sdss}


@dataclass(frozen=True)
class SurveyArea:
    """Effective survey solid angle Omega(z), pluggable into the
    Sigma_prj/DeltaSigma_prj z-integral outer weight.

    Three variants, matching ``y3_cluster_cpp/src/models/omega_z_*.hh``:

    - ``kind="unity"`` (default): Omega(z) = 1 -- no survey-area weighting.
      Reproduces the historical/current behaviour of this pipeline exactly.
    - ``kind="constant"``: Omega(z) = ``value`` (steradians), independent of
      z. A pure overall rescaling: every z-integral output (rnd and cl
      alike) scales by this same factor, since it multiplies the shared
      ``outer_weight`` before the rnd/cl channels diverge.
    - ``kind="polynomial"``: Omega(z) = a piecewise polynomial survey-area
      fit. Reuses ``selection_function.survey.omega_z_des``/``omega_z_sdss``
      (themselves ported from ``omega_z_des.hh``/``omega_z_sdss.hh``)
      rather than duplicating coefficients here. ``model`` selects
      ``"des"`` or ``"sdss"``.

    Parameters
    ----------
    kind : str
        One of ``"unity"``, ``"constant"``, ``"polynomial"``.
    value : float
        Solid angle in steradians, used only when ``kind="constant"``.
    model : str
        Polynomial fit to use, used only when ``kind="polynomial"``.
    """

    kind: str = "unity"
    value: float = 1.0
    model: str = "des"

    def __post_init__(self) -> None:
        if self.kind not in ("unity", "constant", "polynomial"):
            raise ValueError(
                f"unknown SurveyArea kind {self.kind!r}, expected one of "
                "'unity', 'constant', 'polynomial'")
        if self.kind == "polynomial" and self.model not in _POLY_MODELS:
            raise ValueError(
                f"unknown polynomial model {self.model!r}, expected one of "
                f"{sorted(_POLY_MODELS)}")

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        if self.kind == "unity":
            return np.ones_like(z)
        if self.kind == "constant":
            return np.full_like(z, float(self.value))
        return _POLY_MODELS[self.model](z)
