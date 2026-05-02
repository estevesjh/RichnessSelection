"""Projection model P(lob | ltr, z) — Costanzi 2026 eq. 17.

Loads the DES Y3 best-fit projection parameters from
`prj_params_DESY3_lss_lin_dep_getdist_v1.txt` (Matteo's notebook cell 13)
and provides P(lob | ltr, z) = (1 - f_prj) * delta(lob - ltr)
                               + f_prj * tau * exp(-tau * (lob - ltr))
(with a hard cutoff at lob < ltr, modulo the mask factor).

Reading order of the 15-row x 10-column table (z nodes at
z in [0.10, 0.80, 15]):
    cols: a_tau, b_tau, a_mu, b_mu, a_sig, b_sig, a_fprj, b_fprj, a_fmsk, b_fmsk
"""
from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DEFAULT = os.path.abspath(os.path.join(_HERE, "..", "..", "data"))
_PRJ_PARAMS = "prj_params_DESY3_lss_lin_dep_getdist_v1.txt"


_cache = {}


def _load_params():
    if "spl" in _cache:
        return _cache["spl"]
    data_dir = os.environ.get("RICHNESS_SELECTION_DATA_DIR", _DATA_DEFAULT)
    path = os.path.join(data_dir, _PRJ_PARAMS)
    params = np.loadtxt(path).T
    z_bins = np.linspace(0.10, 0.80, 15)
    (a_tau, b_tau, a_mu, b_mu, a_sig, b_sig,
     a_fprj, b_fprj, a_fmsk, b_fmsk) = params
    splines = dict(
        atau=ius(z_bins, a_tau, k=1),
        btau=ius(z_bins, b_tau, k=1),
        amu=ius(z_bins, a_mu, k=1),
        bmu=ius(z_bins, b_mu, k=1),
        asig=ius(z_bins, a_sig, k=1),
        bsig=ius(z_bins, b_sig, k=1),
        afprj=ius(z_bins, a_fprj, k=1),
        bfprj=ius(z_bins, b_fprj, k=1),
        afmsk=ius(z_bins, a_fmsk, k=1),
        bfmsk=ius(z_bins, b_fmsk, k=1),
    )
    _cache["spl"] = splines
    return splines


def tau_model(lob, z):
    s = _load_params()
    return float(s["btau"](z)) / lob ** float(s["atau"](z))


def fprj_model(lob, z):
    """f_prj(lob, z); Matteo's form: b / (1 + exp(-lob/1))^a."""
    s = _load_params()
    a = float(s["afprj"](z))
    b = float(s["bfprj"](z))
    return b / (1.0 + np.exp(-lob / 1.0)) ** a


def P_lob_given_ltr(lob, ltr, z):
    """P(lob | ltr, z) from Costanzi 2026 eq. 17.

    Returns a pdf in lob.  The delta function at lob == ltr is smeared into a
    narrow Gaussian (width = sqrt(ltr)) so numerical integrals over lob pick
    up both components.
    """
    lob = np.atleast_1d(np.asarray(lob, dtype=float))
    ltr = float(ltr)
    if ltr <= 0:
        return np.zeros_like(lob)
    tau = tau_model(ltr, z)
    fprj = fprj_model(ltr, z)
    fprj = min(1.0, fprj)
    # Exponential tail
    tail = np.where(lob >= ltr,
                    fprj * tau * np.exp(-tau * (lob - ltr)),
                    0.0)
    # Delta-function approximation: narrow Gaussian centred at ltr.
    width = max(1.0, np.sqrt(ltr))
    delta_like = ((1.0 - fprj) / (np.sqrt(2 * np.pi) * width)
                  * np.exp(-0.5 * ((lob - ltr) / width) ** 2))
    return tail + delta_like
