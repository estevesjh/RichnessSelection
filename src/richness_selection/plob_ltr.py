"""Observed-richness likelihood P(lob | ltr, z) -- EMG kernel (C19 Eq. 16).

Implements Eq. (costanzi_kernel) of
``costanzi2026_sigma_prj_b_eff.tex'':

    P(lob | ltr, z) = (1 - f_prj) N(lob; mu, sigma)
                      + f_prj (tau/2) exp[(tau/2)(2mu + tau sigma^2 - 2 lob)]
                        * erfc((mu + tau sigma^2 - lob) / (sqrt(2) sigma))

with mu = ltr + Delta_mu(z) (the background Gaussian mean shift).
Four parameters per z, read from
``prj_params_DESY3_lss_lin_dep_getdist_v1.txt'':
  atau / btau  : tau_model(lob) = btau / lob**atau
  amu  / bmu   : mu_model(lob)  = amu + bmu * lob            (=> Delta_mu)
  asig / bsig  : sig_model(lob) = bsig * lob**asig
  afprj/bfprj  : fprj_model(lob) = bfprj / (1 + exp(-lob))**afprj
  afmsk/bfmsk  : fmask_model(lob) = bfmsk * lob**afmsk       (unused v0.1)

(a, b) parameters from Matteo's notebook cell 13.  The "lin" in the file
name means the (a, b) themselves are interpolated linearly in z over 15
redshift nodes in [0.10, 0.80].

The C26 paper's simplified delta+exp form (paper Eq. 9) is also
available via ``P_lob_given_ltr_C26`` for cross-checking the paper
plots, but it is not what we use in production.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import erfc, erf

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
        atau=ius(z_bins, a_tau, k=1, ext=3),
        btau=ius(z_bins, b_tau, k=1, ext=3),
        amu=ius(z_bins, a_mu, k=1, ext=3),
        bmu=ius(z_bins, b_mu, k=1, ext=3),
        asig=ius(z_bins, a_sig, k=1, ext=3),
        bsig=ius(z_bins, b_sig, k=1, ext=3),
        afprj=ius(z_bins, a_fprj, k=1, ext=3),
        bfprj=ius(z_bins, b_fprj, k=1, ext=3),
        afmsk=ius(z_bins, a_fmsk, k=1, ext=3),
        bfmsk=ius(z_bins, b_fmsk, k=1, ext=3),
    )
    _cache["spl"] = splines
    return splines


def mu_model(lin, z):
    """Gaussian mean mu(lin, z) = a_mu(z) + b_mu(z) * lin.

    Following Matteo's cell-13 convention, ``mu`` here is the FULL
    Gaussian mean in lob-space when called with lin=ltr (not just a
    Delta-mu shift).  For the Y3 best fits b_mu ~ 1, so mu(ltr) ~ ltr
    + a_mu with a_mu slightly negative.
    """
    s = _load_params()
    return float(s["amu"](z)) + float(s["bmu"](z)) * lin


def sig_model(lob, z):
    """sigma(lob, z) = bsig * lob^asig."""
    s = _load_params()
    return float(s["bsig"](z)) * lob ** float(s["asig"](z))


def tau_model(lob, z):
    """tau(lob, z) = btau / lob^atau."""
    s = _load_params()
    return float(s["btau"](z)) / lob ** float(s["atau"](z))


def fprj_model(lob, z):
    """f_prj(lob, z) = bfprj / (1 + exp(-lob/1))^afprj (Matteo's logistic form)."""
    s = _load_params()
    a = float(s["afprj"](z))
    b = float(s["bfprj"](z))
    return b / (1.0 + np.exp(-lob / 1.0)) ** a


def fmask_model(lob, z):
    """f_mask(lob, z) = bfmsk * lob^afmsk (unused in v0.1)."""
    s = _load_params()
    return float(s["bfmsk"](z)) * lob ** float(s["afmsk"](z))


def _norm_pdf(x, mu, sigma):
    sigma = np.asarray(sigma, dtype=float)
    return (np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2.0 * np.pi)))


def P_lob_given_ltr(lob, ltr, z):
    """Full EMG kernel P(lob | ltr, z) from Eq. (costanzi_kernel).

        P(lob | ltr, z) = (1 - f_prj) N(lob; mu, sigma)
                        + f_prj * (tau/2) * exp[(tau/2)(2 mu + tau sigma^2 - 2 lob)]
                          * erfc((mu + tau sigma^2 - lob) / (sqrt(2) sigma))

    Per Matteo's cell-13 convention, the four parameters are
    functions of ``ltr``:
        mu(ltr)    = a_mu  + b_mu  * ltr    (full Gaussian mean, not Delta)
        sigma(ltr) = b_sig * ltr**a_sig
        tau(ltr)   = b_tau / ltr**a_tau
        f_prj(ltr) = b_fprj / (1 + exp(-ltr))**a_fprj
    """
    lob_arr = np.atleast_1d(np.asarray(lob, dtype=float))
    ltr = float(ltr)

    mu = mu_model(ltr, z)
    sigma = sig_model(ltr, z)
    tau = tau_model(ltr, z)
    fprj = min(1.0, fprj_model(ltr, z))

    # Gaussian piece
    gauss = _norm_pdf(lob_arr, mu, sigma)

    # EMG piece
    exp_arg = 0.5 * tau * (2.0 * mu + tau * sigma ** 2 - 2.0 * lob_arr)
    erfc_arg = (mu + tau * sigma ** 2 - lob_arr) / (np.sqrt(2.0) * sigma)
    emg = 0.5 * tau * np.exp(exp_arg) * erfc(erfc_arg)

    out = (1.0 - fprj) * gauss + fprj * emg
    return out if out.size > 1 else float(out[0])


def P_lob_given_ltr_C26(lob, ltr, z):
    """Paper C26 Eq. 9 particular case: delta(lob - ltr) + exp tail.

    Provided for cross-checking against paper Fig. 2; NOT the
    production kernel.
    """
    lob = np.atleast_1d(np.asarray(lob, dtype=float))
    ltr = float(ltr)
    tau = tau_model(ltr, z)
    fprj = min(1.0, fprj_model(ltr, z))
    tail = np.where(lob >= ltr,
                    fprj * tau * np.exp(-tau * (lob - ltr)),
                    0.0)
    # Discrete delta weight, handled as a flag in marginalisation code.
    width = 1.0
    delta_like = ((1.0 - fprj) / (np.sqrt(2 * np.pi) * width)
                  * np.exp(-0.5 * ((lob - ltr) / width) ** 2))
    out = tail + delta_like
    return out if out.size > 1 else float(out[0])
