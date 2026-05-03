"""Closed-form bin-integrated kernels K_i (richness) and K_j (redshift).

All equation numbers refer to
``docs/richness_selection_function.tex``.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erf, erfc, erfcx

from ..plob_ltr import mu_model, sig_model, tau_model, fprj_model


_SQRT2 = np.sqrt(2.0)


def _Phi(x):
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + erf(x / _SQRT2))


def _F_EMG_vec(x, mu, sigma, tau):
    """Vectorised EMG CDF F_EMG(x; mu, sigma, tau).

    ``x, mu, sigma, tau`` broadcast against each other.  Uses the
    scaled complementary error function to avoid ``exp * erfc``
    overflow / underflow in either tail.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    z = (x - mu) / sigma
    u = (tau * sigma - z) / _SQRT2
    neg = u < 0.0
    abs_u = np.where(neg, -u, u)
    exp_mz2 = np.exp(-0.5 * z ** 2)

    # tail = exp(A) * Phi(z - tau sigma),  A = -tau(x-mu) + (tau sigma)^2 / 2
    # u >= 0: 0.5 erfcx(u) exp(-z^2/2)
    # u <  0: exp(A) - 0.5 erfcx(-u) exp(-z^2/2)
    tail_base = 0.5 * erfcx(abs_u) * exp_mz2
    # exp(A) only needed where u < 0
    A = -tau * (x - mu) + 0.5 * (tau * sigma) ** 2
    exp_A = np.where(neg, np.exp(A), 0.0)
    tail = np.where(neg, exp_A - tail_base, tail_base)

    return np.clip(_Phi(z) - tail, 0.0, 1.0)


def F_EMG(x, mu, sigma, tau):
    """EMG CDF F_EMG(x; mu, sigma, tau) (Eq. 19).

    Rewritten via the scaled complementary error function
    ``erfcx(t) = exp(t^2) erfc(t)`` to stay finite in both tails.
    Broadcasts ``(x, mu, sigma, tau)`` through numpy rules.
    """
    out = _F_EMG_vec(x, mu, sigma, tau)
    if np.ndim(x) == 0 and np.ndim(mu) == 0:
        return float(out)
    return out


def K_i(ltr, z, lam_min, lam_max):
    """Closed-form bin-integrated observed-richness kernel (Eq. 16).

        K_i(ltr, z) = (1 - fprj) [Phi((lmax - mu) / sigma)
                                  - Phi((lmin - mu) / sigma)]
                    + fprj      [F_EMG(lmax) - F_EMG(lmin)]

    Parameters use the Costanzi EMG parametrisation read through
    ``plob_ltr`` splines (four functions of ``ltr`` and ``z``).

    Fully vectorised over ``ltr``; ``z`` is a scalar.

    Parameters
    ----------
    ltr : float or array
        Latent (true) richness at which the kernel is evaluated.
    z : float
        Halo redshift (used to read the spline parameters).
    lam_min, lam_max : float
        Richness bin edges.

    Returns
    -------
    K_i : float or array
        Probability mass that a halo at ``ltr`` is assigned an
        observed richness inside ``[lam_min, lam_max]``.
    """
    ltr_arr = np.atleast_1d(np.asarray(ltr, dtype=float))

    # One spline eval per parameter over the full ltr grid.  The
    # plob_ltr models are all vectorised in their first argument.
    mu = mu_model(ltr_arr, z)
    sigma = sig_model(ltr_arr, z)
    tau = tau_model(ltr_arr, z)
    fprj = np.minimum(1.0, fprj_model(ltr_arr, z))

    gauss_piece = (_Phi((lam_max - mu) / sigma)
                   - _Phi((lam_min - mu) / sigma))
    emg_piece = (_F_EMG_vec(lam_max, mu, sigma, tau)
                 - _F_EMG_vec(lam_min, mu, sigma, tau))
    out = (1.0 - fprj) * gauss_piece + fprj * emg_piece

    if np.ndim(ltr) == 0:
        return float(out[0])
    return out


def K_j(ztr, zob_min, zob_max, sigma_z):
    """Gaussian-CDF observed-redshift kernel (Eq. 12).

        K_j(ztr) = Phi((zob_max - ztr) / sigma_z)
                 - Phi((zob_min - ztr) / sigma_z)

    ``sigma_z`` is the photo-z scatter (may be richness-bin dependent;
    caller passes the scalar that applies to the bin of interest).
    """
    ztr = np.asarray(ztr, dtype=float)
    return (_Phi((zob_max - ztr) / sigma_z)
            - _Phi((zob_min - ztr) / sigma_z))
