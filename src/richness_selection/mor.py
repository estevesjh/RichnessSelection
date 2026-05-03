"""HOD mass-observable relation (Costanzi 2026 / DES Y1 best fit).

Poisson-Gaussian convolution via the "pltr_M" form from Matteo's
Costanzi2026_SelectionBias.ipynb (cell 10).

Mean satellite count:
    l_sat(M, z) = ((M - M_min) / M_pivot)^alpha * ((1+z) / (1+z_pivot))^epsilon
Total true richness:
    l_tr(M, z) = 1 + l_sat
Intrinsic scatter:
    sig_intr(M, z) = sigma_intr * l_sat
Poisson-Gaussian convolved pdf P(ltr | M, z):
    lam = l_sat + (l_sat * sigma_intr)^2
    x   = ltr + (l_sat * sigma_intr)^2
    P   = exp(-lam + (x-1) ln(lam) - gammaln(x))

The DES Y1 NC+3x2pt best-fit parameters are used as defaults.  Caller
can override any of them through the constructor.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln


class MOR:
    """HOD mass-observable relation with Poisson-Gaussian convolution."""

    def __init__(self,
                 log10_Mmin: float = 11.3852818,
                 alpha: float = 0.858693714,
                 log10_M1: float = 12.6964410,
                 sigma_intr: float = 0.180949022,
                 epsilon: float = 0.0,
                 z_pivot: float = 0.4544):
        self.M_min = 10.0 ** log10_Mmin
        self.M1 = 10.0 ** log10_M1
        self.M_pivot = self.M1 - self.M_min
        self.alpha = alpha
        self.sigma_intr = sigma_intr
        self.epsilon = epsilon
        self.z_pivot = z_pivot

    def l_sat(self, M, z):
        """Mean number of satellite galaxies above the richness threshold."""
        M = np.asarray(M, dtype=float)
        z = np.asarray(z, dtype=float)
        frac = np.clip((M - self.M_min) / self.M_pivot, 1e-30, None)
        return frac ** self.alpha * ((1.0 + z) / (1.0 + self.z_pivot)) ** self.epsilon

    def l_tr(self, M, z):
        return 1.0 + self.l_sat(M, z)

    def sig_intr(self, M, z):
        return self.sigma_intr * self.l_sat(M, z)

    # Effective first / second moments of P(ltr | M, z) used to
    # bracket the GL quadrature in selection_function.S_i.
    def ltr_mean(self, M, z):
        """Effective mean of the shifted Poisson; to leading order
        this is l_sat.  Added for a uniform MOR API shared with
        LogNormalMOR."""
        return self.l_sat(M, z)

    def ltr_sigma(self, M, z):
        """Effective sigma: sqrt(l_sat + (sigma_intr * l_sat)^2)."""
        m = self.l_sat(M, z)
        return np.sqrt(m + (m * self.sigma_intr) ** 2)

    def pdf(self, ltr, M, z):
        """P(ltr | M, z) via Poisson-Gaussian convolution.  Broadcast-safe
        over (ltr, M); z is broadcast against M.
        """
        ltr = np.asarray(ltr, dtype=float)
        m = self.l_sat(M, z)
        mi = (m * self.sigma_intr) ** 2
        lam = m + mi
        x = ltr + mi          # relies on ltr, mi having compatible shapes
        ln_gamma = gammaln(x)
        val = np.exp(-lam + (x - 1.0) * np.log(np.clip(lam, 1e-300, None))
                     - ln_gamma)
        return np.where(ltr >= 0.0, val, 0.0)

    def lambda_mean_below(self, M, z, lob, ltr_n=400):
        """<ltr>_{< lob}(M, z) = int_0^lob ltr P(ltr | M, z) dltr via trapz.

        Integrates up to min(lob, ltr_cap) where ltr_cap is tracked to the
        10*sqrt(lam) tail of the Poisson-Gaussian so we resolve the peak even
        when lob is far beyond it.
        """
        M_arr = np.atleast_1d(np.asarray(M, dtype=float))
        # Rough pdf width per mass
        m = self.l_sat(M_arr, z)
        mi = (m * self.sigma_intr) ** 2
        lam = m + mi
        ltr_max_per_M = lam + 15.0 * np.sqrt(lam + 1.0)
        upper = min(float(lob), float(np.max(ltr_max_per_M)) + 1.0)
        ltr_grid = np.linspace(0.0, upper, ltr_n)
        p = self.pdf(ltr_grid[:, None], M_arr[None, :], z)   # (Nltr, NM)
        integrand = ltr_grid[:, None] * p
        out = np.trapz(integrand, ltr_grid, axis=0)
        if np.ndim(M) == 0:
            return float(out[0])
        return out

    # Alias used by tests + old code
    def mean_lambda_below(self, M, z, lob):
        return self.lambda_mean_below(M, z, lob)


class LogNormalMOR:
    """Log-normal mass-observable relation (Costanzi 2021 Eq. 2).

    Parametrises the mean of ln(lambda) as a linear combination of
    ln(M/Mp) and ln((1+z)/(1+zp)); scatter is log-normal with the
    Costanzi-2021 variance

        sigma^2_ln(lambda) = D^2 + (<lambda> - 1) / <lambda>^2,

    where the second term is the Poisson-like contribution from
    discrete galaxy counts.

    Default parameters are the Costanzi 2021 DES+SPT best fit (their
    Tab. 2): pivot mass Mp = 3e14 h^-1 Msun, pivot redshift zp=0.45.
    """

    def __init__(self,
                 A_lambda: float = 76.9,
                 B_lambda: float = 1.020,
                 C_lambda: float = 0.29,
                 D_lambda: float = 0.23,
                 M_pivot: float = 3.0e14,
                 z_pivot: float = 0.45):
        self.A_lambda = A_lambda
        self.B_lambda = B_lambda
        self.C_lambda = C_lambda
        self.D_lambda = D_lambda
        self.M_pivot = M_pivot
        self.z_pivot = z_pivot

    def ln_lambda_mean(self, M, z):
        """<ln lambda>(M, z), Eq. (34) of the doc."""
        M = np.asarray(M, dtype=float)
        z = np.asarray(z, dtype=float)
        return (np.log(self.A_lambda)
                + self.B_lambda * np.log(M / self.M_pivot)
                + self.C_lambda * np.log((1.0 + z) / (1.0 + self.z_pivot)))

    def _sigma_ln_lambda(self, M, z):
        """sigma_ln_lambda(M, z) from Eq. (35)."""
        lam_lin = np.exp(self.ln_lambda_mean(M, z))
        # Guard against lam < 1 (the Poisson-like term would flip sign).
        lam_lin = np.maximum(lam_lin, 1.0 + 1e-12)
        var = self.D_lambda ** 2 + (lam_lin - 1.0) / lam_lin ** 2
        return np.sqrt(var)

    def ltr_mean(self, M, z):
        """Linear-space mean <lambda>(M, z) = exp(mu + sig^2/2)."""
        mu = self.ln_lambda_mean(M, z)
        s2 = self._sigma_ln_lambda(M, z) ** 2
        return np.exp(mu + 0.5 * s2)

    def ltr_sigma(self, M, z):
        """Linear-space scatter sigma_lambda = <lambda> sqrt(exp(sig^2) - 1)."""
        s2 = self._sigma_ln_lambda(M, z) ** 2
        return self.ltr_mean(M, z) * np.sqrt(np.expm1(s2))

    def pdf(self, ltr, M, z):
        """P(ltr | M, z) log-normal density, Eq. (36)."""
        ltr = np.asarray(ltr, dtype=float)
        mu_ln = self.ln_lambda_mean(M, z)
        sig_ln = self._sigma_ln_lambda(M, z)
        # 0 for ltr <= 0, else (1 / (ltr sigma sqrt(2 pi))) exp(-(...)^2/2)
        val = np.where(
            ltr > 0.0,
            np.exp(-0.5 * ((np.log(np.clip(ltr, 1e-300, None)) - mu_ln)
                            / sig_ln) ** 2)
            / (np.clip(ltr, 1e-300, None) * sig_ln * np.sqrt(2.0 * np.pi)),
            0.0,
        )
        return val
