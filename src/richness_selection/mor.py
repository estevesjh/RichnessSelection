"""Log-normal mass-observable relation (MOR).

    mu_ln_lambda(M, z) = A_mu + B_mu ln(M/3e14) + C_mu ln(1+z)
    ln lambda | M, z ~ Normal(mu, sigma^2)

Closed form (S9) for the partial first moment used in the projection
kernel integrand:
    <lambda>_{< lob}(M, z)
        = exp(mu + sigma^2/2) * Phi( (ln lob - mu - sigma^2) / sigma )
with Phi the standard Normal CDF.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erf


class MOR:
    """Log-normal mass-observable relation with S9 closed form."""

    def __init__(self, A_mu=np.log(60.0), B_mu=0.9, C_mu=-0.3, sigma=0.25,
                 M_pivot=3e14):
        self.A_mu = A_mu
        self.B_mu = B_mu
        self.C_mu = C_mu
        self.sigma = sigma
        self.M_pivot = M_pivot

    def mu_ln_lambda(self, M, z):
        M = np.asarray(M, dtype=float)
        z = np.asarray(z, dtype=float)
        return (self.A_mu
                + self.B_mu * (np.log(M) - np.log(self.M_pivot))
                + self.C_mu * np.log(1.0 + z))

    def pdf(self, lam, M, z):
        """Log-normal p(lambda | M, z)."""
        lam = np.asarray(lam, dtype=float)
        mu = self.mu_ln_lambda(M, z)
        sig = self.sigma
        x = np.log(lam)
        return (np.exp(-0.5 * ((x - mu) / sig) ** 2)
                / (lam * sig * np.sqrt(2.0 * np.pi)))

    def lambda_mean_below(self, M, z, lob):
        """S9 closed form: int_0^lob lambda p(lambda | M, z) d lambda."""
        mu = self.mu_ln_lambda(M, z)
        sig = self.sigma
        arg = (np.log(lob) - mu - sig * sig) / sig
        Phi = 0.5 * (1.0 + erf(arg / np.sqrt(2.0)))
        return np.exp(mu + 0.5 * sig * sig) * Phi
