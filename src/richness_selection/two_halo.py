"""Traditional two-halo term for the projected surface density.

    Sigma_2h(R | lob, zob)  = rho_m * b_sel_ls * C_xi(R)
    DeltaSigma_2h(R)        = Sigmabar_2h(<R) - Sigma_2h(R)

with the cylinder projection of the matter correlation function

    C_xi(R) = int_{-L}^{+L} dl  xi_NL( sqrt(R^2 + l^2), zob )

All lengths are comoving [cMpc/h]; l is the comoving line-of-sight
offset from the cluster.  rho_m = Omega_m * rho_crit,0 (comoving mean
matter density today) and the output carries the pipeline's lensing
units [Msun/h / pc^2] via the same 1e-12 (cMpc/h)^-2 -> pc^-2 factor
as ``nfw.py``.

Why nothing else appears
------------------------
The neighbor-counting form (Costanzi 2026 Eq. 13) collapses to this
expression at two-halo scales:

* volume element and angular conversion cancel identically
  (dV/dOmega/dz = chi^2 dchi/dz against the 1/chi^2 from the
  theta -> transverse map): "the volume goes to 1";
* the photo-z kernel w_z ~ 1 over the ~10-20 cMpc/h support of xi
  (verified to < 0.6%);
* the mass integral obeys the halo-model sum rule
  int dM n(M) b(M) M = rho_m, so no HMF, bias, or NFW neighbor
  profile survives -- lensing sees all matter, mass-weighted bias 1.

The uniform mean-field ("rnd") term is excluded by definition: the
two-halo term is the *excess* over the mean field, and a uniform sheet
has DeltaSigma = 0 exactly.

Exclusion
---------
``exclusion=True`` (default) zeroes xi inside the 3-D ball
r < R_excl = R_lambda(lob) * (1 + zob), the comoving one-halo exclusion
of the Costanzi convention.  Only affects R <~ 2 cMpc/h.

b_sel_ls
--------
The large-scale plateau of the marginalised selection bias
b_sel(theta | lob, zob), evaluated at theta_ls = 30 cMpc/h / chi(zob)
where it is scale-independent.  This is the full selection-weighted
mean cluster bias (Tinker-level amplitude), not a ratio.
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology
from .sel_bias import SelBias
from .geometry import R_lambda

# (cMpc/h)^-2 -> pc^-2, same convention as nfw.py
_CMPCH2_TO_PC2 = 1.0e-12


class TwoHalo:
    """Traditional 2h surface density: rho_m * b_sel_ls * C_xi(R).

    Parameters
    ----------
    cosmo, sel_bias : shared stack objects (xi_NL is taken from
        ``sel_bias.xi_NL``, matching SigmaPrj).
    L_los : float
        Comoving LoS half-depth of the cylinder [cMpc/h].  xi has
        decayed by ~100; the integral converges well before 200.
    n_los : int
        Trapezoid nodes on [-L_los, +L_los].
    exclusion : bool
        Zero xi inside r < R_excl(lob, zob).
    """

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 L_los: float = 200.0, n_los: int = 4001,
                 exclusion: bool = True):
        self.cosmo = cosmo
        self.sel_bias = sel_bias
        self.xi_NL = sel_bias.xi_NL
        self.L_los = float(L_los)
        self.n_los = int(n_los)
        self.exclusion = bool(exclusion)
        self._bsel_cache = {}

    # ---------------- pieces -------------------------------------------------

    def b_sel_ls(self, lob, zob) -> float:
        """Large-scale plateau of the marginalised selection bias."""
        key = (float(lob), float(zob))
        if key not in self._bsel_cache:
            pre = self.sel_bias.bias_precompute(lob, zob)
            fn = self.sel_bias.marginalised_bias(lob, zob, precomp=pre)
            theta_ls = 30.0 / float(self.cosmo.chi(zob))
            self._bsel_cache[key] = float(fn(np.array([theta_ls]))[0])
        return self._bsel_cache[key]

    def C_xi(self, R, lob, zob):
        """Cylinder projection int dl xi(sqrt(R^2+l^2), zob) [cMpc/h]."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        ell = np.linspace(-self.L_los, self.L_los, self.n_los)
        r3d = np.sqrt(R[:, None] ** 2 + ell[None, :] ** 2)
        v = self.xi_NL(r3d, zob)
        if self.exclusion:
            R_excl = float(R_lambda(lob)) * (1.0 + zob)
            v = np.where(r3d > R_excl, v, 0.0)
        return np.trapezoid(v, ell, axis=1)

    # ---------------- public -------------------------------------------------

    def sigma(self, R, lob, zob):
        """Sigma_2h(R) [Msun/h / pc^2]; R comoving [cMpc/h]."""
        R = np.atleast_1d(np.asarray(R, dtype=float))
        amp = self.cosmo.rho_m0() * self.b_sel_ls(lob, zob) * _CMPCH2_TO_PC2
        return amp * self.C_xi(R, lob, zob)

    def delta_sigma(self, R, lob, zob, *, n_grid: int = 240,
                    R_min_grid: float = 0.02):
        """DeltaSigma_2h(R) = Sigmabar(<R) - Sigma(R) [Msun/h / pc^2].

        Exact cumulative on an internal dense log grid from
        ``R_min_grid`` to max(R); the disk interior to the grid start
        contributes Sigma(R_min_grid) * R_min_grid^2 (constant-Sigma
        cap, sub-0.1% for the xi profile).
        """
        R = np.atleast_1d(np.asarray(R, dtype=float))
        Rg = np.geomspace(R_min_grid, float(R.max()), int(n_grid))
        Sg = self.sigma(Rg, lob, zob)
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (Sg[1:] * Rg[1:] + Sg[:-1] * Rg[:-1]) * np.diff(Rg))])
        cum_in = Sg[0] * Rg[0] ** 2 / 2.0
        Sbar_g = 2.0 * (cum + cum_in) / Rg ** 2
        Sbar = np.interp(R, Rg, Sbar_g)
        return Sbar - np.interp(R, Rg, Sg)

    def __call__(self, R, lob, zob, *, return_decomposition: bool = False):
        """DeltaSigma_2h(R); mirrors the DeltaSigmaPrj call signature.

        ``return_decomposition=True`` returns ``{cl, rnd, total}`` with
        ``rnd = 0`` identically -- the mean field is excluded by
        definition of the two-halo term.
        """
        ds = self.delta_sigma(R, lob, zob)
        if return_decomposition:
            zero = np.zeros_like(ds)
            return dict(total=ds, rnd=zero, cl=ds)
        return ds
