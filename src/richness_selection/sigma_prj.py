"""Sigma_prj(R | lob, zob) orchestrator.

Evaluates Costanzi 2026 Eq.~13:

    < Sigma_prj(R | lob, zob) > = int dz dV/dOmega int dM n(M,z)
                                  * int dtheta sin(theta)
                                    [1 + b(M,z) b_sel(theta) xi_NL(z,theta)]
                                  * Sigma_mis(R | M, z, theta, zob)

Key numerical choices (documented in docs/z_integral_recipe.tex):

- z-axis: sel_bias._z_grid_option_{E,D,C} dispatch, default Option C.
  (Inherits the same z-grid selection the sel_bias operator uses.)
- theta-axis: SPLIT at theta_R = R / D_A(zob), the location of the
  NFW peak Sigma_mis(R, R_theta = R).  Log-GL on [eps, theta_R] and
  [theta_R, theta_max].  This mirrors the split-at-exclusion trick
  that made sel_bias converge at Nth=10: peak at boundary -> GL
  nodes cluster around it -> converges at modest N per half.
  Default: N_inner = N_outer = 50.
- M-axis: shared with sel_bias (GL on ln M).

The crucial difference from sel_bias is the theta structure:
sel_bias's theta range is (theta_excl, 2 theta_lob) with no interior
peak (xi_NL peaks at the left endpoint -- which IS theta_excl).
Sigma_prj's theta range is (0, 30/D_A) with the Sigma_mis peak at
theta_R that moves with R, so the per-R split is essential.
"""
from __future__ import annotations
import numpy as np

from .cosmology import Cosmology
from .nfw import NFWMiscentered
from .sel_bias import SelBias
from .gl import gl_nodes
from .geometry import R_lambda


class SigmaPrj:
    """Projected two-halo surface density (Costanzi 2026 Eq. 13)."""

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 nfw: NFWMiscentered,
                 n_theta_inner: int = 10,
                 n_theta_outer: int = 150):
        self.cosmo = cosmo
        self.sel_bias = sel_bias
        self.nfw = nfw
        self.hmf = sel_bias.hmf
        self.bias = sel_bias.bias
        self.grid = sel_bias.grid
        self.xi_NL = sel_bias.xi_NL
        self.n_theta_inner = n_theta_inner
        self.n_theta_outer = n_theta_outer

    def _theta_grid_for_R(self, R_val: float, D_A_o: float, theta_max: float):
        """Split-at-theta_R log-GL grid for one R value.

        Returns (thetas, weights) such that
            int_0^theta_max f(theta) dtheta ~ sum(weights * f(thetas)).
        """
        theta_R = max(R_val / D_A_o, 1e-8)
        n_in, n_out = self.n_theta_inner, self.n_theta_outer
        eps_in = 1e-4 * theta_R
        if theta_R >= theta_max:
            # Peak outside integration range -- single log-GL segment
            u, wu = gl_nodes(np.log(max(eps_in, 1e-8)),
                              np.log(theta_max), n_in + n_out)
            th = np.exp(u)
            return th, wu * th
        # Inner half: [eps, theta_R], log-GL
        u_in, w_in = gl_nodes(np.log(eps_in), np.log(theta_R), n_in)
        th_in = np.exp(u_in)
        wth_in = w_in * th_in
        # Outer half: [theta_R, theta_max], log-GL
        u_out, w_out = gl_nodes(np.log(theta_R), np.log(theta_max), n_out)
        th_out = np.exp(u_out)
        wth_out = w_out * th_out
        return (np.concatenate([th_in, th_out]),
                np.concatenate([wth_in, wth_out]))

    def __call__(self, R, lob, zob):
        """Evaluate <Sigma_prj(R | lob, zob)> for an array of R values."""
        g = self.grid
        R = np.atleast_1d(R).astype(float)

        # Build the SAME z-grid that sel_bias uses (inherits z_scheme choice).
        chi_o = float(self.cosmo.chi(zob))
        D_A_o = chi_o / (1.0 + zob)
        R_excl = R_lambda(lob) * (1.0 + zob)

        # Build the z-grid via the sel_bias dispatcher
        from scipy.optimize import bisect
        from .photoz import zmin4zkernel, zmax4zkernel, w_z, sigma_z
        try:
            z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
            z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
        except ValueError:
            sig = float(sigma_z(zob))
            z_fg_lo, z_bg_hi = max(0.01, zob - sig), zob + sig

        zs_ref = np.linspace(0.0, 2.0, 2000)
        chi_ref = self.cosmo.chi(zs_ref)
        dchi_dz_ref = np.gradient(chi_ref, zs_ref)

        scheme = getattr(self.sel_bias, 'z_scheme', 'E')
        if scheme == 'E':
            zs, wzs = self.sel_bias._z_grid_option_E(
                lob, zob, g.Nz, chi_o, R_excl, z_fg_lo, z_bg_hi,
                zs_ref, chi_ref, dchi_dz_ref)
        elif scheme == 'D':
            zs, wzs = self.sel_bias._z_grid_option_D(
                lob, zob, g.Nz, chi_o, R_excl, z_fg_lo, z_bg_hi,
                zs_ref, chi_ref, dchi_dz_ref)
        else:  # C
            zs, wzs = self.sel_bias._z_grid_option_C(
                lob, zob, g.Nz, chi_o, R_excl, z_fg_lo, z_bg_hi,
                zs_ref, chi_ref, dchi_dz_ref)

        chi_z = self.cosmo.chi(zs)
        dV = self.cosmo.dV_dzdOm(zs)
        wz_kern = w_z(zs, zob)

        # M grid (shared with sel_bias)
        ln_M_min = np.log(10.0 ** np.log10(self.sel_bias.min_mass4integral))
        ln_M_max = np.log(10.0 ** self.sel_bias.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, g.NM)
        Ms = np.exp(lnMs)
        M_weight = wM * Ms                    # d ln M Jacobian absorbed

        # Precompute bias quantities at (lob, zob)
        pre = self.sel_bias.bias_precompute(lob, zob)

        # theta_max = 30 cMpc/h / D_A(zob)  (paper convention)
        theta_max = 30.0 / D_A_o

        out = np.zeros_like(R)

        # Precompute a cubic-spline cache of b_sel_marginalised(theta) on a
        # fixed dense log-theta grid covering the full support; later evaluations
        # inside the (z, R) loops are O(1) spline lookups instead of ~100 ms
        # marginalisation calls each.
        from scipy.interpolate import CubicSpline
        th_cache_lo, th_cache_hi = 1e-8, theta_max
        th_cache = np.geomspace(th_cache_lo, th_cache_hi, 200)
        bsel_cache = self.sel_bias.b_sel_marginalised(
            th_cache, lob, zob, precomp=pre)
        bsel_spline = CubicSpline(np.log(th_cache), bsel_cache,
                                   extrapolate=False)

        # Outer loop over z (so per-z theta grid for exclusion + split-at-theta_R).
        for iz in range(zs.size):
            z = zs[iz]
            if wz_kern[iz] <= 0.0:
                continue
            chi_z_i = chi_z[iz]
            dV_i = dV[iz]
            wz_i = wzs[iz]
            n_mz = self.hmf(Ms, z)
            bM_mz = self.bias(Ms, z)

            # Inner loop over R: each R has its own theta grid split at theta_R
            for iR, R_val in enumerate(R):
                ths, wth = self._theta_grid_for_R(R_val, D_A_o, theta_max)
                sin_th = np.sin(ths)
                R_theta = ths * D_A_o
                # 3-D separation (for xi_NL and exclusion)
                cos_th = np.cos(ths)
                dchi = np.sqrt(np.maximum(
                    chi_z_i**2 + chi_o**2 - 2 * chi_z_i * chi_o * cos_th, 0.0))
                xi_vals = self.xi_NL(dchi, zob)
                if self.sel_bias.exclusion:
                    xi_vals = np.where(dchi < R_excl, 0.0, xi_vals)
                # b_sel(theta) from cached spline -- O(Nth) not O(Nth * n_ltr * ...)
                bsel_th = bsel_spline(np.log(ths))

                # Sigma_mis(R_val, R_theta, M, z): stacked per-M call to the
                # spline (cheaper than M-vectorising the RectBivariateSpline).
                S_mis = np.stack(
                    [self.nfw.sigma_grid(np.array([R_val]), R_theta,
                                          float(M), z).ravel()
                     for M in Ms], axis=0)             # (NM, Nth)

                # bracket (NM, Nth): 1 + b(M,z) bsel(theta) xi(z,theta)
                bracket = 1.0 + bM_mz[:, None] * (bsel_th * xi_vals)[None, :]

                # theta integral + M integral in one einsum
                th_weight = wth * 2.0 * np.pi * sin_th
                theta_contrib = np.einsum(
                    't,Mt,Mt->M', th_weight, bracket, S_mis)
                out[iR] += (wz_i * dV_i * wz_kern[iz]
                            * np.sum(M_weight * n_mz * theta_contrib))

        return out
