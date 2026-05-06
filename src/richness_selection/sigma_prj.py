"""Sigma_prj(R | lob, zob) orchestrator.

Evaluates Costanzi 2026 Eq.~13:

    < Sigma_prj(R | lob, zob) > = int dz dV/dOmega int dM n(M,z)
                                  * int dtheta sin(theta)
                                    [1 + b(M,z) b_sel(theta) xi_NL(z,theta)]
                                  * Sigma_mis(R | M, z, theta, zob)

Structure
---------

theta is the OUTER integral.  For each theta, we build
    N_rnd(R, theta)   = int dz dV w_z int dM n Sigma_mis(R | M, z, R_mis)
    N_cl(R, theta)    = int dz dV w_z int dM n b Sigma_mis(R | M, z, R_mis)
                          * xi_NL(|Dr|, zob) * 1[theta > theta_excl(z)]
with  R_mis = theta * D_A(z_ob).  The full integrand per theta is
    2 pi sin(theta) * [ N_rnd(R, theta) + b_sel(theta) * N_cl(R, theta) ]
which we accumulate over a split-at-breakpoints log-GL theta grid.

Key numerical choices
---------------------

- z-axis: inherits ``SelBias._z_grid`` (ring + outer-fg + outer-bg).
- theta-grid: log-GL on each segment defined by breakpoints
    {theta_excl_o, theta_R_min, theta_R_max, theta_lambda, 2*theta_lambda,
     theta_max}.  This is where the integrand has features (peaks, bends,
     boundaries); log-GL per segment clusters nodes at both ends of each
     segment, which is where the resolution matters.
- exclusion: per-z LoS-slab via ``theta_excl(z)``, matching the
  convention of ``SelBias._P_operator``.  The current theta value is
  compared against ``theta_excl(z)`` and xi_NL is zeroed below it -
  no 3-D ball mask.
- NFW convention: ``nfw.sigma_grid`` returns paper Eq. 14 values
  (factor of 2 already applied inside the lookup).
"""
from __future__ import annotations
import numpy as np

from scipy.optimize import bisect

from .cosmology import Cosmology
from .nfw import NFWMiscentered
from .sel_bias import SelBias
from .gl import gl_nodes
from .geometry import R_lambda, theta_lambda
from .photoz import w_z, sigma_z, zmin4zkernel, zmax4zkernel
from .config import R_MAX_CMPCH


class SigmaPrj:
    """Projected two-halo (cl+LSS) surface density around a richness-selected
    cluster.

    ``__call__`` returns the two-halo correlation-excess surface density
    ``Sigma_{2h}(R) = int dz dV w_z int dM n(M,z) b(M,z) b_sel(theta)
                     xi_NL(|Dr|, z_ob) Sigma_mis(R|M,z,R_mis)``
    integrated over ``2*pi sin(theta) d theta``.  This is the
    correlation-only piece -- the ``1`` of ``[1 + b xi]`` is a uniform
    cosmological background that drops out of Delta Sigma and is
    normally excluded from the two-halo lensing observable.

    ``return_decomposition=True`` exposes three arrays:
      ``rnd``   the ``[1]`` piece (uniform-background integral);
      ``cl``    the ``b xi_NL`` piece (default return value);
      ``total`` the sum ``rnd + cl`` (Costanzi 2026 Eq. 13 full form).

    Parameters
    ----------
    n_theta_per_seg : int
        Gauss-Legendre nodes per theta-segment.  Default 30.
    R_max_cMpch : float
        Upper bound on the theta integral, expressed in cMpc/h of
        projected radius (``theta_max = R_max_cMpch / D_A(z_ob)``).
        Defaults to ``config.R_MAX_CMPCH = 30.0``, matching the
        ``twoD_prj_NFW`` hard truncation in Matteo's SelectionBias
        notebook.  The cl piece converges on its own (``xi_NL``
        decays fast in r); the RND piece scales with this cutoff.
    """

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 nfw: NFWMiscentered,
                 n_theta_per_seg: int = 30,
                 R_max_cMpch: float = R_MAX_CMPCH):
        self.cosmo = cosmo
        self.sel_bias = sel_bias
        self.nfw = nfw
        self.hmf = sel_bias.hmf
        self.bias = sel_bias.bias
        self.grid = sel_bias.grid
        self.xi_NL = sel_bias.xi_NL
        self.n_theta_per_seg = int(n_theta_per_seg)
        self.R_max_cMpch = float(R_max_cMpch)

    # ---------------- context ------------------------------------------------

    def _build_zM_context(self, lob, zob) -> dict:
        g = self.grid
        chi_o = float(self.cosmo.chi(zob))
        D_A_o = chi_o / (1.0 + zob)
        R_excl = R_lambda(lob) * (1.0 + zob)

        try:
            z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
            z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
        except ValueError:
            sig = float(sigma_z(zob))
            z_fg_lo, z_bg_hi = max(0.01, zob - sig), zob + sig

        zs_ref = np.linspace(0.0, 2.0, 2000)
        chi_ref = self.cosmo.chi(zs_ref)
        dchi_dz_ref = np.gradient(chi_ref, zs_ref)

        zs, wzs = self.sel_bias._z_grid(
            lob, zob, g.Nz, chi_o, R_excl, z_fg_lo, z_bg_hi,
            zs_ref, chi_ref, dchi_dz_ref)

        chi_z = self.cosmo.chi(zs)
        dV = self.cosmo.dV_dzdOm(zs)
        wz_kern = w_z(zs, zob)

        ln_M_min = np.log(10.0 ** np.log10(self.sel_bias.min_mass4integral))
        ln_M_max = np.log(10.0 ** self.sel_bias.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, g.NM)
        Ms = np.exp(lnMs)
        M_weight = wM * Ms

        n_mz = self.hmf(Ms[:, None], zs[None, :])     # (NM, Nz)
        bM_mz = self.bias(Ms[:, None], zs[None, :])   # (NM, Nz)

        # per-z theta_excl (LoS-slab exclusion, matching SelBias._P_operator)
        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl_clipped = np.clip(cos_excl, -1.0, 1.0)
        theta_excl_z = np.arccos(cos_excl_clipped)
        no_excl = cos_excl >= 1.0 - 1e-12
        theta_excl_z = np.where(no_excl, 0.0, theta_excl_z)

        # outer weight = wzs * dV * w_z(z)
        outer_weight = wzs * dV * wz_kern

        # Precompute per-M NFW scale radii and rho_s (z-independent in the
        # fixed-concentration convention) so Sigma_mis_per_theta(theta) is
        # just a vectorised spline lookup + exp.
        rho_m = self.cosmo.Om0 * 2.77533742639e11
        r200m = (3.0 * Ms / (4.0 * np.pi * 200.0 * rho_m)) ** (1.0 / 3.0)
        rs_M = r200m / self.nfw.c
        fc = np.log(1.0 + self.nfw.c) - self.nfw.c / (1.0 + self.nfw.c)
        rho_s = rho_m * (200.0 / 3.0) * self.nfw.c ** 3 / fc

        return dict(
            zs=zs, chi_z=chi_z, outer_weight=outer_weight,
            Ms=Ms, M_weight=M_weight, n_mz=n_mz, bM_mz=bM_mz,
            theta_excl_z=theta_excl_z,
            chi_o=chi_o, D_A_o=D_A_o, R_excl=R_excl,
            rs_M=rs_M, rho_s=rho_s,
        )

    # ---------------- theta grid ---------------------------------------------

    def _theta_grid(self, lob, zob, R_vec, ctx):
        """Log-GL on segments split at feature breakpoints."""
        chi_o = ctx["chi_o"]
        D_A_o = ctx["D_A_o"]
        R_excl = ctx["R_excl"]
        theta_lam = float(theta_lambda(lob, zob, self.cosmo))
        theta_excl_o = R_excl / chi_o          # on-ring LoS exclusion
        theta_R_arr = R_vec / D_A_o            # every requested R lands a breakpoint
        theta_R_min = float(np.min(theta_R_arr))
        theta_R_max = float(np.max(theta_R_arr))

        # theta_max corresponds to a projected radius R_max at z_ob (cMpc/h).
        # If the user requested a particularly large R, extend the grid to
        # cover 3 * theta_R_max too, so the breakpoints stay meaningful.
        theta_max = max(self.R_max_cMpch / D_A_o, 3.0 * theta_R_max)

        lower = max(1e-8, 0.1 * min(theta_excl_o, theta_R_min, theta_lam))
        breakpoints = sorted(set(
            [lower, theta_excl_o, theta_lam, 2.0 * theta_lam, theta_max]
            + list(theta_R_arr)
        ))
        breakpoints = [b for b in breakpoints if b > 0 and b <= theta_max]
        if breakpoints[0] > lower:
            breakpoints = [lower] + breakpoints
        if breakpoints[-1] < theta_max:
            breakpoints = breakpoints + [theta_max]
        # dedupe close-by breakpoints
        clean = [breakpoints[0]]
        for b in breakpoints[1:]:
            if b > clean[-1] * (1.0 + 1e-6):
                clean.append(b)
        breakpoints = clean

        thetas = []
        weights = []
        n_per = self.n_theta_per_seg
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            u, wu = gl_nodes(np.log(a), np.log(b), n_per)
            th = np.exp(u)
            thetas.append(th)
            weights.append(wu * th)          # d theta = theta d ln theta
        thetas = np.concatenate(thetas)
        weights = np.concatenate(weights)

        info = dict(
            theta_lam=theta_lam,
            theta_excl_o=theta_excl_o,
            theta_R_min=theta_R_min,
            theta_R_max=theta_R_max,
            theta_max=theta_max,
            breakpoints=np.array(breakpoints),
        )
        return thetas, weights, info

    # ---------------- b_sel cache --------------------------------------------

    def _bsel_at(self, thetas, lob, zob, precomp):
        """One vectorised b_sel_marginalised call at all theta nodes."""
        return self.sel_bias.b_sel_marginalised(
            thetas, lob, zob, precomp=precomp)

    # ---------------- per-theta inner (z, M) integral ------------------------

    def _z_inner(self, R_vec, theta, zob, ctx):
        """Return (N_rnd[R], N_cl[R]) for one theta value.

        Exploits the fact that ``nfw._rs_and_rhos`` has no z-dependence
        (fixed concentration, rho_m is today's mean density), so at
        fixed theta Sigma_mis(M, R) is z-independent and the M-contraction
        can be done once and then weighted by the z-integrated HMF /
        HMF*bias quantities.
        """
        chi_z = ctx["chi_z"]
        outer_weight = ctx["outer_weight"]
        M_weight = ctx["M_weight"]
        n_mz = ctx["n_mz"]; bM_mz = ctx["bM_mz"]
        theta_excl_z = ctx["theta_excl_z"]
        chi_o = ctx["chi_o"]
        Sigma_mis_MR = ctx["Sigma_mis_per_theta"](theta)   # (NM, NR)

        cos_th = np.cos(theta)
        dchi = np.sqrt(np.maximum(
            chi_z ** 2 + chi_o ** 2 - 2.0 * chi_z * chi_o * cos_th, 0.0))
        xi_vals = self.xi_NL(dchi, zob)
        mask = theta > theta_excl_z
        xi_vals = np.where(mask, xi_vals, 0.0)

        w_rnd_M = M_weight * (n_mz * outer_weight[None, :]).sum(axis=1)
        w_cl_M = M_weight * (n_mz * bM_mz *
                              (outer_weight * xi_vals)[None, :]).sum(axis=1)
        N_rnd = w_rnd_M @ Sigma_mis_MR
        N_cl = w_cl_M @ Sigma_mis_MR
        return N_rnd, N_cl

    # ---------------- public -------------------------------------------------

    def _kernel_closure(self, R, ctx):
        """Return a callable ``kernel(theta) -> (NM, NR) Sigma_mis``.

        Subclasses override this to swap the underlying lookup spline
        (e.g. ``DeltaSigmaPrj`` substitutes ``_dsig_spl``).  The
        factor-of-2 / rs*rho_s prefactor is unchanged: both the Sigma
        and the Delta-Sigma tables share the same half-paper
        convention (see ``nfw.py`` module docstring).
        """
        rs_M = ctx["rs_M"]; rho_s = ctx["rho_s"]; D_A_o = ctx["D_A_o"]
        _spl = self.nfw._spl
        _lnx_lo = self.nfw._lnx_lo; _lnx_hi = self.nfw._lnx_hi
        _lnxmis_lo = self.nfw._lnxmis_lo; _lnxmis_hi = self.nfw._lnxmis_hi
        ln_R = np.log(R)[None, :] - np.log(rs_M)[:, None]   # (NM, NR)
        ln_R = np.clip(ln_R, _lnx_lo, _lnx_hi)
        prefac_M = (2.0 * (2.0 * np.pi) * rs_M * rho_s)     # (NM,)

        def kernel(theta):
            R_theta = theta * D_A_o
            lnxmis = np.log(R_theta / rs_M)                  # (NM,)
            lnxmis = np.clip(lnxmis, _lnxmis_lo, _lnxmis_hi)
            out = np.empty_like(ln_R)                        # (NM, NR)
            for iM in range(rs_M.size):
                out[iM] = prefac_M[iM] * np.exp(
                    _spl(lnxmis[iM:iM + 1], ln_R[iM])).ravel()
            return out
        return kernel

    def __call__(self, R, lob, zob, *, return_decomposition: bool = False):
        R = np.atleast_1d(R).astype(float)
        ctx = self._build_zM_context(lob, zob)
        pre = self.sel_bias.bias_precompute(lob, zob)
        thetas, w_theta, th_info = self._theta_grid(lob, zob, R, ctx)
        bsel_vals = self._bsel_at(thetas, lob, zob, pre)

        ctx["Sigma_mis_per_theta"] = self._kernel_closure(R, ctx)

        out_rnd = np.zeros_like(R)
        out_cl = np.zeros_like(R)
        for it, (th, wth) in enumerate(zip(thetas, w_theta)):
            N_rnd, N_cl = self._z_inner(R, th, zob, ctx)
            prefac = wth * 2.0 * np.pi * np.sin(th)
            out_rnd += prefac * N_rnd
            out_cl += prefac * bsel_vals[it] * N_cl

        if return_decomposition:
            return dict(
                total=out_rnd + out_cl,
                rnd=out_rnd, cl=out_cl,
                theta_info=th_info,
            )
        return out_cl
