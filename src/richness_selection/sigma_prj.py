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
- NFW convention: ``nfw.sigma_grid`` returns the C++ kernel value
  ``Sigma_mis = 2 * r_s * rho_eff * exp(ln f) * 1e-12`` in
  ``Msun/h / pc^2``  (see ``nfw.py`` module docstring).
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
from .survey_area import SurveyArea


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
    survey_area : SurveyArea
        Effective survey solid angle Omega(z), multiplied into the
        z-integral's ``outer_weight``.  Defaults to ``SurveyArea()``
        (``kind="unity"``, i.e. Omega(z)=1), reproducing this class's
        historical behaviour exactly -- see ``survey_area.py`` module
        docstring for why that is also the empirically-validated default.
    """

    def __init__(self, cosmo: Cosmology, sel_bias: SelBias,
                 nfw: NFWMiscentered,
                 n_theta_per_seg: int = 30,
                 R_max_cMpch: float = R_MAX_CMPCH,
                 survey_area: SurveyArea = SurveyArea(),
                 tmap: str = "DA",
                 closure: bool = False):
        if tmap not in ("DA", "comoving"):
            raise ValueError(f"tmap must be 'DA' or 'comoving', got {tmap!r}")
        if closure and tmap != "comoving":
            raise ValueError("closure=True requires tmap='comoving' (the "
                             "point-mass collapse of the counter-term is "
                             "exact only in the comoving map)")
        self.cosmo = cosmo
        self.sel_bias = sel_bias
        self.nfw = nfw
        self.hmf = sel_bias.hmf
        self.bias = sel_bias.bias
        self.grid = sel_bias.grid
        self.xi_NL = sel_bias.xi_NL
        self.n_theta_per_seg = int(n_theta_per_seg)
        self.R_max_cMpch = float(R_max_cMpch)
        self.survey_area = survey_area
        # theta <-> transverse map for the kernel offset and the theta_R
        # breakpoints: "DA" (default) R_mis = theta * D_A(zob), matching
        # the C++ port; "comoving" R_mis = theta * chi(zob), matching the
        # Costanzi notebook (docs/costanzi_notebook_diff.md item 4) and
        # the repo's own geometry.py convention.
        self.tmap = tmap
        # closure: halo-model sum-rule closure.  Resolved neighbours use
        # the mass-conserving NFW hard-truncated at r_t = c r_s (each
        # deposits exactly M), and the unresolved mass below the mass
        # cut enters as a smooth sheet with density
        # rho_u(z) = rho_m - int n M dM and bias
        # b_u(z) = (rho_m - int n b M dM) / rho_u (exact Tinker
        # consistency assumed for the complement).  The cl amplitude
        # then sums to rho_m * b_sel * xi by construction.
        self.closure = bool(closure)

    # ---------------- context ------------------------------------------------

    def _build_zM_context(self, lob, zob) -> dict:
        g = self.grid
        chi_o = float(self.cosmo.chi(zob))
        D_A_o = chi_o if self.tmap == "comoving" else chi_o / (1.0 + zob)
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

        # outer weight = wzs * dV * w_z(z) * Omega(z)
        outer_weight = wzs * dV * wz_kern * self.survey_area(zs)

        # Per-M NFW scale radii + rho_eff from the NFW object (C++ recipe:
        # r_200 via rho_crit, rho_eff = delta_c * rho_crit * rho_mult).
        # z-independent at fixed c, so cached here and reused across the
        # theta loop.
        rs_M, rho_eff_M = self.nfw._rs_and_rhos(Ms, zob)

        return dict(
            zs=zs, chi_z=chi_z, outer_weight=outer_weight,
            Ms=Ms, M_weight=M_weight, n_mz=n_mz, bM_mz=bM_mz,
            theta_excl_z=theta_excl_z,
            chi_o=chi_o, D_A_o=D_A_o, R_excl=R_excl,
            rs_M=rs_M, rho_s=rho_eff_M, zob=zob,
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
        """Marginalised b_sel(theta) at all theta nodes.

        Consumes the ``MarginalisedBias`` dataclass (eq. brm of
        docs/richness_selection_frozen.tex with the ltr-marginalised
        plateaus) -- identical to ``b_sel_marginalised`` and available
        on both the production ``SelBias`` and ``FrozenSelBias``.
        """
        return self.sel_bias.marginalised_bias(lob, zob, precomp=precomp)(thetas)

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
        (e.g. ``DeltaSigmaPrj`` substitutes ``_dsig_spl``).  C++ kernel
        prefactor ``2 * r_s * rho_eff * 1e-12`` (same for Sigma and
        DeltaSigma — see ``nfw.py`` module docstring).
        """
        rs_M = ctx["rs_M"]; rho_eff = ctx["rho_s"]; D_A_o = ctx["D_A_o"]
        _spl = self.nfw._spl
        _lnx_lo = self.nfw._lnx_lo; _lnx_hi = self.nfw._lnx_hi
        _lnxmis_lo = self.nfw._lnxmis_lo; _lnxmis_hi = self.nfw._lnxmis_hi
        ln_R = np.log(R)[None, :] - np.log(rs_M)[:, None]   # (NM, NR)
        ln_R = np.clip(ln_R, _lnx_lo, _lnx_hi)
        prefac_M = 2.0 * rs_M * rho_eff * 1.0e-12           # (NM,)  Msun/h/pc^2

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

    def _kernel_closure_trunc(self, R, ctx):
        """Truncated-NFW kernel for closure mode: the phi-averaged
        miscentered column of the profile hard-cut at r_t = c r_s
        (``nfw.truncated_sigma_kernel``), so each resolved neighbour
        deposits exactly its mass M."""
        from numpy.polynomial.legendre import leggauss
        from .nfw import truncated_sigma_kernel
        rs_M = ctx["rs_M"]; rho_eff = ctx["rho_s"]; D_A_o = ctx["D_A_o"]
        rho_eff = np.broadcast_to(np.atleast_1d(rho_eff), rs_M.shape)
        prefac_M = 2.0 * rs_M * rho_eff * 1.0e-12
        # per-M concentration (m200m kind: Duffy-like; cpp kind: fixed)
        if getattr(self.nfw, "kind", "cpp") == "m200m":
            Ms = ctx["Ms"]
            # must match _rs_and_rhos exactly: r_t = c * r_s = r_200m
            c_M = (10.14 * (Ms / 2.0e12) ** -0.081
                   * (1.0 + ctx["zob"]) ** -1.01)
        else:
            c_M = np.full(rs_M.shape, self.nfw.c)
        tables = [truncated_sigma_kernel(float(c)) for c in c_M]

        s, w = leggauss(128)
        s = 0.5 * (s + 1.0)
        wt = (0.5 * w * 2.0 * np.pi * s) / np.pi     # phi-average weights
        t = np.pi * s * s
        sin2 = np.sin(t / 2.0) ** 2
        x_R = R[None, :] / rs_M[:, None]             # (NM, NR)

        def kernel(theta):
            R_mis = theta * D_A_o
            out = np.empty((rs_M.size, R.size))
            for iM in range(rs_M.size):
                xm = R_mis / rs_M[iM]
                u = np.sqrt((x_R[iM][:, None] - xm) ** 2
                            + 4.0 * x_R[iM][:, None] * xm * sin2[None, :])
                lnxg, ftg = tables[iM]
                f_u = np.interp(np.log(np.maximum(u, 1e-30)), lnxg, ftg,
                                left=ftg[0], right=0.0)
                out[iM] = prefac_M[iM] * (f_u @ wt)
            return out
        return kernel

    def _closure_counter(self, R, zob, ctx, bsel_fn):
        """Unresolved-mass counter-term (smooth sheet, point-mass
        collapse in the comoving map): per z,
        rho_u b_u b_sel(theta_R) xi(r3d) with the pipeline's LoS
        weights and exclusion.  Returns (cl_counter, rnd_counter)."""
        from .nfw import RHO_CRIT_0
        chi_z = ctx["chi_z"]; chi_o = ctx["chi_o"]
        outer_weight = ctx["outer_weight"]
        theta_excl_z = ctx["theta_excl_z"]
        M_weight = ctx["M_weight"]
        n_mz = ctx["n_mz"]; bM_mz = ctx["bM_mz"]
        rho_m = float(self.cosmo.Om0) * RHO_CRIT_0
        Ms = ctx["Ms"]
        int_nM = (M_weight * Ms) @ n_mz              # (Nz,)  int n M dM
        int_nbM = (M_weight * Ms) @ (n_mz * bM_mz)   # (Nz,)
        rho_u = np.maximum(rho_m - int_nM, 0.0)
        b_u = np.where(rho_u > 0.0,
                       (rho_m - int_nbM) / np.maximum(rho_u, 1e-30), 0.0)

        theta_R = R / chi_o                          # comoving map
        bsel_R = bsel_fn(theta_R)
        cl = np.empty_like(R)
        rnd = np.empty_like(R)
        w_geo = outer_weight / chi_z ** 2 * 1.0e-12  # sheet -> Msun/h/pc^2
        for i, (Rv, th) in enumerate(zip(R, theta_R)):
            dchi = np.sqrt(np.maximum(
                chi_z ** 2 + chi_o ** 2
                - 2.0 * chi_z * chi_o * np.cos(th), 0.0))
            xiv = self.xi_NL(dchi, zob)
            xiv = np.where(th > theta_excl_z, xiv, 0.0)
            cl[i] = bsel_R[i] * np.sum(w_geo * rho_u * b_u * xiv)
            rnd[i] = np.sum(w_geo * rho_u)
        return cl, rnd

    def __call__(self, R, lob, zob, *, return_decomposition: bool = False):
        R = np.atleast_1d(R).astype(float)
        ctx = self._build_zM_context(lob, zob)
        pre = self.sel_bias.bias_precompute(lob, zob)
        thetas, w_theta, th_info = self._theta_grid(lob, zob, R, ctx)
        bsel_vals = self._bsel_at(thetas, lob, zob, pre)

        ctx["Sigma_mis_per_theta"] = (
            self._kernel_closure_trunc(R, ctx) if self.closure
            else self._kernel_closure(R, ctx))

        out_rnd = np.zeros_like(R)
        out_cl = np.zeros_like(R)
        for it, (th, wth) in enumerate(zip(thetas, w_theta)):
            N_rnd, N_cl = self._z_inner(R, th, zob, ctx)
            prefac = wth * 2.0 * np.pi * np.sin(th)
            out_rnd += prefac * N_rnd
            out_cl += prefac * bsel_vals[it] * N_cl

        if self.closure:
            bsel_fn = self.sel_bias.marginalised_bias(lob, zob, precomp=pre)
            cl_c, rnd_c = self._closure_counter(R, zob, ctx, bsel_fn)
            out_cl += cl_c
            out_rnd += rnd_c

        if return_decomposition:
            return dict(
                total=out_rnd + out_cl,
                rnd=out_rnd, cl=out_cl,
                theta_info=th_info,
            )
        return out_cl
