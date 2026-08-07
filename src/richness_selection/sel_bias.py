"""Optical cluster bias b_lob(theta | lob, ltr, zob) — Costanzi 2026.

This is a clean implementation of Costanzi et al. 2026 equations 3-9.
It does NOT follow Matteo's selbias notebook cell 16 (which uses a
different intrinsic-bias construction that requires two undefined
constants `inter_boost_bias` / `slope_boost_bias`).  Instead we stick
to the published paper:

  rho_prj(lam, z, theta | lob, zob) = w_z(z, zob) f_A(theta, lam, z, lob, zob) lam

      P[X]       = int dz (dV/dzdOmega) int dM n(M,z) int d lam P(lam|M,z)
                   * 2 pi int dtheta sin(theta) rho_prj(lam, z, theta | lob, zob)
                   * X(M, z, theta | zob)

  <Delta_prj_bkg> = P[1]
  I1             = P[ b(M,z) xi_NL(z,zob,theta) sigmoid(theta) ]
  I2             = P[ b(M,z) xi_NL(z,zob,theta) ]
  Delta_prj_RND  = <Delta_prj_bkg> + b_eff(lob,zob) * I2   (eq. 3 with b_lob -> b_eff)

  b_infty        = b_eff(lob,zob) * (1 + 0.13 delta_prj)       (eq. 7)
  delta_prj      = (lob - ltr)/Delta_prj_RND - 1
  b_zero         = [(lob - ltr) - <Delta_prj_bkg> - b_infty*I1] / (I2 - I1)  (eq. 8)

  b_lob(theta)   = b_zero + (b_infty - b_zero) / (1 + exp(-k(theta - theta0)))
                   with  k = 2.5/theta_lob,  theta0 = theta_lob/2      (eq. 6)

The angular f_A is the true theta-dependent aperture-overlap fraction
(Matteo `area_overlap` in cell 16), NOT the closed-form
(1 + theta_ltr/theta_lob)^(-2) from cell 14 (which is already the
theta-integrated version and would double-count the theta integral).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import bisect

from .cosmology import Cosmology
from .pk import PkGrid
from .hmf import HMF
from .bias import Bias
from .mor import MOR
from .geometry import R_lambda, theta_lambda, area_overlap
from .photoz import w_z, sigma_z, zmin4zkernel, zmax4zkernel
from .gl import gl_nodes
from .config import DEFAULT_GRID, GridConfig


@dataclass(frozen=True)
class BiasPlateaus:
    """lambda_tr-marginalised selection-bias plateaus at one (lob, zob).

    Per-ltr quantities are vectors on the GL marginalisation grid
    (eqs. bls + bss of docs/richness_selection_frozen.tex evaluated on
    the whole grid at once); the only scalars are the GL-marginalised
    plateaus, sum(w_ltr * b_rm_X_ltr_vec).
    """
    lob: float
    zob: float
    ltr: np.ndarray = field(repr=False)            # GL nodes over lambda_tr
    w_ltr: np.ndarray = field(repr=False)          # normalised GL x P(ltr|lob,zob)
    delta_prj: np.ndarray = field(repr=False)      # per-ltr, eq. bls
    b_rm_ss_ltr_vec: np.ndarray = field(repr=False)   # per-ltr eq. bss
    b_rm_ls_ltr_vec: np.ndarray = field(repr=False)   # per-ltr eq. bls
    b_rm_ss: float = float("nan")                  # marginalised small-scale
    b_rm_ls: float = float("nan")                  # marginalised large-scale


@dataclass(frozen=True)
class MarginalisedBias:
    """The marginalised profile b_sel(theta | lob, zob) -- eq. brm.

    This is the object the lensing pipeline consumes: SigmaPrj /
    DeltaSigmaPrj evaluate it on their theta grid.  Built from the
    marginalised plateaus of ``BiasPlateaus`` and the fixed sigmoid
    (k = damping/theta_lam_ob, theta0 = theta0_frac * theta_lam_ob).
    """
    lob: float
    zob: float
    theta_lam_ob: float
    b_rm_ss: float
    b_rm_ls: float
    damping: float = 2.5
    theta0_frac: float = 0.5

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        k = self.damping / self.theta_lam_ob
        theta0 = self.theta0_frac * self.theta_lam_ob
        s = 1.0 / (1.0 + np.exp(-k * (theta - theta0)))
        return self.b_rm_ss + (self.b_rm_ls - self.b_rm_ss) * s


class SelBias:
    """Optical cluster bias per Costanzi 2026 eq. 3-9."""

    # Paper constants (eq. 6, section II.A)
    damping: float = 2.5           # k = damping / theta_lob
    theta0_frac: float = 0.5       # theta0 = theta0_frac * theta_lob
    boost_slope: float = 0.13      # eq. 7 coefficient
    exclusion: bool = True

    def __init__(self, cosmo: Cosmology, pk: PkGrid, hmf: HMF, bias: Bias,
                 mor: MOR, xi_nl, grid: GridConfig = DEFAULT_GRID,
                 min_mass4integral: float = 1.0e13,
                 ln_M_max_log10: float = 15.5,
                 n_ltr: int = 60):
        """z-axis integration: split by physics into ring + outer-fg
        + outer-bg, GL on z for the ring and GL on ln|Delta chi| for
        the outer halves.  Sub-0.01% precision on P[1], I_1, I_2 at
        Nz=80 in ~26 ms.  The theta-axis is always split at the
        exclusion boundary (per-z theta_excl(z) lower limit of the GL
        interval).
        """
        self.cosmo = cosmo
        self.pk = pk
        self.hmf = hmf
        self.bias = bias
        self.mor = mor
        self.xi_NL = xi_nl
        self.grid = grid
        self.min_mass4integral = min_mass4integral
        self.ln_M_max_log10 = ln_M_max_log10
        self.n_ltr = n_ltr
        self._cache: dict = {}

    # ---------------- small helpers -----------------------------------

    def _theta_lob(self, lob, zob):
        return float(theta_lambda(lob, zob, self.cosmo))

    def _sigmoid_theta(self, theta, lob, zob):
        theta_lob = self._theta_lob(lob, zob)
        k = self.damping / theta_lob
        theta0 = self.theta0_frac * theta_lob
        return 1.0 / (1.0 + np.exp(-k * (theta - theta0)))

    # ---------------- f_P1(z): X=1 specialisation, no bias/xi_NL -------
    #
    # Standalone, additive helper for testing alternative z-axis
    # quadratures against P[1] in isolation (X=1 needs neither bias(M,z)
    # nor xi_NL, so this is cheaper per-z than the full P_operator loop).
    # Does not touch _P_operator; same theta_excl(z) split-at-exclusion
    # convention so f_P1 matches P[1]'s per-z contribution exactly.

    def _f_p1(self, zs, lob, zob):
        """f_P1(z) = dV/dzdOmega(z) * inner(M,lambda,theta; X=1) at each z.

        int dz f_P1(z) == P[1] from _P_operator (same integrand, just
        evaluated at caller-chosen z nodes instead of the ring+outer grid).
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        Nz = zs.size
        theta_lob = self._theta_lob(lob, zob)
        chi_o = float(self.cosmo.chi(zob))
        R_excl = R_lambda(lob) * (1.0 + zob)
        chi_z = self.cosmo.chi(zs)
        dV = self.cosmo.dV_dzdOm(zs)
        wz_kern = w_z(zs, zob)

        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = np.clip(cos_excl, -1.0, 1.0)
        theta_excl_z = np.arccos(cos_excl)
        eps_theta = 1e-6
        no_excl = cos_excl >= 1.0 - 1e-12
        theta_excl_z = np.where(no_excl, eps_theta, theta_excl_z)
        theta_max = 2.0 * theta_lob

        lam_grid, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)
        log10_Mmin = np.log10(self.min_mass4integral)
        ln_M_min = np.log(10.0 ** log10_Mmin)
        ln_M_max = np.log(10.0 ** self.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, self.grid.NM)
        Ms = np.exp(lnMs)
        M_weight = wM * Ms

        n_mz = self.hmf(Ms[:, None], zs[None, :])
        p_lmz = self.mor.pdf(lam_grid[:, None, None], Ms[None, :, None],
                             zs[None, None, :])

        Nth = self.grid.Nth
        F = np.zeros(Nz)
        for iz in range(Nz):
            th_lo = max(theta_excl_z[iz], eps_theta)
            if th_lo >= theta_max:
                continue
            ths_z, wth_z = gl_nodes(th_lo, theta_max, Nth)
            sin_th_z = np.sin(ths_z)
            th_weight_z = wth_z * 2.0 * np.pi * sin_th_z
            theta_lam_l_z = R_lambda(lam_grid) * (1.0 + zs[iz]) / chi_z[iz]
            fA_z = area_overlap(ths_z, theta_lob, theta_lam_l_z)
            ang_P1_z = np.einsum('t,tL->L', th_weight_z, fA_z)
            rho_prefac_z = wz_kern[iz] * lam_grid
            p_lm_z = p_lmz[:, :, iz]
            lam_int_P1_z = np.einsum('L,LM,L->M', wlam, p_lm_z,
                                      rho_prefac_z * ang_P1_z)
            F[iz] = np.sum(M_weight * n_mz[:, iz] * lam_int_P1_z)
        return dV * F

    # ---------------- f_I1(z), f_I2(z): full X=b*xi[*sigma] -----------
    #
    # Same additive-helper pattern as _f_p1, for testing alternative
    # z-axis quadratures against I1/I2 in isolation. xi_NL is clipped to
    # >=0: it goes negative over the BAO-trough tail of the outer decay
    # range at higher zob (checked numerically), which is a physically
    # marginal effect (order 1e-4, in a region already suppressed by
    # w_z->0) but breaks any log-space treatment of the integrand, so we
    # drop it here rather than carry it through.  Does not touch
    # _P_operator.

    def _f_i1_i2(self, zs, lob, zob, clip_negative_xi=True):
        """Return (f_I1(z), f_I2(z)) arrays at the given z nodes.

        int dz f_I2(z) == I2, int dz f_I1(z) == I1 from _P_operator
        (same integrand, evaluated at caller-chosen z nodes).
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        Nz = zs.size
        theta_lob = self._theta_lob(lob, zob)
        chi_o = float(self.cosmo.chi(zob))
        R_excl = R_lambda(lob) * (1.0 + zob)
        chi_z = self.cosmo.chi(zs)
        dV = self.cosmo.dV_dzdOm(zs)
        wz_kern = w_z(zs, zob)

        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = np.clip(cos_excl, -1.0, 1.0)
        theta_excl_z = np.arccos(cos_excl)
        eps_theta = 1e-6
        no_excl = cos_excl >= 1.0 - 1e-12
        theta_excl_z = np.where(no_excl, eps_theta, theta_excl_z)
        theta_max = 2.0 * theta_lob

        lam_grid, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)
        log10_Mmin = np.log10(self.min_mass4integral)
        ln_M_min = np.log(10.0 ** log10_Mmin)
        ln_M_max = np.log(10.0 ** self.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, self.grid.NM)
        Ms = np.exp(lnMs)
        M_weight = wM * Ms

        bM_mz = self.bias(Ms[:, None], zs[None, :])
        n_mz = self.hmf(Ms[:, None], zs[None, :])
        p_lmz = self.mor.pdf(lam_grid[:, None, None], Ms[None, :, None],
                             zs[None, None, :])

        Nth = self.grid.Nth
        F1 = np.zeros(Nz)
        F2 = np.zeros(Nz)
        for iz in range(Nz):
            th_lo = max(theta_excl_z[iz], eps_theta)
            if th_lo >= theta_max:
                continue
            ths_z, wth_z = gl_nodes(th_lo, theta_max, Nth)
            sin_th_z = np.sin(ths_z)
            sigmoid_z = self._sigmoid_theta(ths_z, lob, zob)
            th_weight_z = wth_z * 2.0 * np.pi * sin_th_z
            cos_th_z = np.cos(ths_z)
            dchi_z = np.sqrt(np.maximum(
                chi_z[iz] ** 2 + chi_o ** 2
                - 2.0 * chi_z[iz] * chi_o * cos_th_z, 0.0))
            xi_z = self.xi_NL(dchi_z, zob)
            if clip_negative_xi:
                xi_z = np.maximum(xi_z, 0.0)
            theta_lam_l_z = R_lambda(lam_grid) * (1.0 + zs[iz]) / chi_z[iz]
            fA_z = area_overlap(ths_z, theta_lob, theta_lam_l_z)

            ang_I2_z = np.einsum('t,tL,t->L', th_weight_z, fA_z, xi_z)
            ang_I1_z = np.einsum('t,t,tL,t->L', th_weight_z, sigmoid_z,
                                  fA_z, xi_z)

            rho_prefac_z = wz_kern[iz] * lam_grid
            p_lm_z = p_lmz[:, :, iz]
            lam_int_I2_z = np.einsum('L,LM,L->M', wlam, p_lm_z,
                                      rho_prefac_z * ang_I2_z)
            lam_int_I1_z = np.einsum('L,LM,L->M', wlam, p_lm_z,
                                      rho_prefac_z * ang_I1_z)

            F2[iz] = np.sum(M_weight * n_mz[:, iz] * bM_mz[:, iz] * lam_int_I2_z)
            F1[iz] = np.sum(M_weight * n_mz[:, iz] * bM_mz[:, iz] * lam_int_I1_z)
        return dV * F1, dV * F2

    # ---------------- b_eff(lob, zob) ---------------------------------

    def b_eff(self, lob, zob):
        """b_eff(lob, zob) = int dM P(M|lob,zob) b(M,zob)  (paper eq. 7 caption).

        Reweights halo bias by the mass distribution given observed
        richness.  Uses HOD P(ltr=lob | M, z) as a proxy for
        P(lob|M,z) at first pass (Poisson-Gaussian convolved over noise
        is the full story; that's Matteo's `plob_Mz`, but for Tinker
        bias evaluated on M, the main sensitivity is the peak mass
        which both proxies capture).
        """
        key = ("b_eff", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        log10_Mmin = np.log10(self.min_mass4integral)
        m_grid = 10.0 ** np.linspace(log10_Mmin, self.ln_M_max_log10, 100)
        n_m = self.hmf(m_grid, zob)
        b_m = self.bias(m_grid, zob)
        P = self.mor.pdf(np.array([float(lob)])[:, None],
                         m_grid[None, :], zob).ravel()
        wt = n_m * P * m_grid
        num = np.trapezoid(wt * b_m, np.log(m_grid))
        den = np.trapezoid(wt, np.log(m_grid))
        val = float(num / den) if den > 0 else float("nan")
        self._cache[key] = val
        return val

    # ---------------- z-grid builder ----------------------------------

    def _z_grid(self, lob, zob, Nz, chi_o, R_excl,
                z_fg_lo, z_bg_hi,
                zs_ref, chi_ref, dchi_dz_ref):
        """Split the z-axis by physics: ring R1 + outer fg/bg (R2+R3).

        R1 [|z-zob| < dz_excl]:         GL in z, n_ring = max(9, Nz/4)
        R2+R3 outer regions per side:   GL in u = ln|Delta_chi_par|,
                                        n_outer = max(15, (Nz-n_ring)/2)

        Note: adding a core/wing split within the outer region does NOT
        improve precision at Nz=80 -- the bottleneck is the Nth=20 theta
        staircase inside the ring (exclusion boundary moves discretely
        across theta-GL nodes as z walks through the ring).  Improving
        precision beyond ~0.3% on I1, I2 requires raising Nth, not
        reshuffling the z-grid.
        """
        dchi_dz_at_zob = float(np.interp(zob, zs_ref, dchi_dz_ref))
        dz_excl = R_excl / dchi_dz_at_zob

        n_ring = max(9, Nz // 4)
        z_ring_lo = max(zob - dz_excl, z_fg_lo)
        z_ring_hi = min(zob + dz_excl, z_bg_hi)
        z_ring, w_ring = gl_nodes(z_ring_lo, z_ring_hi, n_ring)

        n_outer = max(15, (Nz - n_ring) // 2)
        dis_fg_max = chi_o - float(self.cosmo.chi(z_fg_lo))
        dis_bg_max = float(self.cosmo.chi(z_bg_hi)) - chi_o

        if R_excl < dis_fg_max:
            u_fg, w_u_fg = gl_nodes(np.log(R_excl), np.log(dis_fg_max), n_outer)
            dis_fg = np.exp(u_fg)
            z_fg = np.interp(chi_o - dis_fg, chi_ref, zs_ref)
            dchi_dz_fg = np.interp(z_fg, zs_ref, dchi_dz_ref)
            w_z_fg = w_u_fg * dis_fg / dchi_dz_fg
        else:
            z_fg = np.array([]); w_z_fg = np.array([])

        if R_excl < dis_bg_max:
            u_bg, w_u_bg = gl_nodes(np.log(R_excl), np.log(dis_bg_max), n_outer)
            dis_bg = np.exp(u_bg)
            z_bg = np.interp(chi_o + dis_bg, chi_ref, zs_ref)
            dchi_dz_bg = np.interp(z_bg, zs_ref, dchi_dz_ref)
            w_z_bg = w_u_bg * dis_bg / dchi_dz_bg
        else:
            z_bg = np.array([]); w_z_bg = np.array([])

        z_fg_sort = z_fg[::-1] if z_fg.size else z_fg
        w_fg_sort = w_z_fg[::-1] if w_z_fg.size else w_z_fg
        zs = np.concatenate([z_fg_sort, z_ring, z_bg])
        wzs = np.concatenate([w_fg_sort, w_ring, w_z_bg])
        return zs, wzs


    # ---------------- Operator P[X] (eq. 9) ---------------------------
    #
    # Returns three scalars: P[1], I1, I2 (eqs 8/9).
    # This is the central routine and is deliberately *not* cached past
    # (lob, zob) since repeat calls within a likelihood step are rare.

    def _P_operator(self, lob, zob):
        """Compute (P[1], I1, I2) at (lob, zob) per equations 8-9.

        Grids:
            z     -- photo-z kernel half-width around zob (5 sigma)
            lnM   -- halo mass
            lam   -- projected halo true richness (0, lob]
            theta -- angular separation in (0, 2 theta_lob]
        """
        g = self.grid
        theta_lob = self._theta_lob(lob, zob)

        # LoS-distance grid (Matteo-style: log-spaced in |Delta chi|,
        # dense near z_ob where xi_NL spikes).
        # Foreground + background each log-spaced in Delta chi.
        chi_o = float(self.cosmo.chi(zob))

        # Support bounds from the photo-z kernel bisect helpers
        try:
            z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
            z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
        except ValueError:
            # bisect can fail at very small zob; fall back to a symmetric band
            sig = float(sigma_z(zob))
            z_fg_lo, z_bg_hi = max(0.01, zob - sig), zob + sig

        chi_fg_lo = float(self.cosmo.chi(z_fg_lo))
        chi_bg_hi = float(self.cosmo.chi(z_bg_hi))

        # Build the z grid + weights (ring + outer-fg + outer-bg split).
        R_excl = R_lambda(lob) * (1.0 + zob)
        zs_ref = np.linspace(0.0, 2.0, 2000)
        chi_ref = self.cosmo.chi(zs_ref)
        dchi_dz_ref = np.gradient(chi_ref, zs_ref)

        zs, wzs = self._z_grid(lob, zob, g.Nz_bias,
                               chi_o, R_excl,
                               z_fg_lo, z_bg_hi,
                               zs_ref, chi_ref, dchi_dz_ref)

        chi_z = self.cosmo.chi(zs)
        dV = self.cosmo.dV_dzdOm(zs)
        wz_kern = w_z(zs, zob)

        # M-grid
        log10_Mmin = np.log10(self.min_mass4integral)
        ln_M_min = np.log(10.0 ** log10_Mmin)
        ln_M_max = np.log(10.0 ** self.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, g.NM)          # (NM,)
        Ms = np.exp(lnMs)

        # lambda (true richness) grid: eq. 3 runs lam over (0, lob]
        lam_grid, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)  # (Nlam,)

        # theta grid: SPLIT-AT-EXCLUSION per z.  The exclusion mask
        # xi_NL = 0 for Delta_chi < R_excl creates a hard step at
        # theta_excl(z); placing a GL grid that STARTS at theta_excl(z)
        # eliminates the step from the integrand and reaches converged
        # precision at Nth ~ 10 instead of needing Nth > 200 with the
        # fixed (0, 2 theta_lob) grid + mask.
        #
        # theta_excl(z) from cos(theta_excl) = (chi_z^2 + chi_o^2 - R_excl^2) / (2 chi_z chi_o)
        # (clipped to [-1,1]).  When the argument > 1, |Delta_chi_par| > R_excl
        # so no exclusion: theta_excl = 0, integrate the full (eps, 2 theta_lob).
        theta_max = 2.0 * theta_lob
        eps_theta = 1e-6

        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = np.clip(cos_excl, -1.0, 1.0)
        theta_excl_z = np.arccos(cos_excl)                      # (Nz,)
        # Where |Delta_chi_par| > R_excl, cos_excl > 1 before clip -> clip
        # sets it to 1 -> theta_excl = 0.  Identify that case and set to eps.
        no_excl = cos_excl >= 1.0 - 1e-12
        theta_excl_z = np.where(no_excl, eps_theta, theta_excl_z)
        # If no exclusion applies AT ALL (all z outside exclusion band),
        # fall back to the old fixed-grid path; rare but check.

        Nth = g.Nth
        Nlam = lam_grid.size
        Nz = zs.size

        # M-z grids
        bM_mz = self.bias(Ms[:, None], zs[None, :])            # (NM, Nz)
        n_mz = self.hmf(Ms[:, None], zs[None, :])              # (NM, Nz)
        p_lmz = self.mor.pdf(lam_grid[:, None, None],
                             Ms[None, :, None],
                             zs[None, None, :])                # (Nlam, NM, Nz)
        M_weight = wM * Ms                                      # (NM,)

        # Loop over z: each z has its own theta grid from theta_excl(z)
        # to 2*theta_lob.  Inside the loop we do the same contractions
        # as before but on a per-z theta.
        P1_per_z = np.zeros(Nz)
        I2_per_z = np.zeros(Nz)
        I1_per_z = np.zeros(Nz)

        for iz in range(Nz):
            th_lo = max(theta_excl_z[iz], eps_theta)
            if th_lo >= theta_max:
                # fully excluded at this z, integrand is zero
                continue
            ths_z, wth_z = gl_nodes(th_lo, theta_max, Nth)
            sin_th_z = np.sin(ths_z)
            sigmoid_z = self._sigmoid_theta(ths_z, lob, zob)
            th_weight_z = wth_z * 2.0 * np.pi * sin_th_z        # (Nth,)
            cos_th_z = np.cos(ths_z)

            # 3-D separation at this z
            dchi_z = np.sqrt(np.maximum(
                chi_z[iz] ** 2 + chi_o ** 2
                - 2.0 * chi_z[iz] * chi_o * cos_th_z, 0.0))     # (Nth,)
            xi_z = self.xi_NL(dchi_z, zob)                      # (Nth,)
            # No mask needed: theta_lo = theta_excl(z) already excludes.

            # f_A(theta, lam) at this z
            theta_lam_l_z = R_lambda(lam_grid) * (1.0 + zs[iz]) / chi_z[iz]
            fA_z = area_overlap(ths_z, theta_lob, theta_lam_l_z)  # (Nth, Nlam)

            # Angular integrals at this z (contract theta)
            ang_P1_z = np.einsum('t,tL->L', th_weight_z, fA_z)                # (Nlam,)
            ang_I2_z = np.einsum('t,tL,t->L', th_weight_z, fA_z, xi_z)        # (Nlam,)
            ang_I1_z = np.einsum('t,t,tL,t->L', th_weight_z, sigmoid_z,
                                  fA_z, xi_z)                                 # (Nlam,)

            # lambda integral (contract L)
            rho_prefac_z = wz_kern[iz] * lam_grid                             # (Nlam,)
            p_lm_z = p_lmz[:, :, iz]                                          # (Nlam, NM)
            lam_int_P1_z = np.einsum('L,LM,L->M', wlam, p_lm_z,
                                      rho_prefac_z * ang_P1_z)                # (NM,)
            lam_int_I2_z = np.einsum('L,LM,L->M', wlam, p_lm_z,
                                      rho_prefac_z * ang_I2_z)                # (NM,)
            lam_int_I1_z = np.einsum('L,LM,L->M', wlam, p_lm_z,
                                      rho_prefac_z * ang_I1_z)                # (NM,)

            # M integral (contract M)
            P1_per_z[iz] = np.sum(M_weight * n_mz[:, iz] * lam_int_P1_z)
            I2_per_z[iz] = np.sum(M_weight * n_mz[:, iz] * bM_mz[:, iz] * lam_int_I2_z)
            I1_per_z[iz] = np.sum(M_weight * n_mz[:, iz] * bM_mz[:, iz] * lam_int_I1_z)

        # Outer z-integral: sum (wz * dV * per_z)
        P1 = float(np.sum(wzs * dV * P1_per_z))
        I2 = float(np.sum(wzs * dV * I2_per_z))
        I1 = float(np.sum(wzs * dV * I1_per_z))

        return P1, I1, I2

    # ---------------- precompute / per-ltr assembly -------------------

    def bias_precompute(self, lob, zob):
        """Compute (P1, I1, I2, b_eff, Delta_prj_RND) at (lob, zob) once."""
        key = ("pre", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        P1, I1, I2 = self._P_operator(lob, zob)
        beff = self.b_eff(lob, zob)
        Delta_RND = P1 + beff * I2
        pre = dict(lob=lob, zob=zob,
                   P1=P1, I1=I1, I2=I2,
                   b_eff=beff, Delta_RND=Delta_RND,
                   denom=I2 - I1)
        self._cache[key] = pre
        return pre

    def bias_from_precomp(self, precomp, ltr):
        """Return (b_zero, b_infty, delta_prj) at this ltr."""
        P1 = precomp["P1"]; I1 = precomp["I1"]; I2 = precomp["I2"]
        beff = precomp["b_eff"]; D_RND = precomp["Delta_RND"]
        denom = precomp["denom"]; lob = precomp["lob"]

        delta_prj = (lob - ltr) / D_RND - 1.0
        b_infty = beff * (1.0 + self.boost_slope * delta_prj)
        if abs(denom) < 1e-12 * (abs(I1) + abs(I2) + 1e-30):
            b_zero = b_infty
        else:
            b_zero = ((lob - ltr) - P1 - b_infty * I1) / denom
        return dict(delta_prj=delta_prj, b_zero=b_zero, b_infty=b_infty)

    def bias_pipeline(self, lob, zob, ltr):
        pre = self.bias_precompute(lob, zob)
        out = self.bias_from_precomp(pre, ltr)
        out.update(pre)
        out["ltr"] = ltr
        return out

    # ---------------- b_lob(theta) (eq. 6) ----------------------------

    def b_lob_theta(self, theta, ltr, zob, lob, precomp=None):
        """Scale-dependent optical cluster bias b_lob(theta | lob, ltr, zob)."""
        pre = precomp if precomp is not None else self.bias_precompute(lob, zob)
        pr = self.bias_from_precomp(pre, ltr)
        s = self._sigmoid_theta(np.asarray(theta, dtype=float), lob, zob)
        return pr["b_zero"] + (pr["b_infty"] - pr["b_zero"]) * s

    # back-compat aliases matching the old notebook API
    def b_sel_of_theta(self, theta, lob, zob, ltr, precomp=None):
        return self.b_lob_theta(theta, ltr, zob, lob, precomp=precomp)

    def b_sel_lob_ltr_theta(self, theta, ltr, zcl, lob):
        return self.b_lob_theta(theta, ltr, zcl, lob)

    def eff_bias_ltr(self, ltr, zcl):
        """Alias for b_eff(ltr, zcl) (keeps old test names working)."""
        return self.b_eff(ltr, zcl)

    def _marginalised_plateaus(self, lob, zob, precomp,
                               ltr_grid_size, use_plob_ltr):
        """Return ``(b_zero_bar, b_infty_bar)``: the plateau averages
        of ``bias_from_precomp`` under ``P(ltr | lob, zob)``.

        Cached per ``(lob, zob, ltr_grid_size, use_plob_ltr)`` because
        the underlying plateau integrals are ltr-only (no theta
        dependence), so ``b_sel_marginalised(theta, ...)`` only has to
        evaluate the analytic sigmoid once per call rather than an
        ltr-loop per theta-node.  The per-ltr ``b_zero`` / ``b_infty``
        closures are the same ones returned by ``bias_from_precomp``.

        Derivation: ``b_lob_theta(theta | ltr)`` = b_zero(ltr)(1-sigma)
        + b_infty(ltr) sigma, and sigma(theta) is ltr-independent, so
        ``int dltr P(ltr) b_lob_theta(theta | ltr) = b_zero_bar
        (1 - sigma(theta)) + b_infty_bar sigma(theta)``.  Formalised
        in ``docs/richness_selection.tex`` eq.~\\eqref{eq:b_marg_sigmoid}.
        """
        p = self.plateaus(lob, zob, ltr_grid_size=ltr_grid_size,
                          precomp=precomp, use_plob_ltr=use_plob_ltr)
        return (p.b_rm_ss, p.b_rm_ls)

    # ---------------- vectorised closure + marginalised plateaus ------
    #
    # Shared by the production and frozen methods (only
    # ``bias_precompute`` differs underneath): the per-ltr closure
    # eqs. (bls) + (bss) of docs/richness_selection_frozen.tex,
    # evaluated on the whole lambda_tr GL grid at once, and the
    # GL-marginalised plateaus b_rm_ss / b_rm_ls built from them.

    def _closure_ltr_vec(self, precomp, ltr_vec):
        """Vectorised ``bias_from_precomp``: eqs. (bls) + (bss) on an
        ltr array.  Returns (delta_prj, b_rm_ss_ltr_vec, b_rm_ls_ltr_vec)."""
        ltr_vec = np.asarray(ltr_vec, dtype=float)
        P1 = precomp["P1"]; I1 = precomp["I1"]; I2 = precomp["I2"]
        beff = precomp["b_eff"]; D_RND = precomp["Delta_RND"]
        denom = precomp["denom"]; lob = precomp["lob"]

        delta = (lob - ltr_vec) / D_RND - 1.0
        b_ls = beff * (1.0 + self.boost_slope * delta)
        if abs(denom) < 1e-12 * (abs(I1) + abs(I2) + 1e-30):
            b_ss = b_ls.copy()
        else:
            b_ss = ((lob - ltr_vec) - P1 - b_ls * I1) / denom
        return delta, b_ss, b_ls

    def _ltr_marginalisation_grid(self, lob, zob, ltr_grid_size,
                                  use_plob_ltr):
        """(ltr nodes, normalised GL x P(ltr | lob, zob) weights)."""
        t_nodes, t_wts = gl_nodes(1.0, 3.0 * float(lob), ltr_grid_size * 2)
        log10_Mmin = np.log10(self.min_mass4integral)
        m_grid = 10.0 ** np.linspace(log10_Mmin, self.ln_M_max_log10, 50)
        hmf_m = self.hmf(m_grid, zob)
        p_ltr_M = self.mor.pdf(t_nodes[:, None], m_grid[None, :], zob)
        prior_ltr = np.trapezoid(p_ltr_M * (hmf_m * m_grid)[None, :],
                                 np.log(m_grid), axis=1)
        if use_plob_ltr:
            from .plob_ltr import P_lob_given_ltr
            p_lob_ltr = np.array([float(P_lob_given_ltr(lob, float(t), zob))
                                  for t in t_nodes])
            p_ltr = p_lob_ltr * prior_ltr
        else:
            p_ltr = prior_ltr
        weight = t_wts * p_ltr
        den = float(np.sum(weight))
        w_norm = weight / den if den > 0 else np.full_like(weight, np.nan)
        return t_nodes, w_norm

    def plateaus(self, lob, zob, ltr_grid_size=None, precomp=None,
                 use_plob_ltr: bool = True) -> BiasPlateaus:
        """lambda_tr-marginalised plateaus at (lob, zob) -> BiasPlateaus.

        Per-ltr vectors (eqs. bls, bss on the GL grid) plus the
        marginalised scalars b_rm_ss / b_rm_ls.  Cached per
        (lob, zob, grid, convention, operator values).
        """
        if ltr_grid_size is None:
            ltr_grid_size = self.grid.ltr_grid_size
        pre = precomp if precomp is not None else self.bias_precompute(lob, zob)
        cache_key = ("brm_plateaus", float(lob), float(zob),
                     int(ltr_grid_size), bool(use_plob_ltr),
                     float(pre["P1"]), float(pre["I1"]), float(pre["I2"]),
                     float(pre["Delta_RND"]))
        if cache_key in self._cache:
            return self._cache[cache_key]

        ltr, w_ltr = self._ltr_marginalisation_grid(
            lob, zob, ltr_grid_size, use_plob_ltr)
        delta, b_ss_vec, b_ls_vec = self._closure_ltr_vec(pre, ltr)
        val = BiasPlateaus(
            lob=float(lob), zob=float(zob), ltr=ltr, w_ltr=w_ltr,
            delta_prj=delta,
            b_rm_ss_ltr_vec=b_ss_vec, b_rm_ls_ltr_vec=b_ls_vec,
            b_rm_ss=float(np.sum(w_ltr * b_ss_vec)),
            b_rm_ls=float(np.sum(w_ltr * b_ls_vec)))
        self._cache[cache_key] = val
        return val

    def marginalised_bias(self, lob, zob, ltr_grid_size=None, precomp=None,
                          use_plob_ltr: bool = True) -> MarginalisedBias:
        """The theta-callable b_sel(theta | lob, zob) -- eq. brm with the
        marginalised plateaus.  This is the object SigmaPrj consumes."""
        p = self.plateaus(lob, zob, ltr_grid_size=ltr_grid_size,
                          precomp=precomp, use_plob_ltr=use_plob_ltr)
        return MarginalisedBias(
            lob=float(lob), zob=float(zob),
            theta_lam_ob=self._theta_lob(lob, zob),
            b_rm_ss=p.b_rm_ss, b_rm_ls=p.b_rm_ls,
            damping=self.damping, theta0_frac=self.theta0_frac)

    def b_rm(self, theta, lob, zob, **kwargs):
        """Marginalised profile b_sel(theta, lob, zob) (eq. brm)."""
        return self.marginalised_bias(lob, zob, **kwargs)(theta)

    def b_sel_marginalised(self, theta, lob, zob, ltr_grid_size=None,
                           precomp=None, use_plob_ltr: bool = True):
        """Marginalised bias per Eq. (b_marg_lt) of the TeX.

        P(ltr | lob, zob) propto P(lob | ltr, zob) * prior(ltr, zob),
        with prior(ltr, zob) = int dM n(M,z) P(ltr | M, z).

        P(lob | ltr, z) is the full EMG kernel of
        Eq. (costanzi_kernel):  (1 - f_prj) Gaussian + f_prj * EMG.
        This is smooth in ltr -- no delta function -- so the
        marginalisation is a single GL quadrature.

        Because ``b_lob_theta`` is linear in ``(b_zero, b_infty)`` and
        the sigmoid ``sigma(theta)`` is ltr-independent, the
        ltr-marginalisation commutes with the sigmoid assembly: we
        precompute the plateau averages ``b_zero_bar``, ``b_infty_bar``
        once per ``(lob, zob)`` (see ``_marginalised_plateaus``) and
        then evaluate the analytic sigmoid at ``theta`` in a single
        vectorised call.  This is numerically identical (to machine
        precision) to looping over the ltr grid for every theta-node
        and is substantially cheaper when ``theta`` is an array.

        With ``use_plob_ltr=False`` the P(lob|ltr) factor is dropped
        and only prior(ltr) weights enter (diagnostic only: wrong
        physics).
        """
        if ltr_grid_size is None:
            ltr_grid_size = self.grid.ltr_grid_size
        pre = precomp if precomp is not None else self.bias_precompute(lob, zob)

        b_zero_bar, b_infty_bar = self._marginalised_plateaus(
            lob, zob, pre, ltr_grid_size, use_plob_ltr)

        theta_arr = np.asarray(theta, dtype=float)
        if not np.isfinite(b_zero_bar):
            return np.full_like(theta_arr, np.nan)
        s = self._sigmoid_theta(theta_arr, lob, zob)
        return b_zero_bar + (b_infty_bar - b_zero_bar) * s
