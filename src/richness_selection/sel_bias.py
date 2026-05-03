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
                 n_ltr: int = 60,
                 z_scheme: str = "E"):
        """
        Parameters
        ----------
        z_scheme : {"E", "D", "C"}
            z-axis integration recipe.
            "E" (default): split by physics into ring + outer-fg + outer-bg,
                GL on z for the ring, GL on ln|Delta chi| for the outer.
                Kernel-agnostic; ~0.3% on I1/I2 at Nz=80.
            "D": arcsinh transform  t = asinh(|Delta_chi_par| / R_excl).
                Single GL grid per side, ~0.3-0.5% at Nz=80.
            "C": CDF-based adaptive grid.  Build a rough f(z) on a dense
                (400-node) log-spaced grid, form its |cumulative|,
                place the final GL nodes at uniform-CDF levels so each
                node carries ~equal integrand weight.  ~0.05% on I1/I2
                at Nz=80 (~6x better than E/D).  Slightly more expensive
                per call due to the rough-grid setup.
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
        self.z_scheme = z_scheme
        self._cache: dict = {}

    # ---------------- small helpers -----------------------------------

    def _theta_lob(self, lob, zob):
        return float(theta_lambda(lob, zob, self.cosmo))

    def _sigmoid_theta(self, theta, lob, zob):
        theta_lob = self._theta_lob(lob, zob)
        k = self.damping / theta_lob
        theta0 = self.theta0_frac * theta_lob
        return 1.0 / (1.0 + np.exp(-k * (theta - theta0)))

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
        num = np.trapz(wt * b_m, np.log(m_grid))
        den = np.trapz(wt, np.log(m_grid))
        val = float(num / den) if den > 0 else float("nan")
        self._cache[key] = val
        return val

    # ---------------- z-grid schemes ----------------------------------

    def _z_grid_option_E(self, lob, zob, Nz, chi_o, R_excl,
                         z_fg_lo, z_bg_hi,
                         zs_ref, chi_ref, dchi_dz_ref):
        """Option E: split by physics into R1 (ring) + R2+R3 (outer fg+bg).

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

    def _rough_f_I2_proxy(self, lob, zob):
        """Cheap scalar f(z) proxy for Option C's CDF-based adaptive grid.

        Computes P[1]-integrand + I_2-integrand at each z.  The sum
        captures both distributional features: P[1] peaks at z=zob
        (smooth core, no xi spike), I_2 peaks at z=zob +/- dz_excl
        (two peaks at exclusion boundary).  A CDF built on |P[1]+I_2|
        places GL nodes around BOTH weight centres, so the resulting
        grid is good for all three quantities (P[1], I_1, I_2).
        """
        g = self.grid
        theta_lob = self._theta_lob(lob, zob)
        chi_o = float(self.cosmo.chi(zob))
        R_excl = R_lambda(lob) * (1.0 + zob)

        ln_M_min = np.log(10.0 ** np.log10(self.min_mass4integral))
        ln_M_max = np.log(10.0 ** self.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, g.NM)
        Ms = np.exp(lnMs); M_weight = wM * Ms
        lam_grid, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)

        theta_max_hi = 2.0 * theta_lob
        eps_theta = 1e-6

        def f_at_z(z):
            """|f_P1(z)| + |f_I2(z)| using split-at-exclusion theta grid."""
            chi_z = float(self.cosmo.chi(z))
            dV = float(self.cosmo.dV_dzdOm(z))
            wz_val = float(w_z(np.array([z]), zob)[0])
            if wz_val <= 0:
                return 0.0
            # theta_excl(z) split -- matches _P_operator's theta grid choice
            cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
                2.0 * chi_z * chi_o + 1e-30)
            cos_excl = min(max(cos_excl, -1.0), 1.0)
            th_lo = np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12 else eps_theta
            th_lo = max(th_lo, eps_theta)
            if th_lo >= theta_max_hi:
                return 0.0
            ths_z, wth_z = gl_nodes(th_lo, theta_max_hi, g.Nth)
            sin_th_z = np.sin(ths_z)
            th_weight_z = wth_z * 2.0 * np.pi * sin_th_z
            cos_th_z = np.cos(ths_z)
            dchi = np.sqrt(np.maximum(
                chi_z ** 2 + chi_o ** 2 - 2 * chi_z * chi_o * cos_th_z, 0.0))
            xi_th = self.xi_NL(dchi, zob)     # no mask; split already excludes
            theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
            fA = area_overlap(ths_z, theta_lob, theta_lam_l)
            ang_P1 = np.einsum('t,tL->L', th_weight_z, fA)
            ang_I2 = np.einsum('t,tL,t->L', th_weight_z, fA, xi_th)
            P_lmz = self.mor.pdf(lam_grid[:, None], Ms[None, :], z)
            rho_prefac = wz_val * lam_grid
            lam_int_P1 = np.einsum('L,LM,L->M', wlam, P_lmz,
                                    rho_prefac * ang_P1)
            lam_int_I2 = np.einsum('L,LM,L->M', wlam, P_lmz,
                                    rho_prefac * ang_I2)
            n_m = self.hmf(Ms, z); b_m = self.bias(Ms, z)
            f_P1 = dV * np.sum(M_weight * n_m * lam_int_P1)
            f_I2 = dV * np.sum(M_weight * n_m * b_m * lam_int_I2)
            return abs(f_P1) + abs(f_I2)

        return f_at_z

    def _z_grid_option_C(self, lob, zob, Nz, chi_o, R_excl,
                         z_fg_lo, z_bg_hi,
                         zs_ref, chi_ref, dchi_dz_ref):
        """Option C: CDF-based adaptive z-grid.

        1. Build a ~400-node rough z-grid (log-spaced in |z-zob| on each
           side so the exclusion peak and ring are densely sampled).
        2. Evaluate the real inner integrand f(z) (using I_2 as proxy)
           on the rough grid.
        3. Build the cumulative |f(z)| and normalise.
        4. Place the final Nz GL nodes at uniform CDF levels; the weights
           pick up a 1/(df/dz) Jacobian naturally.

        The method gives roughly 6x precision per node compared to
        fixed-grid Options E/D because nodes automatically cluster
        around whichever features dominate the integral.  Extra cost:
        one full rough-grid evaluation (~400 * inner-cost).
        """
        # Step 1: rough z-grid.  Log-spaced |z-zob| on each side.
        # 200 per side (400 total) resolves the sharp ring peak
        # reliably; smaller grids give unstable CDF inversion because
        # the peak is smoothed too much.  Setup cost ~10x inner-integrand
        # cost = ~200 ms per call at the reference point.
        n_rough_half = max(200, Nz // 2)
        dz_fg_max = zob - z_fg_lo
        dz_bg_max = z_bg_hi - zob
        eps_z = 1e-6
        dz_fg = np.exp(np.linspace(np.log(eps_z), np.log(dz_fg_max),
                                    n_rough_half))
        dz_bg = np.exp(np.linspace(np.log(eps_z), np.log(dz_bg_max),
                                    n_rough_half))
        z_rough = np.sort(np.concatenate(
            [zob - dz_fg, np.array([zob]), zob + dz_bg]))

        # Step 2: evaluate rough proxy
        f_at_z = self._rough_f_I2_proxy(lob, zob)
        f_rough = np.array([f_at_z(z) for z in z_rough])

        # Step 3: CDF of |f|
        from scipy.integrate import cumulative_trapezoid
        abs_f = np.abs(f_rough)
        cdf_raw = cumulative_trapezoid(abs_f, z_rough, initial=0.0)
        total = cdf_raw[-1]
        if total <= 0:
            # Degenerate case -- fall back to Option E
            return self._z_grid_option_E(lob, zob, Nz, chi_o, R_excl,
                                         z_fg_lo, z_bg_hi,
                                         zs_ref, chi_ref, dchi_dz_ref)
        cdf = cdf_raw / total

        # Step 4: place final GL nodes at uniform CDF levels.
        u_cdf, wu = gl_nodes(0.0, 1.0, Nz)
        z_nodes = np.interp(u_cdf, cdf, z_rough)
        # dz/du Jacobian = total/|f(z)|  (since F(z) = cdf_raw(z)/total
        # and u = F(z), so dz/du = 1/(dF/dz) = total/|f|).
        # The |f| MUST be evaluated at the SAME z_nodes where we'll
        # compute the integrand later, so interpolate from rough curve.
        f_at_nodes = np.interp(z_nodes, z_rough, abs_f)
        # Guard against zero rows
        f_at_nodes = np.maximum(f_at_nodes, 1e-30 * total)
        dz_du = total / f_at_nodes
        wzs = wu * dz_du
        return z_nodes, wzs

    def _z_grid_option_D(self, lob, zob, Nz, chi_o, R_excl,
                         z_fg_lo, z_bg_hi,
                         zs_ref, chi_ref, dchi_dz_ref):
        """Option D: arcsinh transform (tailored to I1 / I2).

        Change of variables  t = asinh(|Delta_chi_par| / R_excl)
        maps both the ring plateau (small |Delta_chi_par|) and the
        exclusion peak (Delta_chi_par = R_excl) into a smooth function
        of t with bounded Jacobian  dd/dt = R_excl cosh(t).

        Single GL grid per side (no region split), symmetric around
        zob.  Half of Nz per side.
        """
        n_side = max(15, Nz // 2)
        dis_fg_max = chi_o - float(self.cosmo.chi(z_fg_lo))
        dis_bg_max = float(self.cosmo.chi(z_bg_hi)) - chi_o
        t_fg_max = float(np.arcsinh(dis_fg_max / R_excl))
        t_bg_max = float(np.arcsinh(dis_bg_max / R_excl))

        # Foreground:  t in [0, t_fg_max];  d = R_excl sinh(t), z_fg = z at chi_o - d
        t_fg, w_t_fg = gl_nodes(0.0, t_fg_max, n_side)
        dis_fg = R_excl * np.sinh(t_fg)
        z_fg = np.interp(chi_o - dis_fg, chi_ref, zs_ref)
        dchi_dz_fg = np.interp(z_fg, zs_ref, dchi_dz_ref)
        # dz/dt = (dd/dt) / (dchi/dz) = R_excl cosh(t) / (c/H(z))
        dz_dt_fg = R_excl * np.cosh(t_fg) / dchi_dz_fg
        w_z_fg = w_t_fg * dz_dt_fg

        # Background
        t_bg, w_t_bg = gl_nodes(0.0, t_bg_max, n_side)
        dis_bg = R_excl * np.sinh(t_bg)
        z_bg = np.interp(chi_o + dis_bg, chi_ref, zs_ref)
        dchi_dz_bg = np.interp(z_bg, zs_ref, dchi_dz_ref)
        dz_dt_bg = R_excl * np.cosh(t_bg) / dchi_dz_bg
        w_z_bg = w_t_bg * dz_dt_bg

        # Assemble (foreground ascending in z, then background)
        zs = np.concatenate([z_fg[::-1], z_bg])
        wzs = np.concatenate([w_z_fg[::-1], w_z_bg])
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

        # Build the z grid + weights via the chosen scheme.
        R_excl = R_lambda(lob) * (1.0 + zob)
        zs_ref = np.linspace(0.0, 2.0, 2000)
        chi_ref = self.cosmo.chi(zs_ref)
        dchi_dz_ref = np.gradient(chi_ref, zs_ref)

        if self.z_scheme == "E":
            zs, wzs = self._z_grid_option_E(lob, zob, g.Nz,
                                            chi_o, R_excl,
                                            z_fg_lo, z_bg_hi,
                                            zs_ref, chi_ref, dchi_dz_ref)
        elif self.z_scheme == "D":
            zs, wzs = self._z_grid_option_D(lob, zob, g.Nz,
                                            chi_o, R_excl,
                                            z_fg_lo, z_bg_hi,
                                            zs_ref, chi_ref, dchi_dz_ref)
        elif self.z_scheme == "C":
            zs, wzs = self._z_grid_option_C(lob, zob, g.Nz,
                                            chi_o, R_excl,
                                            z_fg_lo, z_bg_hi,
                                            zs_ref, chi_ref, dchi_dz_ref)
        else:
            raise ValueError(f"Unknown z_scheme={self.z_scheme!r}; "
                             "expected 'E', 'D', or 'C'")

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

    def b_sel_marginalised(self, theta, lob, zob, ltr_grid_size=None,
                           precomp=None, use_plob_ltr: bool = True):
        """Marginalised bias per Eq. (b_marg_lt) of the TeX.

        P(ltr | lob, zob) propto P(lob | ltr, zob) * prior(ltr, zob),
        with prior(ltr, zob) = int dM n(M,z) P(ltr | M, z).

        P(lob | ltr, z) is the full EMG kernel of
        Eq. (costanzi_kernel):  (1 - f_prj) Gaussian + f_prj * EMG.
        This is smooth in ltr -- no delta function -- so the
        marginalisation is a single GL quadrature.

        With ``use_plob_ltr=False`` the P(lob|ltr) factor is dropped
        and only prior(ltr) weights enter (diagnostic only: wrong
        physics).
        """
        if ltr_grid_size is None:
            ltr_grid_size = self.grid.ltr_grid_size
        pre = precomp if precomp is not None else self.bias_precompute(lob, zob)

        # EMG kernel has a Gaussian component of width sigma ~ a few, so we
        # need ltr support that covers [lob - 6 sigma, lob + 6 sigma] plus
        # the exponential tail (length ~ 1/tau).  Conservative: ltr in [1, 3*lob].
        t_nodes, t_wts = gl_nodes(1.0, 3.0 * float(lob), ltr_grid_size * 2)

        # prior(ltr, z) = int dM n(M,z) P(ltr | M, z)
        log10_Mmin = np.log10(self.min_mass4integral)
        m_grid = 10.0 ** np.linspace(log10_Mmin, self.ln_M_max_log10, 50)
        hmf_m = self.hmf(m_grid, zob)
        p_ltr_M = self.mor.pdf(t_nodes[:, None], m_grid[None, :], zob)
        prior_ltr = np.trapz(p_ltr_M * (hmf_m * m_grid)[None, :],
                             np.log(m_grid), axis=1)

        if use_plob_ltr:
            from .plob_ltr import P_lob_given_ltr
            p_lob_ltr = np.array([float(P_lob_given_ltr(lob, float(ltr), zob))
                                  for ltr in t_nodes])
            p_ltr = p_lob_ltr * prior_ltr
        else:
            p_ltr = prior_ltr

        theta_arr = np.asarray(theta, dtype=float)
        num = np.zeros_like(theta_arr)
        den = 0.0
        for ltr, w, pw in zip(t_nodes, t_wts, p_ltr):
            weight = float(w * pw)
            if weight == 0.0:
                continue
            num = num + weight * self.b_lob_theta(
                theta_arr, ltr, zob, lob, precomp=pre)
            den += weight
        return num / den if den > 0 else np.full_like(theta_arr, np.nan)
