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
                 n_ltr: int = 60):
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

        # Fixed node count per half (default GridConfig.Nz = 80 -> 40 fg + 40 bg).
        # Log-spacing in |Delta chi| automatically gives ~0.005 effective dz
        # at the outer edge and much finer resolution near the inner edge.
        # Simpson requires an odd number of points per half.
        half = max(5, g.Nz // 2)
        if half % 2 == 0:
            half += 1
        n_fg = n_bg = half

        # Step 3: pin the inner edge of the LoS grid at the halo exclusion
        # radius R_excl = R_lambda(lob) * (1 + zob). Inside exclusion, the
        # integrand is identically zero via the xi_NL mask; spacing nodes
        # there just wastes weight.  Starting at R_excl also places a node
        # exactly on the step-edge of the integrand (xi_NL rises sharply
        # just outside exclusion), which otherwise spoils Simpson/trapz
        # convergence.
        R_excl = R_lambda(lob) * (1.0 + zob)

        dis_fg_max = chi_o - chi_fg_lo
        dis_bg_max = chi_bg_hi - chi_o
        # Inner edge sits just outside the exclusion; clamp so we never
        # cross dis_max if R_excl happens to exceed the photo-z support.
        dis_min_fg = min(R_excl, 0.5 * dis_fg_max)
        dis_min_bg = min(R_excl, 0.5 * dis_bg_max)

        u_fg = np.linspace(np.log(dis_min_fg), np.log(dis_fg_max), n_fg)
        u_bg = np.linspace(np.log(dis_min_bg), np.log(dis_bg_max), n_bg)
        dis_fg = np.exp(u_fg)
        dis_bg = np.exp(u_bg)

        # Convert LoS distances to redshifts (foreground < zob, background > zob)
        chi_fg = chi_o - dis_fg[::-1]           # ascending in z
        chi_bg = chi_o + dis_bg
        chi_zs = np.concatenate([chi_fg, chi_bg])
        zs_ref = np.linspace(0.0, 2.0, 2000)
        chi_ref = self.cosmo.chi(zs_ref)
        zs = np.interp(chi_zs, chi_ref, zs_ref)      # (Nz,)

        chi_z = self.cosmo.chi(zs)                    # (Nz,)
        dV = self.cosmo.dV_dzdOm(zs)                  # (Nz,)
        wz_kern = w_z(zs, zob)                        # (Nz,)

        # Simpson weights in u = ln|Delta chi| on each half, with Jacobian
        # to convert the integral to z:
        #   int dz f(z) = int du [dz/du] f(z(u))
        # with  dz/du = |dz/dchi| * |dchi/du| = (1 / chi'(z)) * |Delta chi|
        # and chi'(z) = c/H(z) = c * dV^(1/3) ... easier: numerical dz/du.
        def _simpson_weights_u(u):
            """Simpson 1/3 weights over a uniform grid of u."""
            n = u.size
            if n % 2 == 0:
                raise ValueError('Simpson needs an odd number of nodes')
            h = (u[-1] - u[0]) / (n - 1)
            w = np.ones(n)
            w[1:-1:2] = 4.0  # odd interior
            w[2:-1:2] = 2.0  # even interior (not including first or last)
            return w * h / 3.0

        w_u_fg = _simpson_weights_u(u_fg)      # (n_fg,)
        w_u_bg = _simpson_weights_u(u_bg)      # (n_bg,)

        # Jacobian dz/du per node: dz/du = |dchi/du| / (dchi/dz)
        # dchi/du = |Delta chi| = exp(u) for both halves
        # dchi/dz = c/H(z), computed numerically from the chi(z) table
        # since cosmo exposes chi but not H directly.
        # We can get dchi/dz from the chi_ref gradient for accuracy.
        dchi_dz_ref = np.gradient(chi_ref, zs_ref)     # (2000,)
        dchi_dz_zs = np.interp(zs, zs_ref, dchi_dz_ref)

        # |dchi/du| per node: fg is reversed, so u_fg[i] maps to zs[i] where
        # zs[:n_fg] is foreground ascending (low to zob) and dis_fg reversed.
        dchi_du_fg = dis_fg[::-1]   # matches zs[:n_fg]
        dchi_du_bg = dis_bg          # matches zs[n_fg:]
        dchi_du = np.concatenate([dchi_du_fg, dchi_du_bg])
        dz_du = dchi_du / dchi_dz_zs

        # Simpson weight in z = w_u * dz/du, per half (reversing for fg)
        w_u = np.concatenate([w_u_fg[::-1], w_u_bg])
        wzs = w_u * dz_du

        # M-grid
        log10_Mmin = np.log10(self.min_mass4integral)
        ln_M_min = np.log(10.0 ** log10_Mmin)
        ln_M_max = np.log(10.0 ** self.ln_M_max_log10)
        lnMs, wM = gl_nodes(ln_M_min, ln_M_max, g.NM)          # (NM,)
        Ms = np.exp(lnMs)

        # lambda (true richness) grid: eq. 3 runs lam over (0, lob]
        lam_grid, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)  # (Nlam,)

        # theta grid: (0, 2 theta_lob], where projected halos contribute
        theta_max = 2.0 * theta_lob
        ths, wth = gl_nodes(1e-6, theta_max, g.Nth)             # (Nth,)
        sin_th = np.sin(ths)                                    # (Nth,)
        sigmoid_th = self._sigmoid_theta(ths, lob, zob)         # (Nth,)

        # 3-D separation (Nz, Nth)
        cos_th = np.cos(ths)
        dchi2 = (chi_z[:, None] ** 2 + chi_o ** 2
                 - 2.0 * chi_z[:, None] * chi_o * cos_th[None, :])
        dchi = np.sqrt(np.maximum(dchi2, 0.0))
        xi_zth = self.xi_NL(dchi.ravel(), zob).reshape(dchi.shape)
        if self.exclusion:
            # Matteo's exclusion radius: R_lambda(lob) * (1 + zcl)
            R_excl = R_lambda(lob) * (1.0 + zob)
            xi_zth = np.where(dchi < R_excl, 0.0, xi_zth)

        # b(M, z) on (NM, Nz)
        bM_mz = self.bias(Ms[:, None], zs[None, :])            # (NM, Nz)
        # HMF n(M, z) on (NM, Nz)
        n_mz = self.hmf(Ms[:, None], zs[None, :])              # (NM, Nz)
        # P(lam | M, z) on (Nlam, NM, Nz)
        p_lmz = self.mor.pdf(lam_grid[:, None, None],
                             Ms[None, :, None],
                             zs[None, None, :])                # (Nlam, NM, Nz)

        # f_A(theta, lam, z, lob, zob): (Nth, Nlam, Nz)
        #   theta_lam(lam, z) = R_lambda(lam) (1+z) / chi(z)
        theta_lam_lz = (R_lambda(lam_grid)[:, None]
                        * (1.0 + zs[None, :]) / chi_z[None, :])  # (Nlam, Nz)
        # area_overlap: for each (lam, z), compute over ths
        # reshape theta_lam to 1-D of length Nlam*Nz and broadcast
        Nth = ths.size; Nlam = lam_grid.size; Nz = zs.size
        fA = np.empty((Nth, Nlam, Nz))
        for iz in range(Nz):
            fA[:, :, iz] = area_overlap(ths, theta_lob, theta_lam_lz[:, iz])

        # Angular weight (theta integral only, eq. 9 requires 2 pi sin theta)
        # theta-integrand (shared for all operators):
        th_weight = wth * 2.0 * np.pi * sin_th                 # (Nth,)
        # For P[1] we need:
        #   2 pi int dtheta sin(theta) f_A(theta, lam, z, lob, zob)
        # -> shape (Nlam, Nz)
        angular_P1 = np.einsum('t,tLz->Lz', th_weight, fA)     # (Nlam, Nz)

        # rho_prj(lam, z | lob, zob)  (pre-integrated over theta, shape (Nlam, Nz))
        rho_prj_P1 = wz_kern[None, :] * angular_P1 * lam_grid[:, None]

        # P[1] integrand:  dV * int d(lnM) M n(M,z) * int d lam P(lam|M,z) * rho_prj
        # We integrate in d(lnM) so the HMF gets multiplied by M.
        lam_integrand_P1 = np.einsum('L,LMz,Lz->Mz',
                                     wlam, p_lmz, rho_prj_P1)   # (NM, Nz)
        # wM are d(lnM) weights -- multiply by M explicitly:
        M_weight = wM * Ms                                       # (NM,)
        M_integrand_P1 = np.einsum('M,MZ,MZ->Z',
                                   M_weight, n_mz, lam_integrand_P1)  # (Nz,)
        P1 = float(np.sum(wzs * dV * M_integrand_P1))

        # For I1, I2 we need the integrand with b(M,z) and xi_NL(z, theta)
        # included.  Theta does NOT factorize from (M, z) for these, so we
        # need the full (Nth, NM, Nz) contraction:
        #   2 pi int dtheta sin(theta) f_A(theta,lam,z) b(M,z) xi(z,theta) [sigmoid]
        #
        # Since b(M,z) and xi(z,theta) don't depend on lam, it's cleanest to
        # split the lam integral into an angular-only factor:
        #   angular_I(theta, lam, z) = f_A(theta, lam, z)
        # and then integrate over theta after multiplying by xi, sigmoid.
        # (Nth, Nlam, Nz) f_A, (Nz, Nth) xi_zth.

        # integrand for I2: 2 pi sin(theta) f_A xi(z,theta)   -> (Nth, Nlam, Nz)
        xi_tz = xi_zth.T                                       # (Nth, Nz)
        # Integrate theta: contract Nth with weight (2 pi sin th) and sum
        #   ang_I2(lam, z) = sum_t wth * 2pi sin th * fA(t,L,z) * xi(t,z)
        ang_I2 = np.einsum('t,tLz,tz->Lz', th_weight, fA, xi_tz)  # (Nlam, Nz)
        # ang_I1 = same but weighted by sigmoid
        ang_I1 = np.einsum('t,t,tLz,tz->Lz',
                           th_weight, sigmoid_th, fA, xi_tz)   # (Nlam, Nz)

        # rho_prj-style prefactor: w_z * lam
        rho_prefac = wz_kern[None, :] * lam_grid[:, None]      # (Nlam, Nz)
        # Integrate lam * P(lam|M,z): (Nlam, Nz) -> (M, z) after weighting
        # Full integrand for I2:
        #   (Nlam, NM, Nz): wlam * p_lmz * (rho_prefac * ang_I2 * b)   -- b is (NM, Nz)
        # Factor out M-independent rho_prefac * ang:
        lam_I2 = rho_prefac * ang_I2                           # (Nlam, Nz)
        lam_I1 = rho_prefac * ang_I1                           # (Nlam, Nz)

        lam_int_I2 = np.einsum('L,LMz,Lz->Mz', wlam, p_lmz, lam_I2)  # (NM, Nz)
        lam_int_I1 = np.einsum('L,LMz,Lz->Mz', wlam, p_lmz, lam_I1)  # (NM, Nz)

        # Multiply by M b(M,z) n(M,z), integrate d(lnM) then d z
        M_I2 = np.einsum('M,MZ,MZ,MZ->Z',
                         M_weight, n_mz, bM_mz, lam_int_I2)    # (Nz,)
        M_I1 = np.einsum('M,MZ,MZ,MZ->Z',
                         M_weight, n_mz, bM_mz, lam_int_I1)    # (Nz,)
        I2 = float(np.sum(wzs * dV * M_I2))
        I1 = float(np.sum(wzs * dV * M_I1))

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
