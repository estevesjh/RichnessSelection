"""Frozen-physics selection bias -- transcription of
docs/richness_selection_frozen.tex, equation by equation.

The note is the spec; every public object here is named after, and
documented against, one of its equation labels:

  Sec. 1 (model + closure)
    eq. (lob_split)   lob = ltr + <Dprj>                    identity
    eq. (brm)         b_rm(theta) = b_rm_ss [1-sigma] + b_rm_ls sigma
                                                            -> ``MarginalisedBias``
                                                               (sel_bias.py) / ``sigma_x``
    eq. (xi_rmh)      xi_rm,h = b(M,z) b_rm(theta) xi_mm(r)  realised by weighting
                                                            the lss operators with b_rm
    eq. (DSprj)       lensing profile                        realised by the existing
                                                            ``DeltaSigmaPrj`` consuming
                                                            ``MarginalisedBias`` via
                                                            ``SigmaPrj._bsel_at``
    eq. (Dprj_rnd)    <Dprj>_rnd = pi thob^2 <Dlam>_rnd      -> ``Dprj_rnd``
    eq. (Dprj_ss/ls)  <Dprj>^X = b_rm^X <Dprj~>^X            closure via
                                                            ``SelBias._closure_ltr_vec``
    eq. (bhalo)       mass-averaged halo bias                -> inherited ``b_eff``
    eq. (bls)         b_rm^ls = b_halo [1 + 0.13 delta_prj],
                      delta_prj = (lob-ltr)/<Dprj>_rnd - 1   POISSON denominator:
                                                            ``as_precomp`` stores
                                                            Delta_RND = <Dprj>_rnd
    eq. (budget)      lob-ltr = rnd + b^ls ls + b^ss ss      test assertion
    eq. (bss)         b_rm^ss closure                        inherited
                                                            ``bias_from_precomp``

  Sec. 2 (frozen algorithm)
    eq. (sky_vars)    pi = chi-chi_o, s = 2 sqrt(chi chi_o) sin(theta/2),
                      r^2 = pi^2 + s^2 (exact)
    eq. (measure)     dz dV/dzdOm 2pi sin th dth = 2pi (1+pi/chi_o) dpi s ds
    eq. (alg_split)   <Dprj~>^ss = excl + cyl                -> ``operators``
    eq. (Ilam)        I_lambda(x) mass-integral moment       -> ``I_lambda``
    eq. (Ilam_ssls)   I^ss = (1-sigma) I, I^ls = sigma I     -> ``sigma_x``
    eq. (densities)   <Dlam>(z), <Dlam>_b(z)                 -> ``Dlam``, ``Dlam_b``
    eq. (Ilam_zero)   I_lambda(0) = <Dlam>_b(zob)            test assertion
    eq. (excl)        exclusion zone, spherical              -> ``Dprj_excl``
    eq. (taylor)      Limber expansion about the axis        inside ``Dprj_cyl``
    eq. (moments)     A_ss, B_ss transverse moments          inside ``Dprj_cyl``
    eq. (amp_drift)   I(x;z) ~ [<Dlam>_b(z)/<Dlam>_b(zob)] I(x;zob)
    eq. (cyl)         free-of-exclusion zone, fg+bg sides    -> ``Dprj_cyl``

Constants fixed by the note: pi_s = 20 R_excl (Sec. "choice of the
split"), theta support x in (0, 2], sigma(x) = [1+e^{-2.5(x-1/2)}]^-1,
x_lambda = (lambda/lob)^0.2, Buzzard slope 0.13 (``boost_slope``).

Performance: everything is numpy-vectorised (the exclusion-zone (r, mu)
block is one batched ``area_overlap`` call per r-segment; ss and ls
share every area_overlap / xi_mm / w evaluation since
I^ls = sigma(x) I pointwise) and cached per (lob, zob): the (M, lambda)
contraction, the z(pi) lookup, the window edges, the <Dlam>/<Dlam>_b
drift splines, the assembled ``FrozenOperators``, and the legacy
precomp dict.  Repeat calls are dict lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import bisect

from .sel_bias import SelBias, BiasPlateaus, MarginalisedBias
from .geometry import R_lambda, area_overlap
from .photoz import w_z, zmin4zkernel, zmax4zkernel
from .gl import gl_nodes

__all__ = ["FrozenOperators", "FrozenSelBias", "SelectionBiasLike",
           "BiasPlateaus", "MarginalisedBias"]

# zone split pi_s = 20 R_excl (note Sec. "The split point balances ...")
PI_S_OVER_REX = 20.0


@dataclass(frozen=True)
class FrozenOperators:
    """Level 1 -- the reduced operators at one (lob, zob).

    All fields are scalars: the theta-integral is already done and
    lambda_tr never appears in eqs. (excl)/(cyl) -- the tilde marks
    unit target bias.  The ltr-dependent, biased operators
    <Dprj>^X = b_rm^X(ltr) * tDprj_X are vectors and live in
    ``BiasPlateaus`` (sel_bias.py), not here.
    """
    lob: float
    zob: float
    Dprj_rnd: float     # <Dprj>_rnd          eq. (Dprj_rnd) / (alg_split)
    tDprj_ss: float     # <Dprj~>^ss_lss      eq. (excl) + eq. (cyl), ss
    tDprj_ls: float     # <Dprj~>^ls_lss      eq. (excl) + eq. (cyl), ls
    b_halo: float       # eq. (bhalo)

    def as_precomp(self) -> dict:
        """Legacy adapter -- the ONLY place the production dict appears.

        Delta_RND = <Dprj>_rnd realises the note's Poisson delta_prj
        convention (eq. bls); the inherited ``bias_from_precomp`` /
        ``_closure_ltr_vec`` then reproduce eqs. (bls) + (bss) verbatim.
        """
        return dict(lob=self.lob, zob=self.zob,
                    P1=self.Dprj_rnd,
                    I1=self.tDprj_ls,
                    I2=self.tDprj_ss + self.tDprj_ls,
                    b_eff=self.b_halo,
                    Delta_RND=self.Dprj_rnd,
                    denom=self.tDprj_ss)


@runtime_checkable
class SelectionBiasLike(Protocol):
    """The interface the lensing pipeline requires of a bias method.

    Both ``SelBias`` (production) and ``FrozenSelBias`` satisfy it;
    ``SigmaPrj`` / ``DeltaSigmaPrj`` should depend on nothing more.
    """
    def bias_precompute(self, lob, zob) -> dict: ...
    def plateaus(self, lob, zob, **kw) -> BiasPlateaus: ...
    def marginalised_bias(self, lob, zob, **kw) -> MarginalisedBias: ...
    def b_rm(self, theta, lob, zob, **kw): ...


class FrozenSelBias(SelBias):
    """``SelBias`` with the frozen-physics reduced operators.

    Only ``bias_precompute`` is overridden (through ``operators``);
    the closure, marginalisation, sigmoid assembly and the entire
    lensing pipeline are inherited unchanged.
    """

    # numerical knobs (validated values of the note's Sec. "Validation")
    n_r: int = 48          # exclusion zone: log-GL(r) per segment
    n_mu: int = 48         # exclusion zone: GL(mu)
    n_pi: int = 80         # free-of-exclusion zone: log-GL(pi) per side
    n_x: int = 200         # GL(x) for the transverse moments A, B
    n_z_drift: int = 15    # coarse grid for the <Dlam>_b(z) drift
    n_z_rnd: int = 48      # GL(z) for the 1-D random-channel integral

    # ---------------- (M, lambda) contraction, cached per (lob, zob) --

    def _frozen_ml(self, lob, zob):
        """One MOR evaluation -> both contractions of the (M, lambda)
        sector at frozen zob: the bias-weighted weights of eq. (Ilam)
        and the unweighted ones of eq. (densities)."""
        key = ("frozen_ml", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        w_b, w_0, x_lam = self._ml_weights_at(lob, zob)
        val = dict(w_b=w_b, w_0=w_0, x_lam=x_lam)
        self._cache[key] = val
        return val

    def _ml_weights_at(self, lob, z):
        """(w_b, w_0, x_lam) at arbitrary z (uncached; used by drifts).

        w_X(lam) = wlam * lam * int dM M n(M,z) [b(M,z)] P(lam|M,z);
        x_lam = (lam/lob)^0.2 is the donor-size ratio of eq. (Ilam).
        """
        lnMs, wM = gl_nodes(np.log(self.min_mass4integral),
                            np.log(10.0 ** self.ln_M_max_log10), 24)
        Ms = np.exp(lnMs)
        Mw = wM * Ms
        lam, wlam = gl_nodes(1e-6, float(lob), self.n_ltr)
        n_m = self.hmf(Ms, z)
        b_m = self.bias(Ms, z)
        P_lm = self.mor.pdf(lam[:, None], Ms[None, :], z)
        w_0 = wlam * lam * (P_lm @ (Mw * n_m))
        w_b = wlam * lam * (P_lm @ (Mw * n_m * b_m))
        x_lam = R_lambda(lam) / R_lambda(lob)
        return w_b, w_0, x_lam

    # ---------------- eq. (densities) ---------------------------------

    def Dlam(self, lob, z):
        """<Dlam>(z): unweighted contamination density (eq. densities)."""
        _, w_0, _ = self._ml_weights_at(lob, z)
        return float(np.sum(w_0))

    def Dlam_b(self, lob, z):
        """<Dlam>_b(z): bias-weighted contamination density."""
        w_b, _, _ = self._ml_weights_at(lob, z)
        return float(np.sum(w_b))

    # ---------------- eq. (Ilam) + eq. (Ilam_ssls) ---------------------

    def I_lambda(self, lob, zob):
        """The frozen mass-integral moment I_lambda(x) (eq. Ilam).

        Vectorised callable of x = s/R_excl.  The channel versions are
        I^ss = (1-sigma_x(x)) I and I^ls = sigma_x(x) I (eq. Ilam_ssls);
        I_lambda(0) = <Dlam>_b(zob) (eq. Ilam_zero).
        """
        ml = self._frozen_ml(lob, zob)
        w_b, x_lam = ml["w_b"], ml["x_lam"]

        def I(x):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            xb = np.broadcast_to(x[:, None], (x.size, x_lam.size))
            return area_overlap(xb, 1.0, x_lam) @ w_b

        return I

    @staticmethod
    def sigma_x(x):
        """Universal sigmoid sigma(x) of eq. (brm), x = theta/thob."""
        return 1.0 / (1.0 + np.exp(-2.5 * (np.asarray(x, dtype=float) - 0.5)))

    # ---------------- line-of-sight tables, cached per (lob, zob) ------

    def _los(self, zob):
        """chi_o, z(pi) interpolation table, photo-z window edges."""
        key = ("frozen_los", float(zob))
        if key in self._cache:
            return self._cache[key]
        chi_o = float(self.cosmo.chi(zob))
        zs_ref = np.linspace(0.0, 2.0, 4000)
        chi_ref = np.asarray(self.cosmo.chi(zs_ref), dtype=float)
        z_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
        z_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
        val = dict(chi_o=chi_o, zs_ref=zs_ref, chi_ref=chi_ref,
                   z_lo=z_lo, z_hi=z_hi)
        self._cache[key] = val
        return val

    def _z_of_pi(self, los, pi):
        return np.interp(los["chi_o"] + pi, los["chi_ref"], los["zs_ref"])

    def _drift(self, lob, zob):
        """eq. (amp_drift): <Dlam>(z) and <Dlam>_b(z) on a coarse
        z-grid, normalised at zob (shape frozen, amplitude exact)."""
        key = ("frozen_drift", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        los = self._los(zob)
        z_coarse = np.linspace(los["z_lo"], los["z_hi"], self.n_z_drift)
        d0 = np.empty(self.n_z_drift)
        db = np.empty(self.n_z_drift)
        for i, zc in enumerate(z_coarse):
            w_b, w_0, _ = self._ml_weights_at(lob, zc)
            d0[i] = np.sum(w_0)
            db[i] = np.sum(w_b)
        ml = self._frozen_ml(lob, zob)
        val = dict(z=z_coarse,
                   Dlam=d0, Dlam_b=db,
                   a_0=d0 / float(np.sum(ml["w_0"])),
                   a_b=db / float(np.sum(ml["w_b"])))
        self._cache[key] = val
        return val

    # ---------------- eq. (Dprj_rnd) / (alg_split), random channel -----

    def Dprj_rnd(self, lob, zob):
        """<Dprj>_rnd = pi thob^2 int dz (dV/dz dOm) w(z, zob) <Dlam>(z).

        eq. (Dprj_rnd) with eq. (alg_split)'s 1-D form -- the clean
        formula of the note (no exclusion carve-out of the random
        channel).  <Dlam>(z) is interpolated off the eq. (amp_drift)
        coarse grid.
        """
        los = self._los(zob)
        drift = self._drift(lob, zob)
        thob = self._theta_lob(lob, zob)

        zs, wz = gl_nodes(los["z_lo"], los["z_hi"], self.n_z_rnd)
        dV = np.asarray(self.cosmo.dV_dzdOm(zs), dtype=float)
        wwin = w_z(zs, zob)
        Dlam_z = np.interp(zs, drift["z"], drift["Dlam"])
        return float(np.pi * thob ** 2
                     * np.sum(wz * dV * wwin * Dlam_z))

    # ---------------- eq. (excl): exclusion zone ------------------------

    def Dprj_excl(self, lob, zob):
        """(<Dprj~>^ss, <Dprj~>^ls) exclusion-zone pieces, eq. (excl).

        2pi int_Rex^rmax dr r^2 xi_mm(r) G_X(r) with
        G_X(r) = 2 int_{mu_lo}^{mu_hi} dmu I^X(x_{r mu}) wbar(r mu),
        mu_lo = sqrt(max(0, 1-(2Rex/r)^2)), mu_hi = min(1, pi_s/r),
        rmax = sqrt(pi_s^2 + 4 Rex^2); log-GL(r) split at the
        derivative kink r = 2 Rex.
        """
        los = self._los(zob)
        ml = self._frozen_ml(lob, zob)
        w_b, x_lam = ml["w_b"], ml["x_lam"]
        Rex = R_lambda(lob) * (1.0 + zob)
        pi_s = PI_S_OVER_REX * Rex
        r_max = np.sqrt(pi_s ** 2 + 4.0 * Rex ** 2)
        mu_t, mu_w = gl_nodes(0.0, 1.0, self.n_mu)

        tot = 0.0
        tot_ls = 0.0
        for (a, b) in ((Rex, 2.0 * Rex), (2.0 * Rex, r_max)):
            lr, wr = gl_nodes(np.log(a), np.log(b), self.n_r)
            rs = np.exp(lr)
            xi_r = self.xi_NL(rs, zob)

            mu_lo = np.sqrt(np.maximum(0.0, 1.0 - (2.0 * Rex / rs) ** 2))
            mu_hi = np.minimum(1.0, pi_s / rs)
            active = mu_hi > mu_lo

            mus = mu_lo[:, None] + (mu_hi - mu_lo)[:, None] * mu_t[None, :]
            wmu = (mu_hi - mu_lo)[:, None] * mu_w[None, :]
            x = (rs[:, None] / Rex) * np.sqrt(np.maximum(1.0 - mus ** 2, 0.0))

            # one batched area_overlap per segment; ss and ls share it
            fA = area_overlap(x.reshape(-1), 1.0, x_lam)
            Ix = (fA @ w_b).reshape(self.n_r, self.n_mu)
            Ix_ls = self.sigma_x(x) * Ix

            pi_grid = rs[:, None] * mus
            wbar = 0.5 * (w_z(self._z_of_pi(los, pi_grid), zob)
                          + w_z(self._z_of_pi(los, -pi_grid), zob))

            G = 2.0 * np.sum(wmu * Ix * wbar, axis=1) * active
            G_ls = 2.0 * np.sum(wmu * Ix_ls * wbar, axis=1) * active

            tot += float(np.sum(wr * rs ** 3 * xi_r * G))
            tot_ls += float(np.sum(wr * rs ** 3 * xi_r * G_ls))

        excl_tot = 2.0 * np.pi * tot
        excl_ls = 2.0 * np.pi * tot_ls
        return excl_tot - excl_ls, excl_ls

    # ---------------- eq. (cyl): free-of-exclusion zone -----------------

    def Dprj_cyl(self, lob, zob):
        """(<Dprj~>^ss, <Dprj~>^ls) free-of-exclusion pieces, eq. (cyl).

        Limber expansion eq. (taylor) -> transverse moments
        eq. (moments), amplitude drift eq. (amp_drift), background and
        foreground sides written explicitly with the (1 +/- pi/chi_o)^2
        distance ratio, out to the window edges z_lo / z_hi.
        """
        los = self._los(zob)
        ml = self._frozen_ml(lob, zob)
        w_b, x_lam = ml["w_b"], ml["x_lam"]
        drift = self._drift(lob, zob)
        chi_o = los["chi_o"]
        Rex = R_lambda(lob) * (1.0 + zob)
        pi_s = PI_S_OVER_REX * Rex

        # eq. (moments): A_X, B_X from one x-grid shared by ss and ls
        x_n, x_w = gl_nodes(0.0, 2.0, self.n_x)
        I_n = area_overlap(x_n, 1.0, x_lam) @ w_b
        I_ls = self.sigma_x(x_n) * I_n
        A_tot = Rex ** 2 * float(np.sum(x_w * x_n * I_n))
        B_tot = Rex ** 4 * float(np.sum(x_w * x_n ** 3 * I_n))
        A_ls = Rex ** 2 * float(np.sum(x_w * x_n * I_ls))
        B_ls = Rex ** 4 * float(np.sum(x_w * x_n ** 3 * I_ls))

        def one_channel(A_X, B_X):
            total = 0.0
            for sgn, z_lim in ((+1.0, los["z_hi"]), (-1.0, los["z_lo"])):
                pi_max = abs(float(self.cosmo.chi(z_lim)) - chi_o)
                if pi_max <= pi_s:
                    continue
                lu, wu = gl_nodes(np.log(pi_s), np.log(pi_max), self.n_pi)
                pis = np.exp(lu)
                zP = self._z_of_pi(los, sgn * pis)
                wwin = w_z(zP, zob)
                a_b = np.interp(zP, drift["z"], drift["a_b"])
                chi_fac = ((chi_o + sgn * pis) / chi_o) ** 2
                xi_v = self.xi_NL(pis, zob)
                dxi = (self.xi_NL(pis * 1.005, zob)
                       - self.xi_NL(pis * 0.995, zob)) / (0.01 * pis)
                integ = A_X * xi_v + (B_X / (2.0 * pis)) * dxi
                total += float(np.sum(wu * pis * chi_fac * wwin * a_b
                                      * integ))
            return 2.0 * np.pi * total

        cyl_tot = one_channel(A_tot, B_tot)
        cyl_ls = one_channel(A_ls, B_ls)
        return cyl_tot - cyl_ls, cyl_ls

    # ---------------- level-1 assembly + legacy adapter -----------------

    def operators(self, lob, zob) -> FrozenOperators:
        """eq. (alg_split): assemble the reduced operators, cached."""
        key = ("frozen_ops", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        excl_ss, excl_ls = self.Dprj_excl(lob, zob)
        cyl_ss, cyl_ls = self.Dprj_cyl(lob, zob)
        val = FrozenOperators(
            lob=float(lob), zob=float(zob),
            Dprj_rnd=self.Dprj_rnd(lob, zob),
            tDprj_ss=excl_ss + cyl_ss,
            tDprj_ls=excl_ls + cyl_ls,
            b_halo=self.b_eff(lob, zob))
        self._cache[key] = val
        return val

    def bias_precompute(self, lob, zob):
        """Legacy-dict view of ``operators`` (Poisson Delta_RND)."""
        key = ("frozen_pre", float(lob), float(zob))
        if key in self._cache:
            return self._cache[key]
        pre = self.operators(lob, zob).as_precomp()
        self._cache[key] = pre
        return pre
