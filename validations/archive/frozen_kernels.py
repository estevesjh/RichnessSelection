"""Frozen-physics reformulation of I_1, I_2: kernels, integrands, analytics.

Companion to docs/richness_selection_frozen.tex. Implements the exact
transverse-variable rewrite

    R_perp = 2 sqrt(chi chi_o) sin(theta/2),   Pi = chi(z) - chi_o,
    Dchi^2 = Pi^2 + R_perp^2                    (exact),
    dz (dV/dz dOm) 2pi sin(theta) dtheta = 2pi (chi/chi_o) dPi R_perp dR_perp,

freezes the slow z-dependences (HMF, bias, MOR, aperture angle) at z_ob,
and assembles I_1, I_2 as

    near zone (|Pi| <= Pi_s):  spherical r-outer
        I_near = 2pi int_R^rmax dr r^2 xi(r) G_X(r),
        G_X(r) = 2 int_{mu_lo}^{mu_hi} dmu F_X((r/R) sqrt(1-mu^2)) w_z(r mu)
    far zone  (|Pi| >  Pi_s):  Limber + first moment correction
        I_far = 2pi sum_pm int dPi (chi/chi_o) w_z(z) s(z)
                    [ A_X xi(|Pi|) + B_X/(2|Pi|) xi'(|Pi|) ]

with F_X(x) the bias-weighted aperture profile (F_1 = sigma F_2 exactly),
A_X = R^2 int x F_X dx, B_X = R^4 int x^3 F_X dx, s(z) = <b lam>(z)/<b lam>(z_ob).

Outputs:
  docs/figs/pedag_frozen_kernels.png   universal angular kernels + poly fits
  docs/figs/pedag_frozen_radial.png    smooth near-zone radial integrand
  docs/figs/pedag_frozen_xifit.png     xi_NL power-law fit over near zone
  docs/figs/pedag_frozen_farzone.png   far-zone Pi-integrand, BAO + w_z window
  validations/cache/frozen_assembly.csv   I1/I2 assembled vs scipy.quad truth
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import bisect

from _common import CACHE_DIR

from richness_selection import (Cosmology, PkGrid, HMF, Bias, MOR, XiNL,
                                SelBias)
from richness_selection.sigma_m import SigmaM
from richness_selection.gl import gl_nodes
from richness_selection.geometry import R_lambda, area_overlap
from richness_selection.photoz import w_z, zmin4zkernel, zmax4zkernel

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "docs", "figs"))
os.makedirs(FIG_DIR, exist_ok=True)

NM, NLTR, NTH = 24, 60, 10
PI_S_OVER_R = 20.0          # near/far split Pi_s = 20 R_excl
POINTS = [(20.0, 0.5), (52.5, 0.425)]


def build_stack():
    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    xi = XiNL(cosmo); xi.build()
    return dict(cosmo=cosmo, pk=pk, sm=sm, hmf=hmf, bias=bias, mor=mor, xi=xi)


# ------------------------------------------------------------------
# Frozen (M, lambda) sector: bias-weighted aperture profile F_X(x)
# ------------------------------------------------------------------

def sigmoid_x(x):
    """Universal sigmoid sigma(x), x = theta/theta_lambda."""
    return 1.0 / (1.0 + np.exp(-2.5 * (np.asarray(x, dtype=float) - 0.5)))


def frozen_weights(stack, lob, z):
    """w_eff(lam) = wlam * lam * int dM M n(M,z) b(M,z) P(lam|M,z).

    The only lambda-theta coupler left is f_A(x, rho(lam)); everything
    else contracts here, once, at the frozen redshift.
    """
    lnMs, wM = gl_nodes(np.log(1e13), np.log(10 ** 15.5), NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, lob, NLTR)
    n_m = stack['hmf'](Ms, z)
    b_m = stack['bias'](Ms, z)
    P_lm = stack['mor'].pdf(lam_grid[:, None], Ms[None, :], z)
    q_b = P_lm @ (M_weight * n_m * b_m)          # (Nlam,)
    w_eff = wlam * lam_grid * q_b
    rho = R_lambda(lam_grid) / R_lambda(lob)     # (lam/lob)^0.2 at frozen z
    return lam_grid, w_eff, rho


def make_F(stack, lob, zob):
    """Return F_2(x), F_1(x) callables (vectorised) at frozen z_ob.

    F_2(x) = sum_lam w_eff(lam) f_A(x, rho(lam));  F_1 = sigma(x) F_2(x).
    x = theta/theta_lambda = R_perp/R_excl. theta_lob scale cancels
    (f_A homogeneous degree 0), so evaluate at a fiducial theta_lob=1.
    """
    lam_grid, w_eff, rho = frozen_weights(stack, lob, zob)

    def F2(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # explicit outer broadcast: area_overlap silently pairs x with rho
        # elementwise whenever len(x) == len(rho) (Matteo's convention)
        xb = np.broadcast_to(x[:, None], (x.size, rho.size))
        fA = area_overlap(xb, 1.0, rho)          # (Nx, Nlam)
        return fA @ w_eff

    def F1(x):
        return sigmoid_x(x) * F2(x)

    return F2, F1


# ------------------------------------------------------------------
# Assembly: near zone (spherical r-outer) + far zone (Limber + moment)
# ------------------------------------------------------------------

def assemble(stack, lob, zob, n_r=48, n_mu=48, n_far=80):
    cosmo = stack['cosmo']
    xi = stack['xi']
    chi_o = float(cosmo.chi(zob))
    R = R_lambda(lob) * (1.0 + zob)              # R_excl [cMpc/h]
    Pi_s = PI_S_OVER_R * R

    zs_ref = np.linspace(0.0, 2.0, 4000)
    chi_ref = cosmo.chi(zs_ref)

    def z_of_Pi(Pi):
        return np.interp(chi_o + Pi, chi_ref, zs_ref)

    F2, F1 = make_F(stack, lob, zob)

    # --- near zone: I = 2pi int_R^rmax dr r^2 xi(r) G_X(r) ------------
    # exact partition {r > R, R_perp <= 2R, |Pi| <= Pi_s}:
    #   mu_lo(r) = sqrt(max(0, 1-(2R/r)^2)),  mu_hi(r) = min(1, Pi_s/r)
    # kink at r = 2R -> split GL segments there.
    r_max_near = np.sqrt(Pi_s ** 2 + 4.0 * R ** 2)
    mu_t, mu_w = gl_nodes(0.0, 1.0, n_mu)        # template on (0,1)

    def G(r, F):
        mu_lo = np.sqrt(max(0.0, 1.0 - (2.0 * R / r) ** 2))
        mu_hi = min(1.0, Pi_s / r)
        if mu_hi <= mu_lo:
            return 0.0
        mus = mu_lo + (mu_hi - mu_lo) * mu_t
        wmu = (mu_hi - mu_lo) * mu_w
        Rp = r * np.sqrt(np.maximum(1.0 - mus ** 2, 0.0))
        # w_z exactly, both LoS signs (parabolic kernel nearly even)
        wz_pm = 0.5 * (w_z(z_of_Pi(r * mus), zob)
                       + w_z(z_of_Pi(-r * mus), zob))
        return 2.0 * float(np.sum(wmu * F(Rp / R) * wz_pm))

    def near(F):
        total = 0.0
        for (a, b) in ((R, 2.0 * R), (2.0 * R, r_max_near)):
            lr, wr = gl_nodes(np.log(a), np.log(b), n_r)
            rs = np.exp(lr)
            xi_r = xi(rs, zob)
            Gs = np.array([G(r, F) for r in rs])
            total += float(np.sum(wr * rs ** 3 * xi_r * Gs))
        return 2.0 * np.pi * total

    # --- far zone: Limber + first moment correction -------------------
    # angular moments at frozen z_ob:
    x_n, x_w = gl_nodes(0.0, 2.0, 200)
    A2 = R ** 2 * float(np.sum(x_w * x_n * F2(x_n)))
    B2 = R ** 4 * float(np.sum(x_w * x_n ** 3 * F2(x_n)))
    A1 = R ** 2 * float(np.sum(x_w * x_n * F1(x_n)))
    B1 = R ** 4 * float(np.sum(x_w * x_n ** 3 * F1(x_n)))

    # amplitude drift <b lam>(z) / <b lam>(z_ob), coarse spline
    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    z_coarse = np.linspace(z_fg_lo, z_bg_hi, 15)
    amp = np.array([np.sum(frozen_weights(stack, lob, zc)[1])
                    for zc in z_coarse])
    amp /= np.sum(frozen_weights(stack, lob, zob)[1])

    def far(A_X, B_X):
        total = 0.0
        for sgn, z_lim in ((-1.0, z_fg_lo), (+1.0, z_bg_hi)):
            Pi_max = abs(float(cosmo.chi(z_lim)) - chi_o)
            if Pi_max <= Pi_s:
                continue
            lu, wu = gl_nodes(np.log(Pi_s), np.log(Pi_max), n_far)
            Pis = np.exp(lu)
            zPi = z_of_Pi(sgn * Pis)
            wzv = w_z(zPi, zob)
            sz = np.interp(zPi, z_coarse, amp)
            # (chi/chi_o)^2: one power from the dPi R_perp dR_perp
            # measure, one from the aperture-moment dilation
            # (x = theta/theta_lam scales the R_perp moment by chi/chi_o)
            chi_fac = ((chi_o + sgn * Pis) / chi_o) ** 2
            xi_v = xi(Pis, zob)
            dxi = (xi(Pis * 1.005, zob) - xi(Pis * 0.995, zob)) / (0.01 * Pis)
            integ = A_X * xi_v + (B_X / (2.0 * Pis)) * dxi
            total += float(np.sum(wu * Pis * chi_fac * wzv * sz * integ))
        return 2.0 * np.pi * total

    I2 = near(F2) + far(A2, B2)
    I1 = near(F1) + far(A1, B1)
    aux = dict(R=R, Pi_s=Pi_s, A2=A2, B2=B2, A1=A1, B1=B1,
               F2=F2, F1=F1, G=G, z_of_Pi=z_of_Pi,
               z_fg_lo=z_fg_lo, z_bg_hi=z_bg_hi, chi_o=chi_o,
               r_max_near=r_max_near)
    return I1, I2, aux


# ------------------------------------------------------------------
# scipy.quad truth (matched-inner, original (z, theta) coordinates)
# ------------------------------------------------------------------

def quad_truth(stack, lob, zob):
    cosmo = stack['cosmo']
    chi_o = float(cosmo.chi(zob))
    theta_lob = R_lambda(lob) * (1.0 + zob) / chi_o
    R_excl = R_lambda(lob) * (1.0 + zob)
    lnMs, wM = gl_nodes(np.log(1e13), np.log(10 ** 15.5), NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, lob, NLTR)
    theta_max = 2.0 * theta_lob
    eps_theta = 1e-6

    def f_inner(z, which):
        chi_z = float(cosmo.chi(z))
        dV = float(cosmo.dV_dzdOm(z))
        wz_val = float(w_z(np.array([z]), zob)[0])
        if wz_val <= 0:
            return 0.0
        cos_excl = (chi_z ** 2 + chi_o ** 2 - R_excl ** 2) / (
            2.0 * chi_z * chi_o + 1e-30)
        cos_excl = min(max(cos_excl, -1.0), 1.0)
        th_lo = np.arccos(cos_excl) if cos_excl < 1.0 - 1e-12 else eps_theta
        th_lo = max(th_lo, eps_theta)
        if th_lo >= theta_max:
            return 0.0
        ths, wth = gl_nodes(th_lo, theta_max, NTH)
        th_weight = wth * 2.0 * np.pi * np.sin(ths)
        sig = 1.0 / (1.0 + np.exp(
            -(2.5 / theta_lob) * (ths - 0.5 * theta_lob)))
        dchi = np.sqrt(np.maximum(
            chi_z ** 2 + chi_o ** 2 - 2 * chi_z * chi_o * np.cos(ths), 0.0))
        xi_th = stack['xi'](dchi, zob)
        theta_lam_l = R_lambda(lam_grid) * (1.0 + z) / chi_z
        fA = area_overlap(ths, theta_lob, theta_lam_l)
        if which == 'P1':
            ang = np.einsum('t,tL->L', th_weight, fA)
        elif which == 'I2':
            ang = np.einsum('t,tL,t->L', th_weight, fA, xi_th)
        else:
            ang = np.einsum('t,t,tL,t->L', th_weight, sig, fA, xi_th)
        P_lmz = stack['mor'].pdf(lam_grid[:, None], Ms[None, :], z)
        lam_int = np.einsum('L,LM,L->M', wlam, P_lmz,
                            wz_val * lam_grid * ang)
        n_m = stack['hmf'](Ms, z)
        if which == 'P1':
            return dV * np.sum(M_weight * n_m * lam_int)
        b_m = stack['bias'](Ms, z)
        return dV * np.sum(M_weight * n_m * b_m * lam_int)

    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    out = {}
    for which in ('P1', 'I1', 'I2'):
        v_fg, _ = quad(f_inner, z_fg_lo, zob, args=(which,),
                       epsrel=1e-6, limit=200)
        v_bg, _ = quad(f_inner, zob, z_bg_hi, args=(which,),
                       epsrel=1e-6, limit=200)
        out[which] = v_fg + v_bg
    return out


# ------------------------------------------------------------------
# Frozen P[1]: 1-D LoS integral with the production exclusion sliver
# ------------------------------------------------------------------

def frozen_P1(stack, lob, zob, n_pi=80):
    """P[1] in the flat-sky reformulation, matching the production
    exclusion convention (theta lower limit theta_excl(z) also for X=1,
    i.e. the 3-D ball R_perp < sqrt(R^2 - Pi^2) is removed):

        P1 = 2pi sum_pm int dPi (chi/chi_o) w_z(z) s0(z)
                 R^2 [ M0(2) - M0(x_excl(Pi)) ],
        x_excl = sqrt(max(0, 1 - (Pi/R)^2)),
        M0(y) = int_0^y x F_0(x) dx,

    with F_0 the lambda-weighted aperture profile WITHOUT the halo
    bias (X=1 carries no b), frozen at z_ob, and s0(z) the slow
    amplitude drift <lam>(z)/<lam>(z_ob). Without the exclusion
    sliver this is the closed form P1 = pi theta_lam^2 int dz
    (dV/dz dOm) w_z <lam>(z) -- the moment identity.
    """
    cosmo = stack['cosmo']
    chi_o = float(cosmo.chi(zob))
    R = R_lambda(lob) * (1.0 + zob)

    lnMs, wM = gl_nodes(np.log(1e13), np.log(10 ** 15.5), NM)
    Ms = np.exp(lnMs); M_weight = wM * Ms
    lam_grid, wlam = gl_nodes(1e-6, lob, NLTR)
    rho = R_lambda(lam_grid) / R_lambda(lob)

    def w_eff0_at(z):
        n_m = stack['hmf'](Ms, z)
        P_lm = stack['mor'].pdf(lam_grid[:, None], Ms[None, :], z)
        q0 = P_lm @ (M_weight * n_m)
        return wlam * lam_grid * q0

    w0 = w_eff0_at(zob)

    # cumulative moment M0(y) of F_0 on a fine x grid
    xg = np.linspace(0.0, 2.0, 801)
    xb = np.broadcast_to(xg[:, None], (xg.size, rho.size))
    F0x = area_overlap(xb, 1.0, rho) @ w0
    M0 = cumulative_trapezoid(xg * F0x, xg, initial=0.0)
    M0_full = float(M0[-1])

    zs_ref = np.linspace(0.0, 2.0, 4000)
    chi_ref = cosmo.chi(zs_ref)

    def z_of_Pi(Pi):
        return np.interp(chi_o + Pi, chi_ref, zs_ref)

    z_fg_lo = float(bisect(zmin4zkernel, -2.0, 2.0, args=(zob,)))
    z_bg_hi = float(bisect(zmax4zkernel, -2.0, 2.0, args=(zob,)))
    z_coarse = np.linspace(z_fg_lo, z_bg_hi, 15)
    s0 = np.array([np.sum(w_eff0_at(zc)) for zc in z_coarse])
    s0 /= np.sum(w0)

    total = 0.0
    for sgn, z_lim in ((-1.0, z_fg_lo), (+1.0, z_bg_hi)):
        Pi_max = abs(float(cosmo.chi(z_lim)) - chi_o)
        segs = [(1e-8, min(R, Pi_max), 'lin')]
        if Pi_max > R:
            segs.append((R, Pi_max, 'log'))
        for a, b, kind in segs:
            if kind == 'lin':
                p, wp = gl_nodes(a, b, 32)
            else:
                lu, wu = gl_nodes(np.log(a), np.log(b), n_pi)
                p = np.exp(lu); wp = wu * p
            zP = z_of_Pi(sgn * p)
            wzv = w_z(zP, zob)
            s0v = np.interp(zP, z_coarse, s0)
            # (chi/chi_o)^2: measure x aperture-moment dilation (see far())
            chi_fac = ((chi_o + sgn * p) / chi_o) ** 2
            x_ex = np.sqrt(np.clip(1.0 - (p / R) ** 2, 0.0, None))
            inner = R ** 2 * (M0_full - np.interp(x_ex, xg, M0))
            total += float(np.sum(wp * chi_fac * wzv * s0v * inner))
    return 2.0 * np.pi * total


# ------------------------------------------------------------------
# End-to-end b_sel(theta): production SelBias vs frozen proposal
# ------------------------------------------------------------------

def frozen_precomp(sb, stack, lob, zob):
    """Assemble the frozen (P1, I1, I2) into a SelBias-compatible
    precomp dict so the downstream pipeline algebra (b_eff, plateaus,
    sigmoid) is shared bit-for-bit with production."""
    I1f, I2f, aux = assemble(stack, lob, zob)
    P1f = frozen_P1(stack, lob, zob)
    beff = sb.b_eff(lob, zob)
    return dict(lob=lob, zob=zob, P1=P1f, I1=I1f, I2=I2f,
                b_eff=beff, Delta_RND=P1f + beff * I2f,
                denom=I2f - I1f), aux


def fig_bsel(sb, pre_prod, pre_frozen, lob, zob, ltrs=(18.0, 15.0, 10.0)):
    theta_lam = sb._theta_lob(lob, zob)
    x = np.linspace(0.02, 3.0, 300)
    theta = x * theta_lam
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True,
                                    height_ratios=[3, 1.3])
    errs = {}
    for i, ltr in enumerate(ltrs):
        col = f'C{i}'
        b_p = sb.b_lob_theta(theta, ltr, zob, lob, precomp=pre_prod)
        b_f = sb.b_lob_theta(theta, ltr, zob, lob, precomp=pre_frozen)
        ax1.plot(x, b_p, col, lw=2,
                 label=fr'$\delta^{{\rm prj}}={lob - ltr:g}$ (production)')
        ax1.plot(x, b_f, col + '--', lw=1.6,
                 label='frozen proposal' if i == 0 else None)
        # normalise by the curve's own scale: b_sel crosses zero at small
        # delta_prj, so a pointwise ratio would blow up at the crossing
        rel = 100.0 * (b_f - b_p) / np.max(np.abs(b_p))
        ax2.plot(x, rel, col, lw=1.5)
        errs[ltr] = float(np.max(np.abs(rel)))
    ax1.axvline(0.5, color='0.6', ls=':', lw=1)
    ax1.set_ylabel(r'$b_{\rm sel}(\theta\,|\,\lambda^{\rm ob},'
                   r'\lambda^{\rm tr},z^{\rm ob})$')
    ax1.legend(fontsize=9)
    ax1.set_title(fr'$(\lambda^{{\rm ob}},z^{{\rm ob}})=({lob:g},{zob:g})$'
                  r' -- production grid vs frozen reformulation')
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_xlabel(r'$\theta/\theta_\lambda$')
    ax2.set_ylabel(r'$(b^{\rm froz}-b^{\rm prod})/\max|b^{\rm prod}|$ [%]')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pedag_frozen_bsel.png'), dpi=150)
    plt.close(fig)
    return errs


# ------------------------------------------------------------------
# Fitting functions: power-law xi, even-poly F  ->  closed forms
# ------------------------------------------------------------------

def fit_xi_powerlaw(stack, zob, r_lo, r_hi, n=60):
    """Local power-law xi(r) ~ (r/r0)^-gamma over [r_lo, r_hi].

    xi_NL is not a global power law over the whole near zone (log-log
    curvature); fit per segment. Over one octave [R, 2R] the fit is
    percent-level; a piecewise-power-law ladder (one gamma_j, r0_j per
    octave) keeps every segment closed-form integrable.
    """
    rs = np.geomspace(r_lo, r_hi, n)
    xi_v = stack['xi'](rs, zob)
    c = np.polyfit(np.log(rs), np.log(xi_v), 1)
    gamma = -c[0]
    r0 = np.exp(c[1] / gamma)
    resid = xi_v / np.exp(np.polyval(c, np.log(rs))) - 1.0
    return gamma, r0, rs, xi_v, resid


def fit_F_evenpoly(F, deg=5):
    """Least-squares even polynomial F(x) ~ sum_k c_k x^{2k} on [0, 2],
    weighted by the measure x dx.

    Fit in the rescaled variable t = (x/2)^2 in [0, 1] (raw x^{2k} up to
    x^{20} spans ~1e6 at x=2 and the normal equations lose the high-k
    coefficients); convert back via c_k = d_k / 4^k.
    """
    x = np.linspace(1e-4, 2.0, 400)
    t = (x / 2.0) ** 2
    Fx = F(x)
    Amat = np.stack([t ** k for k in range(deg + 1)], axis=1)
    wgt = np.sqrt(x)
    d, *_ = np.linalg.lstsq(Amat * wgt[:, None], Fx * wgt, rcond=None)
    return d / 4.0 ** np.arange(deg + 1)


def evenpoly_eval(c, x):
    x = np.asarray(x, dtype=float)
    return sum(ck * x ** (2 * k) for k, ck in enumerate(c))


def analytic_near_core(c, gamma, r0, R):
    """Closed-form near-zone core r in [R, 2R] (mu_lo = 0, mu_hi = 1):

        I_core = 4pi R^3 (R/r0)^{-gamma} sum_k c_k beta_k
                 (2^{2k+3-gamma} - 1) / (2k + 3 - gamma),
        beta_k = int_0^1 (1-mu^2)^k dmu = (2k)!! / (2k+1)!!.

    The 4pi = 2pi (solid measure) x 2 (both LoS signs of Pi = r mu).
    Assumes w_z ~ 1 over the core (|Pi| < 2R), exact to O((2R/sigma_chi)^2).
    """
    total = 0.0
    beta = 1.0
    for k, ck in enumerate(c):
        if k > 0:
            beta *= (2.0 * k) / (2.0 * k + 1.0)
        p = 2 * k + 3 - gamma
        total += ck * beta * (2.0 ** p - 1.0) / p
    return 4.0 * np.pi * R ** 3 * (R / r0) ** (-gamma) * total


def numeric_near_core(stack, F, zob, R, n_r=60, n_mu=60):
    """Same region, exact xi and exact F, w_z ~ 1 (for apples-to-apples)."""
    mu_t, mu_w = gl_nodes(0.0, 1.0, n_mu)
    lr, wr = gl_nodes(np.log(R), np.log(2.0 * R), n_r)
    rs = np.exp(lr)
    xi_r = stack['xi'](rs, zob)
    total = 0.0
    for r, wri, xiv in zip(rs, wr, xi_r):
        Rp = r * np.sqrt(np.maximum(1.0 - mu_t ** 2, 0.0))
        Gv = 2.0 * float(np.sum(mu_w * F(Rp / R)))
        total += wri * r ** 3 * xiv * Gv
    return 2.0 * np.pi * total


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def fig_kernels(F2, F1, lob, zob):
    x = np.linspace(1e-3, 2.2, 400)
    F2x, F1x = F2(x), F1(x)
    norm = F2(np.array([1e-3]))[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(x, F2x / norm, 'C3-', lw=2,
             label=r'$I_\lambda(x)/I_\lambda(0)$')
    ax1.plot(x, F1x / norm, 'C0-', lw=2,
             label=r'$I_\lambda^{\rm ls}(x)/I_\lambda(0)'
                   r' = \sigma(x)\,I_\lambda(x)/I_\lambda(0)$')
    ax1.plot(x, sigmoid_x(x), 'k--', lw=1.2, label=r'$\sigma(x)$')
    ax1.set_xlabel(r'$x = \theta/\theta_{\lambda,\mathrm{ob}} = s/R_{\rm excl}$')
    ax1.set_ylabel('frozen mass-integral moments')
    ax1.legend(fontsize=9); ax1.set_xlim(0, 2.2)
    ax1.axvline(2.0, color='0.7', lw=0.8)
    ax1.set_title(f'$(\\lambda^{{\\rm ob}},z^{{\\rm ob}})=({lob:g},{zob:g})$')

    # right panel: the aperture-overlap fraction f_A(x, rho) itself,
    # for a few projector sizes rho = theta_ltr / theta_lob
    for rho, c in zip((0.55, 0.70, 0.85, 1.00),
                      ('C0', 'C2', 'C1', 'C3')):
        fA = area_overlap(x[:, None], 1.0, np.array([rho]))[:, 0]
        ax2.plot(x, fA, c + '-', lw=2,
                 label=rf'$x_\lambda={rho:g}$'
                       rf'  ($\lambda={lob * rho ** 5:.1f}$)')
        ax2.axvline(1.0 + rho, color=c, ls=':', lw=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, 2.2)
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$f_A(x,x_\lambda) = A_{\rm ov}/(\pi\theta_\lambda^2)$')
    ax2.legend(fontsize=9, title='donor size (support end dotted)',
               title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pedag_frozen_kernels.png'), dpi=150)
    plt.close(fig)


def fig_radial(stack, aux, zob, lob):
    R, Pi_s = aux['R'], aux['Pi_s']
    rs = np.geomspace(R, aux['r_max_near'], 300)
    xi_r = stack['xi'](rs, zob)
    G2 = np.array([aux['G'](r, aux['F2']) for r in rs])
    G1 = np.array([aux['G'](r, aux['F1']) for r in rs])
    y2 = rs ** 2 * xi_r * G2
    y1 = rs ** 2 * xi_r * G1
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(rs / R, y2, 'C3-', lw=2,
              label=r'$r^2\,\xi_{\rm mm}(r)\,G_{\rm ss+ls}(r)$')
    ax.loglog(rs / R, y1, 'C0-', lw=2,
              label=r'$r^2\,\xi_{\rm mm}(r)\,G_{\rm ls}(r)$')
    ax.axvline(1.0, color='k', ls=':', lw=1,
               label=r'$r=R_{\rm excl}$ (hard lower limit)')
    ax.axvline(2.0, color='0.5', ls='--', lw=1,
               label=r'$r=2R_{\rm excl}$ (cylinder wall kink)')
    ax.set_xlabel(r'$r/R_{\rm excl}$')
    ax.set_ylabel('near-zone radial integrand')
    ax.set_title('Twin peaks + plateau are gone: smooth power-law decay')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pedag_frozen_radial.png'), dpi=150)
    plt.close(fig)


def fig_xifit(stack, zob, R, Pi_s):
    """Global single power law fails (log-log curvature); a per-octave
    piecewise power-law ladder is percent-level on every segment."""
    rs = np.geomspace(R, Pi_s, 300)
    xi_v = stack['xi'](rs, zob)
    g_all, r0_all, rs_all, _, res_all = fit_xi_powerlaw(stack, zob, R, Pi_s)

    n_seg = int(np.ceil(np.log2(Pi_s / R)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                    height_ratios=[3, 1])
    ax1.loglog(rs, xi_v, 'C3-', lw=2, label=r'$\xi_{\rm NL}(r)$ (halofit)')
    ax1.loglog(rs, (rs / r0_all) ** (-g_all), 'k--', lw=1.2,
               label=fr'single power law: $\gamma={g_all:.2f}$'
                     fr' (max resid {np.max(np.abs(res_all))*100:.0f}%)')
    worst = 0.0
    for j in range(n_seg):
        a = R * 2.0 ** j
        b = min(R * 2.0 ** (j + 1), Pi_s)
        g_j, r0_j, rs_j, _, res_j = fit_xi_powerlaw(stack, zob, a, b)
        worst = max(worst, float(np.max(np.abs(res_j))))
        ax1.loglog(rs_j, (rs_j / r0_j) ** (-g_j), 'C0-', lw=1,
                   alpha=0.9,
                   label='piecewise per octave' if j == 0 else None)
        ax2.semilogx(rs_j, 100 * res_j, 'C0-', lw=1)
    ax1.set_ylabel(r'$\xi_{\rm NL}(r)$'); ax1.legend(fontsize=9)
    ax1.set_title(fr'Near zone $[R_{{\rm excl}},\,{PI_S_OVER_R:g}'
                  fr'R_{{\rm excl}}]$: piecewise max resid '
                  fr'{worst*100:.1f}%')
    ax2.semilogx(rs_all, 100 * res_all, 'k--', lw=0.8)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_ylim(-6, 6)
    ax2.set_xlabel(r'$r\;[h^{-1}{\rm Mpc}]$')
    ax2.set_ylabel('resid [%]')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pedag_frozen_xifit.png'), dpi=150)
    plt.close(fig)
    return worst


def fig_farzone(stack, aux, zob):
    cosmo = stack['cosmo']
    chi_o = aux['chi_o']
    Pi_max_bg = float(cosmo.chi(aux['z_bg_hi'])) - chi_o
    Pis = np.geomspace(aux['Pi_s'], Pi_max_bg, 400)
    zPi = aux['z_of_Pi'](Pis)
    wzv = w_z(zPi, zob)
    xi_v = stack['xi'](Pis, zob)
    y = Pis * wzv * xi_v * aux['A2']          # log-measure integrand
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(Pis, y / y.max(), 'C3-', lw=2,
                label=r'$\pi\,w(\pi)\,A_{\rm ss+ls}\,\xi_{\rm mm}(\pi)$ (norm.)')
    ax.semilogx(Pis, wzv, 'k--', lw=1,
                label=r'photo-$z$ window $w_z$ (parabolic)')
    ax.axvline(105.0, color='0.5', ls=':', lw=1, label=r'BAO $\sim105\,h^{-1}$Mpc')
    ax.set_xlabel(r'$\pi = |\chi(z)-\chi_o|\;[h^{-1}{\rm Mpc}]$')
    ax.set_ylabel('far-zone integrand (background side)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pedag_frozen_farzone.png'), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------

def main():
    stack = build_stack()
    rows = []

    sb = SelBias(stack['cosmo'], stack['pk'], stack['hmf'], stack['bias'],
                 stack['mor'], xi_nl=stack['xi'])

    print(f"\n{'lob':>6} {'zob':>6} {'src':>7} "
          f"{'P1 err%':>9} {'I1 err%':>9} {'I2 err%':>9}")
    for (lob, zob) in POINTS:
        truth = quad_truth(stack, lob, zob)
        I1, I2, aux = assemble(stack, lob, zob)
        P1 = frozen_P1(stack, lob, zob)
        pre_prod = sb.bias_precompute(lob, zob)

        e_frozen = {k: abs(v / truth[k] - 1) * 100
                    for k, v in (('P1', P1), ('I1', I1), ('I2', I2))}
        e_prod = {k: abs(pre_prod[k] / truth[k] - 1) * 100
                  for k in ('P1', 'I1', 'I2')}
        print(f"{lob:>6.1f} {zob:>6.3f} {'prod':>7} "
              f"{e_prod['P1']:>8.4f}% {e_prod['I1']:>8.4f}% "
              f"{e_prod['I2']:>8.4f}%")
        print(f"{'':>6} {'':>6} {'frozen':>7} "
              f"{e_frozen['P1']:>8.4f}% {e_frozen['I1']:>8.4f}% "
              f"{e_frozen['I2']:>8.4f}%")
        rows.append(dict(lob=lob, zob=zob, P1=P1, I1=I1, I2=I2,
                         P1_quad=truth['P1'],
                         I1_quad=truth['I1'], I2_quad=truth['I2'],
                         P1_prod=pre_prod['P1'], I1_prod=pre_prod['I1'],
                         I2_prod=pre_prod['I2'],
                         err_P1_pct=e_frozen['P1'],
                         err_I1_pct=e_frozen['I1'],
                         err_I2_pct=e_frozen['I2']))

        pre_frozen = dict(lob=lob, zob=zob, P1=P1, I1=I1, I2=I2,
                          b_eff=pre_prod['b_eff'],
                          Delta_RND=P1 + pre_prod['b_eff'] * I2,
                          denom=I2 - I1)
        if (lob, zob) == POINTS[0]:
            bsel_errs = fig_bsel(sb, pre_prod, pre_frozen, lob, zob)
            print("  b_sel(theta) max |rel err| vs production: "
                  + ", ".join(f"ltr={k:g}: {v:.3f}%"
                              for k, v in bsel_errs.items()))
            rows[-1].update({f'bsel_err_ltr{int(k)}_pct': v
                             for k, v in bsel_errs.items()})

        if (lob, zob) == POINTS[0]:
            # moment sanity: int 2x F_2 dx == <b lam> (closed-form identity)
            _, w_eff, _ = frozen_weights(stack, lob, zob)
            blam = float(np.sum(w_eff))
            mom = 2.0 * aux['A2'] / aux['R'] ** 2
            print(f"  moment identity: int 2x F2 dx = {mom:.6e} "
                  f"vs <b lam> = {blam:.6e} "
                  f"({abs(mom/blam-1)*100:.4f}% -- closed form, exact)")

            # broad fit (figure: shows why one global power law fails)
            gamma, r0, rs, xi_v, resid = fit_xi_powerlaw(
                stack, zob, aux['R'], aux['Pi_s'])
            print(f"  xi global power law [R, {PI_S_OVER_R:g}R]: "
                  f"gamma={gamma:.4f} r0={r0:.4f}, "
                  f"max |resid| = {np.max(np.abs(resid))*100:.2f}% "
                  f"(too curved -- use piecewise)")
            # local octave fit for the closed-form core [R, 2R]
            g_c, r0_c, rs_c, xi_c, res_c = fit_xi_powerlaw(
                stack, zob, aux['R'], 2.0 * aux['R'])
            print(f"  xi local power law [R, 2R]: gamma={g_c:.4f} "
                  f"r0={r0_c:.4f}, max |resid| = "
                  f"{np.max(np.abs(res_c))*100:.2f}%")

            c2 = fit_F_evenpoly(aux['F2'], deg=10)
            c1 = fit_F_evenpoly(aux['F1'], deg=10)
            core_num = numeric_near_core(stack, aux['F2'], zob, aux['R'])
            core_ana = analytic_near_core(c2, g_c, r0_c, aux['R'])
            print(f"  near-core closed form (I2 piece): analytic "
                  f"{core_ana:.5e} vs numeric {core_num:.5e} "
                  f"({abs(core_ana/core_num-1)*100:.2f}%)")
            core_num1 = numeric_near_core(stack, aux['F1'], zob, aux['R'])
            core_ana1 = analytic_near_core(c1, g_c, r0_c, aux['R'])
            print(f"  near-core closed form (I1 piece): analytic "
                  f"{core_ana1:.5e} vs numeric {core_num1:.5e} "
                  f"({abs(core_ana1/core_num1-1)*100:.2f}%)")

            fig_kernels(aux['F2'], aux['F1'], lob, zob)
            fig_radial(stack, aux, zob, lob)
            worst_pw = fig_xifit(stack, zob, aux['R'], aux['Pi_s'])
            print(f"  piecewise (per-octave) xi fit: max |resid| = "
                  f"{worst_pw*100:.2f}%")
            fig_farzone(stack, aux, zob)
            rows[-1].update(gamma=gamma, r0=r0,
                            core_I2_analytic_err_pct=abs(
                                core_ana / core_num - 1) * 100,
                            core_I1_analytic_err_pct=abs(
                                core_ana1 / core_num1 - 1) * 100)

    out = os.path.join(CACHE_DIR, "frozen_assembly.csv")
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out}")
    print(f"Figures in {FIG_DIR}/pedag_frozen_*.png")


if __name__ == "__main__":
    main()
