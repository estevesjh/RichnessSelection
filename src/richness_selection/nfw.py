"""Miscentered NFW surface density via precomputed "single" lookup table.

Tables (loaded once at construction):
    table_1000_1e-03_5e+03_single_logx.txt           999 pts   ln(R/R_s)
    table_1000_1e-03_5e+03_single_logxmis.txt        249 pts   ln(R_mis/R_s)
    table_1000_1e-03_5e+03_log_sigma_single.txt      250 x 1000  ln f(x, x_mis)

The shipped ``log_deltasigma_single`` table is deliberately NOT read:
its ``ln g`` storage cannot represent the negative branch
``DeltaSigma_mis(R < R_mis) < 0`` and floors it to ~0, which inflates
the two-halo cl piece by ~1.5x and breaks the uniform-field rnd
cancellation.  ``_dsig_spl`` is instead the *signed* excess
reconstructed at construction from the Sigma table (per x_mis row,
``bar-Sigma(<R) - Sigma(R)``), splined in linear space -- see
``_build_signed_dsigma_spline``.

C++ convention (authoritative: y3_cluster_cpp ``NFW_SIGMA_MIS`` and
``NFW_DSIGMA_MIS``)
----------------------------------------------------------------------

The stored tables ``f`` and ``g`` are the same ones used by
``y3_cluster_cpp``.  The C++ reconstruction recipe (mirrored here) is::

    r_200  = cbrt( 3 M / (800 pi rho_crit) )        [cMpc/h]
    r_s    = r_200 / c,                c = 4                    (default)
    delta_c = (200 c^3 / 3) / (ln(1+c) - c/(1+c))
    rho_eff = delta_c * rho_crit * Omega_m                      (cMpc/h units)

    Sigma_mis      = 2 * r_s * rho_eff * exp(ln f) * 1e-12      [Msun/h / pc^2]
    DeltaSigma_mis = 2 * r_s * rho_eff * exp(ln g) * 1e-12      [Msun/h / pc^2]

The ``1e-12`` factor converts ``Msun/h / (cMpc/h)^2`` (the natural
units of ``r_s * rho_eff``) into ``Msun/h / pc^2`` so downstream code
carries the C++ lensing-observable units directly.

Divergence from the previous (paper-Eq.-14) convention
------------------------------------------------------

- ``c`` default was 5; the C++ side uses 4.
- ``r_200`` was ``r_200m`` via ``rho_m``; C++ uses ``r_200c`` via ``rho_crit``.
- ``Sigma_mis`` had a ``2 * (2 pi r_s rho_s)`` prefactor (pair of factors
  chosen to land on Costanzi 2026 Eq. 14 / Wright & Brainerd 2000 convention);
  C++ uses ``2 * r_s * rho_eff * 1e-12`` — no ``2 pi``, natural C++ units.

The two conventions are related by a mass-dependent rescaling (different
``r_s`` *and* a per-call constant), so callers are *not* isomorphic up to
an overall constant. Regression goldens changed when this module was
switched over; see ``docs/sigma_prj_refactor.md`` Section 2.
"""
from __future__ import annotations
import os
import numpy as np
from scipy.interpolate import RectBivariateSpline

from .cosmology import Cosmology
from .config import NFW_TABLE_DIR


# rho_crit,0 at H0 = 100 km/s/Mpc, in Msun/h / (cMpc/h)^3.
RHO_CRIT_0 = 2.77533742639e11

# (cMpc/h)^-2 -> pc^-2  conversion.  Natural (r_s * rho_eff) units are
# Msun/h / (cMpc/h)^2; multiply by (Mpc/pc)^-2 = 1e-12 to get Msun/h/pc^2.
CMPCH2_TO_PC2 = 1.0e-12


# Taylor windows about the x = 1 kink of the W&B kernels, where the
# direct closed forms lose ~1e-16/|x-1| to 0/0 cancellation (the
# clustered miscentering quadrature nodes land exactly there).  Exact
# sympy coefficients, ported from CLensPy ``clenspy.halo.nfw``; with
# |x-1| <= 1e-2 the truncation error is |c_9| d^9 ~ 1e-19 while the
# direct branch outside stays < ~2e-14.
_WB_F_SERIES = np.array([
    1 / 3, -2 / 5, 13 / 35, -20 / 63, 61 / 231, -94 / 429,
    1181 / 6435, -1896 / 12155, 6223 / 46189,
])
_WB_G_SERIES = np.array([
    10 / 3 - 4 * np.log(2), -88 / 15 + 8 * np.log(2),
    296 / 35 - 12 * np.log(2), -3508 / 315 + 16 * np.log(2),
    1373 / 99 - 20 * np.log(2), -49930 / 3003 + 24 * np.log(2),
    9601 / 495 - 28 * np.log(2), -186556 / 8415 + 32 * np.log(2),
    11521457 / 461890 - 36 * np.log(2),
])
_WB_SERIES_WINDOW = 1e-2


def wright_brainerd_f(x, eps=_WB_SERIES_WINDOW):
    """Wright & Brainerd (2000) centered NFW surface-density kernel f(x):
    ``Sigma_cen = 2 r_s rho_s f(x)``.  Analytic, with a Taylor window
    at the x = 1 kink (ported from CLensPy ``clenspy.halo.nfw``)."""
    x = np.array(x, dtype=float)
    res = np.empty_like(x)
    mask_l = x < 1.0 - eps
    mask_g = x > 1.0 + eps
    mask_c = ~(mask_l | mask_g)
    res[mask_c] = np.polynomial.polynomial.polyval(
        x[mask_c] - 1.0, _WB_F_SERIES)

    xl = x[mask_l]
    res[mask_l] = (1.0 - 2.0 / np.sqrt(1.0 - xl ** 2)
                   * np.arctanh(np.sqrt((1.0 - xl) / (1.0 + xl)))) \
        / (xl ** 2 - 1.0)

    xg = x[mask_g]
    res[mask_g] = (1.0 - 2.0 / np.sqrt(xg ** 2 - 1.0)
                   * np.arctan(np.sqrt((xg - 1.0) / (xg + 1.0)))) \
        / (xg ** 2 - 1.0)
    return res


def truncated_sigma_kernel(c, n_x: int = 1500, n_los: int = 96):
    """Projected NFW hard-truncated at the 3D halo boundary r_t = c r_s.

    Returns ``(lnx_grid, ft_grid)`` with ``Sigma_t = 2 r_s rho_s f_t(x)``
    and ``f_t(x) = 0`` for ``x >= c``.  The column is the exact LoS
    integral of the NFW density over ``|l| < sqrt(c^2 - x^2)`` (GL
    quadrature; the integrand is smooth).  Mass conservation is exact:
    ``int 2 pi x f_t dx = m_3(c) = ln(1+c) - c/(1+c)``, i.e. the halo
    projects all of (and only) its 3D mass -- the property the
    untruncated profile lacks (log-divergent 2D tail), which breaks the
    two-halo neighbour mass budget.
    """
    from numpy.polynomial.legendre import leggauss
    s, w = leggauss(int(n_los))
    s = 0.5 * (s + 1.0)          # [0, 1]
    w = 0.5 * w
    lnx = np.linspace(np.log(1e-3), np.log(c * (1.0 - 1e-9)), int(n_x))
    x = np.exp(lnx)
    l_max = np.sqrt(np.maximum(c * c - x * x, 0.0))     # (Nx,)
    l = l_max[:, None] * s[None, :]                     # (Nx, Nlos)
    r = np.sqrt(x[:, None] ** 2 + l ** 2)
    rho = 1.0 / (r * (1.0 + r) ** 2)                    # NFW / rho_s
    # Sigma_t / (2 r_s rho_s) = (1/2) * 2 int_0^lmax rho dl / (2)   ->
    # Sigma_t = 2 rho_s r_s int_0^lmax rho dl  =>  f_t = int_0^lmax rho dl
    ft = (rho @ w) * l_max
    return lnx, ft


def wright_brainerd_g(x, eps=_WB_SERIES_WINDOW):
    """Wright & Brainerd (2000) centered NFW excess kernel g(x):
    ``DeltaSigma_cen = r_s rho_s g(x)``.  Analytic, with a Taylor
    window at the x = 1 kink (ported from CLensPy
    ``clenspy.halo.nfw.NfwProfile._gNfw``)."""
    x = np.array(x, dtype=float)
    res = np.empty_like(x)
    mask_l = x < 1.0 - eps
    mask_g_ = x > 1.0 + eps
    mask_c = ~(mask_l | mask_g_)
    res[mask_c] = np.polynomial.polynomial.polyval(
        x[mask_c] - 1.0, _WB_G_SERIES)

    xl = x[mask_l]
    s = np.sqrt(1.0 - xl ** 2)
    atanh = np.arctanh(s / (1.0 + xl))
    res[mask_l] = (8.0 * atanh / (xl ** 2 * s)
                   + 4.0 / xl ** 2 * np.log(xl / 2.0)
                   - 2.0 / (xl ** 2 - 1.0)
                   + 4.0 * atanh / ((xl ** 2 - 1.0) * s))

    mask_g = x > 1.0 + eps
    xg = x[mask_g]
    s = np.sqrt(xg ** 2 - 1.0)
    atan = np.arctan(s / (1.0 + xg))
    res[mask_g] = (8.0 * atan / (xg ** 2 * s)
                   + 4.0 / xg ** 2 * np.log(xg / 2.0)
                   - 2.0 / (xg ** 2 - 1.0)
                   + 4.0 * atan / ((xg ** 2 - 1.0) ** 1.5))
    return res


class NFWMiscentered:
    """Miscentered NFW Sigma(R | M, z, R_mis) from the Y3 lookup table.

    C++ convention (``y3_cluster_cpp::NFW_SIGMA_MIS``):
    ``c = 4`` default, ``r_200`` via ``rho_crit``, output in
    ``Msun/h / pc^2``.  See module docstring.
    """

    def __init__(self, cosmo: Cosmology, table_dir=NFW_TABLE_DIR,
                 c: float = 4.0, rho_crit: float = RHO_CRIT_0,
                 rho_mult: float | None = None,
                 kind: str = "cpp"):
        if kind not in ("cpp", "m200m"):
            raise ValueError(f"kind must be 'cpp' or 'm200m', got {kind!r}")
        self.cosmo = cosmo
        self.kind = kind
        self.c = float(c)
        self._rho_crit = float(rho_crit)
        # Default rho_mult = Omega_m (matches C++ ``rho_mult = omega_m``).
        self._rho_mult = (float(rho_mult) if rho_mult is not None
                          else float(cosmo.Om0))
        self._log_x = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_single_logx.txt"))
        self._log_xmis = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_single_logxmis.txt"))
        log_sigma = np.loadtxt(os.path.join(
            table_dir, "table_1000_1e-03_5e+03_log_sigma_single.txt"))

        lnxmis = self._log_xmis[: log_sigma.shape[0]]
        lnx = self._log_x[: log_sigma.shape[1]]
        self._lnx_lo, self._lnx_hi = lnx[0], lnx[-1]
        self._lnxmis_lo, self._lnxmis_hi = lnxmis[0], lnxmis[-1]
        self._spl = RectBivariateSpline(lnxmis, lnx, log_sigma, kx=1, ky=1)
        self._dsig_spl = self._build_signed_dsigma_spline(lnxmis)

    def _build_signed_dsigma_spline(self, lnxmis, n_dense: int = 2000,
                                    n_nodes: int = 128):
        """Signed DeltaSigma_mis lookup, fully analytic (the shipped
        ``log_deltasigma_single`` table is NOT used: its ``ln g``
        storage floors the negative branch
        ``DeltaSigma_mis(R < R_mis) < 0`` to ~0, which inflates the
        two-halo cl piece by ~1.5x and breaks the uniform-field rnd
        cancellation).

        Closed form (ported from CLensPy
        ``clenspy.lensing.miscentering``): with the law-of-cosines
        nodes ``u(t)^2 = (x - x_mis)^2 + 4 x x_mis sin^2(t/2)``,

            Sigma_mis(x | x_mis)     = (1/pi) int_0^pi f(u(t)) dt
            Sbar_mis(<x | x_mis)     = (1/(2 pi x^2)) int_0^pi
                                       [u^2 + x^2 - x_mis^2] Sbar(u) dt
            gs = Sbar_mis - Sigma_mis                       (signed)

        in units of ``2 r_s rho_s``, where ``f = wright_brainerd_f`` and
        ``Sbar(<u) = g_WB(u)/2 + f(u)`` are the analytic centered W&B
        kernels.  The by-parts reduction plus this substitution is
        smooth for both lobes -- no cusp at x = x_mis, no endpoint
        singularity; fixed Gauss-Legendre on the clustering map
        ``t = pi s^2`` (Jacobian in the weights) resolves the integrable
        log at u -> |x - x_mis|.  Stored as a linear-space bilinear
        spline on ``(lnxmis, lnx_dense)``; one-time <~1 s cost, per-eval
        cost identical to the old lookup.
        """
        from numpy.polynomial.legendre import leggauss
        s, w = leggauss(int(n_nodes))
        s = 0.5 * (s + 1.0)
        wt = 0.5 * w * 2.0 * np.pi * s               # includes dt = 2 pi s ds
        t = np.pi * s * s

        lnx_g = np.linspace(self._lnx_lo, self._lnx_hi, int(n_dense))
        x_g = np.exp(lnx_g)
        xm_g = np.exp(np.asarray(lnxmis, dtype=float))
        sin2 = np.sin(t / 2.0) ** 2

        gs = np.empty((xm_g.size, x_g.size))
        for i, xm in enumerate(xm_g):                # per x_mis row
            u = np.sqrt((x_g[:, None] - xm) ** 2
                        + 4.0 * x_g[:, None] * xm * sin2[None, :])
            f_u = wright_brainerd_f(u)
            sbar_u = 0.5 * wright_brainerd_g(u) + f_u
            sig_mis = (f_u @ wt) / np.pi
            kern = u * u + (x_g ** 2 - xm ** 2)[:, None]
            sbar_mis = ((kern * sbar_u) @ wt) / (2.0 * np.pi * x_g ** 2)
            gs[i] = sbar_mis - sig_mis
        return RectBivariateSpline(np.asarray(lnxmis, dtype=float),
                                   lnx_g, gs, kx=1, ky=1)

    def _rs_and_rhos(self, M, z):
        """``(r_s, rho_s)`` per the constructor ``kind``.

        ``kind="cpp"`` (default, y3_cluster_cpp parity): fixed ``c``,
        ``r_200c`` via rho_crit, ``rho_eff = delta_c rho_crit rho_mult``
        -- the reconstructed halo carries ``Omega_m * M``.

        ``kind="m200m"`` (Costanzi-notebook convention): ``r_200m`` via
        the mean density, Duffy-like ``c(M, z)``, mass-conserving
        ``rho_s = rho_m delta_char`` -- the reconstructed halo carries
        ``M`` exactly.  ``c`` is evaluated at the passed ``z`` (the
        prj pipelines pass ``zob``; the < 5% drift of c over the
        correlated LoS support is neglected to keep the kernels
        z-hoistable).
        """
        if self.kind == "m200m":
            rho_m = self._rho_crit * float(self.cosmo.Om0)
            r_200 = np.cbrt(3.0 * M / (800.0 * np.pi * rho_m))
            c = 10.14 * (np.asarray(M, dtype=float) / 2.0e12) ** -0.081 \
                * (1.0 + z) ** -1.01
            rs = r_200 / c
            fc = np.log(1.0 + c) - c / (1.0 + c)
            return rs, rho_m * (200.0 * c ** 3 / 3.0) / fc
        c = self.c
        rhoc = self._rho_crit
        r_200 = np.cbrt(3.0 * M / (800.0 * np.pi * rhoc))
        rs = r_200 / c
        fc = np.log(1.0 + c) - c / (1.0 + c)
        delta_c = (200.0 * c ** 3 / 3.0) / fc
        return rs, delta_c * rhoc * self._rho_mult

    def sigma_grid(self, R_arr, R_mis_arr, M, z):
        """Sigma_mis(R, R_mis | M) in the C++ convention [Msun/h / pc^2].

        Returns a (N_Rmis, N_R) array.
        """
        rs, rho_eff = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        lnF = self._spl(lnxmis, lnx)
        norm = 2.0 * rs * rho_eff
        return norm * np.exp(lnF) * CMPCH2_TO_PC2

    def delta_sigma_grid(self, R_arr, R_mis_arr, M, z):
        """Signed DeltaSigma_mis(R, R_mis | M) [Msun/h / pc^2].

        Returns a (N_Rmis, N_R) array.  Same ``2 * r_s * rho_eff * 1e-12``
        prefactor as ``sigma_grid``.  Values come from the signed
        reconstruction (``_build_signed_dsigma_spline``): negative for
        R < R_mis, Wright & Brainerd centered excess at R_mis -> 0.
        """
        rs, rho_eff = self._rs_and_rhos(M, z)
        lnx = np.clip(np.log(R_arr / rs), self._lnx_lo, self._lnx_hi)
        lnxmis = np.clip(np.log(R_mis_arr / rs),
                         self._lnxmis_lo, self._lnxmis_hi)
        gs = self._dsig_spl(lnxmis, lnx)
        norm = 2.0 * rs * rho_eff
        return norm * gs * CMPCH2_TO_PC2
