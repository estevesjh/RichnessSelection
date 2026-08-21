"""Frozen-physics lensing observable -- docs/richness_selection_frozen.tex
Sec. "Extension: frozen physics for <DeltaSigma_prj>".

``FrozenDeltaSigmaPrj`` subclasses ``DeltaSigmaPrj`` and overrides only
``__call__``: the theta-grid, the (z, M) context, the kernel (signed
``DeltaSigma_mis`` lookup) and the marginalised b_rm(theta) are all
inherited, so any residual against production is purely the Sec.-4
reduction.

The reduction, per the note:

  rnd channel (eq. rnd_exact / ntilde) -- EXACT, no freeze: the kernel
  is z-free, so the z-integral commutes past it.  The contraction
  w_rnd_M = int dz (dV/dzdOm) w(z) n(M,z) is hoisted out of the theta
  loop (production recomputes this theta-independent object at every
  node).  The frozen rnd channel is therefore machine-identical to
  production.

  cl channel (eqs. nb_drift, Psi_lens, DSprj_frozen) -- freeze the
  shape of n(M,z) b(M,z) at zob, keep the amplitude via the drift
  a_b(z) anchored on r_s(M) (the R200 scale, the kernel's leading mass
  weighting: DSmis ~ r_s rho_eff at fixed R/r_s).  The per-theta
  (z, M) block then factorises into Psi(theta) (1-D in z, vectorised
  over the whole theta grid at once, exclusion mask identical to
  production) times the frozen M-contraction of the kernel stack.

Cost: the O(N_M x N_z) inner block per theta-node becomes one
O(N_z)-vectorised Psi plus an O(N_M) matvec; the remaining per-node
cost is the shared kernel lookup.
"""
from __future__ import annotations

import numpy as np

from .delta_sigma_prj import DeltaSigmaPrj

__all__ = ["FrozenDeltaSigmaPrj"]


class FrozenDeltaSigmaPrj(DeltaSigmaPrj):
    """DeltaSigmaPrj with the frozen-physics (z, M) factorisation."""

    def _kernel_stack(self, R, thetas, ctx):
        """(Nth, NM, NR) signed DeltaSigma_mis stack in one grid-spline
        call per M-row.

        For fixed M, lnxmis(theta) = ln(theta D_A / r_s) is monotonic in
        theta, so the whole (theta, R) plane is a tensor-grid evaluation
        of the bilinear lookup -- NM grid calls total, instead of the
        parent closure's NM calls per theta-node.  Clipping duplicates
        are routed through np.unique to keep the grid axis strictly
        increasing.  Numerically identical to the parent closure (same
        spline, same clipping; linear-space values, no exp).
        """
        rs_M = ctx["rs_M"]; rho_eff = ctx["rho_s"]; D_A_o = ctx["D_A_o"]
        _dsig_spl = self.nfw._dsig_spl
        _lnx_lo = self.nfw._lnx_lo; _lnx_hi = self.nfw._lnx_hi
        _lnxmis_lo = self.nfw._lnxmis_lo; _lnxmis_hi = self.nfw._lnxmis_hi
        ln_R = np.clip(np.log(R)[None, :] - np.log(rs_M)[:, None],
                       _lnx_lo, _lnx_hi)                     # (NM, NR)
        prefac_M = 2.0 * rs_M * rho_eff * 1.0e-12            # (NM,)

        out = np.empty((thetas.size, rs_M.size, R.size))
        ln_theta_DA = np.log(thetas * D_A_o)                 # (Nth,)
        for iM in range(rs_M.size):
            lnxmis = np.clip(ln_theta_DA - np.log(rs_M[iM]),
                             _lnxmis_lo, _lnxmis_hi)
            vals, inv = np.unique(lnxmis, return_inverse=True)
            grid = _dsig_spl(vals, ln_R[iM])                 # (Nvals, NR)
            out[:, iM, :] = prefac_M[iM] * grid[inv]
        return out

    def __call__(self, R, lob, zob, *, return_decomposition: bool = False):
        R = np.atleast_1d(R).astype(float)
        ctx = self._build_zM_context(lob, zob)
        pre = self.sel_bias.bias_precompute(lob, zob)
        thetas, w_theta, th_info = self._theta_grid(lob, zob, R, ctx)
        bsel_vals = self._bsel_at(thetas, lob, zob, pre)

        chi_z = ctx["chi_z"]
        outer_weight = ctx["outer_weight"]
        Ms = ctx["Ms"]
        M_weight = ctx["M_weight"]
        n_mz = ctx["n_mz"]          # (NM, Nz)
        bM_mz = ctx["bM_mz"]        # (NM, Nz)
        theta_excl_z = ctx["theta_excl_z"]
        chi_o = ctx["chi_o"]
        rs_M = ctx["rs_M"]

        # ---- rnd channel: exact tilde-n hoist (eq. ntilde) ------------
        w_rnd_M = M_weight * (n_mz * outer_weight[None, :]).sum(axis=1)

        # ---- cl channel: frozen shape + r_s-anchored drift ------------
        # (eq. nb_drift; n(M,zob), b(M,zob) are columns of the ctx grids
        # only if zob is a node -- evaluate exactly instead)
        n_o = self.hmf(Ms, zob)
        b_o = self.bias(Ms, zob)
        anchor = M_weight * rs_M
        a_b = (anchor @ (n_mz * bM_mz)) / float(np.sum(anchor * n_o * b_o))
        w_cl_M = M_weight * n_o * b_o                       # (NM,)

        # ---- Psi(theta): one vectorised (theta, z) pass (eq. Psi_lens)
        cos_th = np.cos(thetas)                             # (Nth,)
        dchi = np.sqrt(np.maximum(
            chi_z[None, :] ** 2 + chi_o ** 2
            - 2.0 * chi_z[None, :] * chi_o * cos_th[:, None], 0.0))
        xi = self.xi_NL(dchi.ravel(), zob).reshape(dchi.shape)
        mask = thetas[:, None] > theta_excl_z[None, :]
        Psi = (np.where(mask, xi, 0.0)
               * (outer_weight * a_b)[None, :]).sum(axis=1)  # (Nth,)

        # ---- assembly (eq. DSprj_frozen): no theta loop at all --------
        S = self._kernel_stack(R, thetas, ctx)               # (Nth, NM, NR)
        prefac = w_theta * 2.0 * np.pi * np.sin(thetas)      # (Nth,)
        out_rnd = np.einsum("t,tmr,m->r", prefac, S, w_rnd_M)
        out_cl = np.einsum("t,tmr,m->r",
                           prefac * bsel_vals * Psi, S, w_cl_M)

        if return_decomposition:
            return dict(total=out_rnd + out_cl,
                        rnd=out_rnd, cl=out_cl,
                        theta_info=th_info)
        return out_cl
