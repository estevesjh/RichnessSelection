"""Package-wide defaults.

Grid sizes, reference paths, and fiducial cosmology live here so every
other module can be imported without relying on module globals at its
own call site. Override per-call via constructor arguments.
"""
from dataclasses import dataclass

DEFAULT_COSMO_PARAMS = dict(
    Om0=0.286,
    Ob0=0.047,
    H0=70.0,
    ns=0.96,
    sigma8=0.82,
    mnu=0.0,
)


@dataclass(frozen=True)
class GridConfig:
    """Integration grid sizes used by Sigma_prj pipeline.

    Defaults chosen for sub-0.01% precision on P[1], I_1, I_2 against
    a scipy.integrate.quad reference at (lob=20, zob=0.5):
      - Nz=80 (40 fg + 40 bg log-spaced in |Delta chi|, Option E)
      - Nth=10 GL nodes (split-at-exclusion; theta_excl(z) is the
        lower limit of the GL interval so the integrand is smooth)
      - NM=24 GL nodes (mass integrand is smooth)

    Nz_bias is a separate, smaller z-node budget used only by
    SelBias._P_operator (P[1], I_1, I_2) -- SigmaPrj/DeltaSigmaPrj
    keep reading Nz (unchanged) for their own z-integration via
    SelBias._z_grid. The ring+outer split in _z_grid enforces hard
    floors (n_ring >= 9, n_outer >= 15 per side), so any Nz below
    ~40 collapses to the same 39-node grid -- 48 sits just above
    that floor with headroom. Convergence vs an Nz=200,Nth=30
    reference across the 12 DES Y3 (lob,zob) bins: worst-case error
    at Nz=48 is 0.045% (well inside the 0.1% tolerance the existing
    quad-matched tests already enforce), for a ~1.7x reduction in
    the expensive per-z (theta,lambda,M) contraction vs Nz=80. See
    docs/richness_selection.tex Sec. "z-axis" and the z-axis
    analytic-exclusion review for the derivation and the convergence
    table this default is pinned from.
    """
    Nz: int = 80
    Nz_bias: int = 48
    NM: int = 24
    Nth: int = 10
    ln_M_min: float = 12.5 * 2.302585092994046   # log(10^12.5)
    ln_M_max: float = 15.5 * 2.302585092994046   # log(10^15.5)
    ltr_grid_size: int = 16


DEFAULT_GRID = GridConfig()

NFW_TABLE_DIR = "/global/common/software/des/jesteves/y3_cluster_cpp/data/nfw_off_center"

# Upper bound on the theta integral, expressed in cMpc/h of projected radius.
# theta_max = R_MAX_CMPCH / D_A(z_ob).  Matches the Sigma_NFW truncation at the
# same scale in Matteo's SelectionBias notebook (twoD_prj_NFW R_max = 30).
R_MAX_CMPCH = 30.0
