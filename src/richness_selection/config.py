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
    """Gauss-Legendre integration grid sizes used by Sigma_prj pipeline."""
    Nz: int = 16
    NM: int = 24
    Nth: int = 20
    ln_M_min: float = 12.5 * 2.302585092994046   # log(10^12.5)
    ln_M_max: float = 15.5 * 2.302585092994046   # log(10^15.5)
    ltr_grid_size: int = 16


DEFAULT_GRID = GridConfig()

NFW_TABLE_DIR = "/global/common/software/des/jesteves/y3_cluster_cpp/data/nfw_off_center"
