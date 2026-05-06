import os
import pytest

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, XiNL, SelBias,
    SigmaPrj, DEFAULT_GRID,
)
from richness_selection.sigma_m import SigmaM


NFW_TABLE_DIR = os.environ.get(
    "RICHNESS_SELECTION_NFW_DIR",
    "/Users/esteves/Documents/Projetos/y3_cluster_cpp/data/nfw_off_center",
)


@pytest.fixture(scope="session")
def xi_nl(cosmo):
    return XiNL(cosmo)


@pytest.fixture(scope="session")
def cosmo():
    return Cosmology(Om0=0.286, Ob0=0.047, H0=70.0, ns=0.96,
                     sigma8=0.82, mnu=0.0)


@pytest.fixture(scope="session")
def pk(cosmo):
    return PkGrid(cosmo)


@pytest.fixture(scope="session")
def sigma_m(pk):
    return SigmaM(pk)


@pytest.fixture(scope="session")
def hmf(sigma_m):
    return HMF(sigma_m)


@pytest.fixture(scope="session")
def bias(sigma_m):
    return Bias(sigma_m)


@pytest.fixture(scope="session")
def mor():
    return MOR()


@pytest.fixture(scope="session")
def sel_bias(cosmo, pk, hmf, bias, mor, xi_nl):
    return SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi_nl)


@pytest.fixture(scope="session")
def nfw(cosmo):
    if not os.path.exists(NFW_TABLE_DIR):
        pytest.skip(f"NFW table dir not found: {NFW_TABLE_DIR}. "
                    "Set RICHNESS_SELECTION_NFW_DIR to override.")
    return NFWMiscentered(cosmo, table_dir=NFW_TABLE_DIR)


@pytest.fixture(scope="session")
def sigma_prj(cosmo, sel_bias, nfw):
    return SigmaPrj(cosmo, sel_bias, nfw)
