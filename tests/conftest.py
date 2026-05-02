import pytest

from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, SelBias, SigmaPrj,
    DEFAULT_GRID,
)
from richness_selection.sigma_m import SigmaM


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
def sel_bias(cosmo, pk, hmf, bias, mor):
    return SelBias(cosmo, pk, hmf, bias, mor)


@pytest.fixture(scope="session")
def nfw(cosmo):
    return NFWMiscentered(cosmo)


@pytest.fixture(scope="session")
def sigma_prj(cosmo, sel_bias, nfw):
    return SigmaPrj(cosmo, sel_bias, nfw)
