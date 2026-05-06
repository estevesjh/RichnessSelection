"""Shared setup for the Sigma_prj validation scripts.

Centralises the local NFW-table path and the heavy-object builder so
each script can `from _common import build_stack, CACHE_DIR, ...`.
"""
from __future__ import annotations
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

NFW_TABLE_DIR = os.environ.get(
    "RICHNESS_SELECTION_NFW_DIR",
    "/Users/esteves/Documents/Projetos/y3_cluster_cpp/data/nfw_off_center",
)
CACHE_DIR = os.path.join(_HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def build_stack():
    """Cosmology + PkGrid + HMF + Bias + MOR + XiNL (built) + NFW + SelBias.

    Returns a dict with named handles.
    """
    from richness_selection import (
        Cosmology, PkGrid, HMF, Bias, MOR, NFWMiscentered, SelBias,
    )
    from richness_selection.sigma_m import SigmaM
    from richness_selection.xi_nl import XiNL

    cosmo = Cosmology()
    pk = PkGrid(cosmo)
    sm = SigmaM(pk)
    hmf = HMF(sm)
    bias = Bias(sm)
    mor = MOR()
    print("[setup] Building XiNL / halofit ...", flush=True)
    t0 = time.time()
    xi = XiNL(cosmo); xi.build()
    print(f"[setup] XiNL built in {time.time() - t0:.1f}s", flush=True)
    nfw = NFWMiscentered(cosmo, table_dir=NFW_TABLE_DIR)
    sb = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
    return dict(cosmo=cosmo, pk=pk, sm=sm, hmf=hmf, bias=bias, mor=mor,
                xi=xi, nfw=nfw, sb=sb)
