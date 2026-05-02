"""Linear matter power spectrum P(k, z) via CAMB with in-process cache.

One PkGrid per Cosmology; CAMB is called exactly once in __init__, and
results are served via bilinear interpolation over (log k, z).  The
class-level _CACHE keyed on the Cosmology tuple lets distinct PkGrid
constructions for the same cosmology reuse the same underlying grid.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np

from .cosmology import Cosmology


@lru_cache(maxsize=8)
def _camb_pk_grid(cosmo_key, z_max, nz, kmin, kmax, nk):
    """Cached CAMB call.  Key is the tuple from Cosmology.key plus grid specs."""
    import camb
    Om0, Ob0, H0, ns, sigma8, mnu = cosmo_key

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ob0 * (H0 / 100.0) ** 2,
                       omch2=(Om0 - Ob0) * (H0 / 100.0) ** 2,
                       mnu=mnu)
    pars.InitPower.set_params(ns=ns)
    zs = np.linspace(0.0, z_max, nz)
    pars.set_matter_power(redshifts=zs[::-1].tolist(), kmax=kmax * 1.1)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)

    kh, z_out, pk_lin = results.get_matter_power_spectrum(
        minkh=kmin, maxkh=kmax, npoints=nk
    )
    z_out = np.asarray(z_out)
    pk_lin = np.asarray(pk_lin)
    order = np.argsort(z_out)
    z_out = z_out[order]
    pk_lin = pk_lin[order]

    # Rescale to requested sigma8 (CAMB's sigma8 depends on As which we
    # didn't set; normalise to the user's sigma8 by top-hat filtering).
    sig8_camb = results.get_sigma8_0()
    return kh, z_out, pk_lin, float(sig8_camb)


class PkGrid:
    """Linear matter power spectrum P(k, z) [ (Mpc/h)^3 ].

    Uses CAMB once, then provides vectorised lookup via log-log bilinear
    interpolation.
    """

    def __init__(self, cosmo: Cosmology, z_max=2.0, nz=32,
                 kmin=1e-4, kmax=50.0, nk=400):
        self.cosmo = cosmo
        self.z_max = z_max

        kh, zs, pk_lin, sig8_camb = _camb_pk_grid(
            cosmo.key, z_max, nz, kmin, kmax, nk
        )
        # Rescale for user-specified sigma8 (pure normalisation; ns unchanged).
        rescale = (cosmo.sigma8 / sig8_camb) ** 2
        self.k = kh                          # (nk,)
        self.z = zs                          # (nz,)
        self.P = pk_lin * rescale            # (nz, nk)
        self._logk = np.log(self.k)
        self._logP = np.log(self.P)

    def __call__(self, k, z=0.0):
        """P(k, z) via bilinear interpolation in (log k, z)."""
        k = np.atleast_1d(k).astype(float)
        z = np.atleast_1d(z).astype(float)
        logk = np.log(k)

        # Clip to grid bounds to avoid runaway extrapolation.
        logk = np.clip(logk, self._logk[0], self._logk[-1])
        z_clip = np.clip(z, self.z[0], self.z[-1])

        # Bilinear: interp in logk for each z column, then interp in z.
        # Vectorised over the (nz_out, nk_out) output shape.
        # Simpler: use scipy.interpolate.RectBivariateSpline.
        if not hasattr(self, "_spl"):
            from scipy.interpolate import RectBivariateSpline
            self._spl = RectBivariateSpline(self.z, self._logk, self._logP,
                                            kx=1, ky=1)
        logP = self._spl(z_clip, logk)
        out = np.exp(logP)
        if out.shape == (1, 1):
            return out[0, 0]
        if out.shape[0] == 1:
            return out[0]
        if out.shape[1] == 1:
            return out[:, 0]
        return out
