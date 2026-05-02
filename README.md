# RichnessSelection

Fast Python engine for the Costanzi 2026 projection-effects bias and
Sigma_prj profile used in the DES cluster cosmology pipeline.

## Install

```bash
pip install -e /pscratch/sd/j/jesteves/github/RichnessSelection
```

## Example

```python
from richness_selection import Cosmology, PkGrid, HMF, Bias, SigmaPrj

cosmo = Cosmology(Om0=0.286, H0=70.0, ns=0.96, sigma8=0.82)
pk    = PkGrid(cosmo)
hmf   = HMF(cosmo, pk)
bias  = Bias(cosmo, pk)
sp    = SigmaPrj(cosmo, pk, hmf, bias)

import numpy as np
R = np.logspace(-1, 1.3, 20)
prof = sp(R, lob=20.0, zob=0.5)
```

## Package layout

- `cosmology.py`   — astropy wrapper with cached chi / D_A / dV interpolators
- `pk.py`          — CAMB-backed linear P(k,z) with in-process cache
- `sigma_m.py`     — σ(M,z) from top-hat filter on P(k)
- `hmf.py`         — Tinker 2008 halo mass function (Δ=200)
- `bias.py`        — Tinker 2010 halo bias b(M,z)
- `mor.py`         — log-normal MOR (Costanzi 2021 style) + S9 closed form
- `geometry.py`    — R_lambda, theta_lambda, two-disk overlap (S1 closed form)
- `photoz.py`      — parabolic photo-z kernel w_z
- `nfw.py`         — miscentered NFW Sigma lookup table loader
- `gl.py`          — Gauss-Legendre nodes with lru_cache
- `sel_bias.py`    — Costanzi-2026 projection-effects integrands and b_sel pipeline
- `sigma_prj.py`   — Sigma_prj orchestrator
