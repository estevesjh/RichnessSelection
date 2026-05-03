# RichnessSelection

Fast Python engine for two building blocks of the DES optical cluster
cosmology pipeline:

1. **Selection-affected cluster bias and two-halo projected profile**
   (Costanzi 2026): assemble `b_sel(λ^ob, z^ob, θ)` from halo + projection
   ingredients without circularity, then integrate against a miscentered
   NFW to get `⟨Σ^prj(R | λ^ob, z^ob)⟩`. Doc:
   `docs/richness_selection.{tex,pdf}`.
2. **Closed-form richness selection function** (Costanzi 2019b / 2021):
   `K_i(λ^tr, z)`, `S_i(M, z)`, and expected number counts `N_ij` for a
   DES-Y3-like bin grid, with `Ω(z)` survey models ported from
   `y3_cluster_cpp`. Doc: `docs/richness_selection_function.{tex,pdf}`.

## Install

```bash
pip install -e /pscratch/sd/j/jesteves/github/RichnessSelection
```

## Quick examples

### Two-halo projected profile

```python
import numpy as np
from richness_selection import (
    Cosmology, PkGrid, HMF, Bias, MOR, XiNL, NFWMiscentered,
    SelBias, SigmaPrj,
)
from richness_selection.sigma_m import SigmaM

cosmo = Cosmology()
pk    = PkGrid(cosmo)
sm    = SigmaM(pk)
hmf   = HMF(sm)
bias  = Bias(sm)
mor   = MOR()
xi    = XiNL(cosmo)
nfw   = NFWMiscentered(cosmo)

sel = SelBias(cosmo, pk, hmf, bias, mor, xi_nl=xi)
sp  = SigmaPrj(cosmo, sel, nfw)

R    = np.logspace(-1, 1.3, 20)                 # cMpc/h
prof = sp(R, lob=20.0, zob=0.5)
```

### Richness selection function + number counts

```python
from richness_selection import MOR, LogNormalMOR, Cosmology, PkGrid, HMF
from richness_selection.sigma_m import SigmaM
from richness_selection.selection_function import (
    K_i, S_i, S_ij, S_threshold, N_ij,
    omega_z_des, omega_z_sdss,
)

cosmo = Cosmology(); pk = PkGrid(cosmo); sm = SigmaM(pk); hmf = HMF(sm)
mor   = MOR()                                   # HOD; or LogNormalMOR()

# Single-bin richness selection S_i(M, z)
Si = S_i(M=3e14, z=0.4, lam_min=20., lam_max=30., mor=mor)

# Above-threshold selection (complementary CDF)
S  = S_threshold(M=3e14, z=0.4, lam_min=20., mor=mor)

# Expected number counts for one (i, j) bin, with DES Y1 Omega(z)
N = N_ij(bin_lam=(20., 30.), bin_z=(0.3, 0.5),
         cosmo=cosmo, hmf=hmf, mor=mor, sigma_z=0.02,
         solid_angle=omega_z_des)
```

## Package layout

```
richness_selection/
├── cosmology.py          astropy wrapper (chi, D_A, dV, growth)
├── pk.py                 CAMB-backed linear P(k,z), in-process cache
├── sigma_m.py            σ(M,z) top-hat filter on P(k)
├── hmf.py                Tinker 2008 halo mass function (Δ=200m)
├── bias.py               Tinker 2010 peak-height halo bias
├── mor.py                HOD + log-normal mass-observable relations
├── plob_ltr.py           Costanzi EMG kernel P(λ^ob | λ^tr, z)
├── xi_nl.py              nonlinear matter correlation ξ_NL(r, z)
├── nfw.py                miscentered NFW Σ_mis lookup
├── geometry.py           R_λ, θ_λ, two-disk overlap (S1 closed form)
├── photoz.py             parabolic photo-z kernel w_z
├── gl.py                 Gauss-Legendre nodes (lru_cache)
├── config.py             grid-size defaults
├── sel_bias.py           Costanzi-2026 b_sel pipeline (Part I of doc)
├── sigma_prj.py          Σ_prj orchestrator
└── selection_function/   Costanzi 2019/2021 selection function
    ├── kernels.py        K_i (closed form), K_j, F_EMG (erfcx-safe)
    ├── selection.py      S_i, S_ij, S_threshold
    ├── number_counts.py  N_ij (fully tensorised (lnM, z) integral)
    └── survey.py         Omega(z): SDSS / DES polynomial fits,
                          tabulated SDSS lookup, constant baseline
```

## Notebooks and docs

- `notebooks/00_selection_function.ipynb` — closed-form K_i, S_i,
  S_threshold, Omega(z), N_ij matrix, HOD-parameter sensitivity.
- `notebooks/01_selection_bias.ipynb` — reproduces Costanzi 2026 Fig. 2
  (scale-dependent `b_sel(θ)` across Δ^prj).
- `notebooks/02_sigma_prj.ipynb` — two-halo `⟨Σ^prj(R | λ^ob, z^ob)⟩`.
- `docs/richness_selection.{tex,pdf}` — Costanzi-2026 model, numerical
  pitfalls (θ split-at-exclusion, z split-by-physics), API glossary.
- `docs/richness_selection_function.{tex,pdf}` — closed-form derivation
  of K_i from the EMG convolution, N_ij Gauss-Legendre recipe.

## Testing and validation

```bash
pytest tests/                      # 41 tests, ~30 s
python validations/selection_function_quad.py   # S_i vs scipy.nquad (2-D)
python validations/quad_matched_inner.py        # b_sel pipeline vs scipy.quad
```

S_i matches `scipy.integrate.nquad` to 1e-3 where S_i ≥ 1e-4 for both
HOD and log-normal MORs; N_ij is well-defined across the DES Y1 bin
grid with realistic rolloff above z=0.65 when `omega_z_des` is passed.

## Performance

N_ij (DES-Y3-grade: N_M=40, N_z=28, N_q=32): **~10 ms per (i, j) cell**,
~0.5 s for a 4×4 matrix. `SelBias.bias_precompute` (Costanzi b_sel
pipeline): **~26 ms per (λ^ob, z^ob)**.
