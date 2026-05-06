# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package does

`richness_selection` implements two Costanzi pipelines for the DES optical cluster cosmology stack, backed by CAMB/astropy and pre-tabulated NFW data:

1. **`SelBias` + `SigmaPrj`** — Costanzi 2026 eqs. 3–13. `SelBias` builds the scale-dependent bias `b_sel(theta | lob, zob)` from a P[X] operator over `(z, M, ltr, theta)`; `SigmaPrj` integrates that bias against a miscentered NFW to produce `⟨Σ^prj(R | λ^ob, z^ob)⟩`. Paper-authoritative doc: `docs/richness_selection.tex`.
2. **`selection_function/`** subpackage — Costanzi 2019/2021 closed-form `K_i`, `S_i`, `S_threshold`, and number counts `N_ij` over a DES-Y3-like bin grid. Paper-authoritative doc: `docs/richness_selection_function.tex`.

The two pipelines share cosmology, the HMF, bias, and MOR layers but are otherwise independent — changes to `sel_bias.py` do not affect `selection_function/`.

## Commands

```bash
pip install -e .                                    # editable install
pytest tests/                                       # ~30 s, 41 tests
pytest tests/test_selection_function.py::TestKi     # single test class
pytest tests/ -k sigma_prj                          # keyword filter
python examples/01_sigma_prj.py                     # end-to-end timing run
python validations/selection_function_quad.py       # S_i vs scipy.nquad
python validations/quad_matched_inner.py            # b_sel vs scipy.quad
```

There is no lint/format config — the repo is pure numerics.

## Architecture that spans files

**Grids and config (`config.py`)** — `GridConfig` (`Nz=80`, `Nth=10`, `NM=24`) is wired in for sub-0.01% precision on `(P[1], I_1, I_2)` at `(lob=20, zob=0.5)`. `DEFAULT_GRID` is shared by `SelBias` and `SigmaPrj`; do not raise `Nth` without understanding the split-at-exclusion trick (see below). `NFW_TABLE_DIR` points at a NERSC path — overriding is required off-NERSC.

**Shared substrate** — `Cosmology` → `PkGrid` (CAMB) → `SigmaM` → `HMF` (Tinker 2008) + `Bias` (Tinker 2010) + `MOR` (HOD default) form the dependency chain threaded through both pipelines via `conftest.py` / notebook setup cells. `XiNL` is lazy-built from `PkGrid` (halofit).

**`SelBias._P_operator` (the central routine)** — computes `(P[1], I1, I2)` via nested quadratures with two critical numerical tricks documented in the docstring and paper:
- **z-axis split by physics** (`_z_grid`): ring `|z-z_ob| < R_excl/dchi/dz` uses GL-in-z; fg/bg halves use GL in `u = ln|Δχ|` so nodes cluster near `z_ob` where `ξ_NL` spikes.
- **theta-axis split at exclusion** (inside the z-loop): `theta_excl(z)` is the lower limit of the per-z GL theta interval. With the hard `ξ_NL=0` step moved onto the boundary, `Nth=10` converges; the old fixed `(0, 2θ_lob)` grid + mask needs `Nth > 200`.

Cached results keyed by `(lob, zob)` live in `self._cache`; `bias_precompute → bias_from_precomp` separates the heavy `(lob, zob)` work from cheap per-`ltr` assembly.

**`SigmaPrj.__call__`** — reuses `SelBias._z_grid` (same z-ring decomposition), then for each R uses its own `_theta_grid_for_R` that **splits at `θ_R = R/D_A(z_ob)`** (the NFW `Σ_mis` peak). A one-shot `b_sel_marginalised(theta)` cubic-spline cache avoids ~100 ms per-node marginalisation calls inside the inner loop.

**`selection_function/` (fully tensorised)** — `number_counts.N_ij` is a 2-D GL quadrature over `(ln M, z)`; `S_i`/`S_threshold` use the closed-form EMG kernel `K_i` from `kernels.py` (erfcx-safe). `survey.py` provides `Ω(z)` models ported from `y3_cluster_cpp` — pass `solid_angle=omega_z_des` to `N_ij` to reproduce realistic `N(z)` rolloff.

**NFW convention pitfall (`nfw.py`)** — the bundled lookup table stores `⟨Σ_NFW⟩_phi / (2π R_s ρ_s)`, which is **half** Costanzi 2026 eq. 14's definition. `sigma_grid` applies the factor of 2 so callers see paper-convention values. Do not add another factor of 2 in `SigmaPrj`.

## External data

`data/` ships the inputs the package reads at runtime:
- `nfw_off_center/` — the three `table_1000_1e-03_5e+03_*.txt` NFW lookup files.
- `omega_z_sdss.txt`, `z_kernel_5perc_ext_z01.txt`, `prj_params_DESY3_lss_lin_dep_getdist_v1.txt`.

If you edit `NFW_TABLE_DIR` in `config.py` or add a new table-path indirection, verify both `NFWMiscentered` (and anything else reading `data/`) resolve.

## Notebooks

- `notebooks/00_selection_function.ipynb` — `K_i`, `S_i`, `S_threshold`, `Ω(z)`, `N_ij` matrix.
- `notebooks/01_selection_bias.ipynb` — reproduces Costanzi 2026 Fig. 2 (`b_sel(θ)`).
- `notebooks/02_sigma_prj.ipynb` — `⟨Σ^prj(R | λ^ob, z^ob)⟩`.
