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
pytest tests/                                       # ~25 s, 45 tests
pytest tests/test_selection_function.py::TestKi     # single test class
pytest tests/test_sigma_prj.py                      # Sigma_prj regressions
pytest tests/ -k sigma_prj                          # keyword filter
python examples/01_sigma_prj.py                     # end-to-end timing run
python validations/sigma_prj_diagnostics.py         # scipy.quad regression
python validations/sigma_prj_theta_lambda.py        # per-bin θ_λ table, etc.
```

Set `RICHNESS_SELECTION_NFW_DIR` (env var) to the NFW lookup directory
off-NERSC; `tests/conftest.py` and `validations/_common.py` both honour
it and tests skip gracefully if the table is missing.

There is no lint/format config — the repo is pure numerics.

## Architecture that spans files

**Grids and config (`config.py`)** — `GridConfig` (`Nz=80`, `Nth=10`, `NM=24`) is wired in for sub-0.01% precision on `(P[1], I_1, I_2)` at `(lob=20, zob=0.5)`. `DEFAULT_GRID` is shared by `SelBias` and `SigmaPrj`; do not raise `Nth` without understanding the split-at-exclusion trick (see below). `NFW_TABLE_DIR` points at a NERSC path — overriding is required off-NERSC. `R_MAX_CMPCH = 30.0` sets the θ upper bound for `SigmaPrj` (matches Matteo's `twoD_prj_NFW` truncation).

**Shared substrate** — `Cosmology` → `PkGrid` (CAMB) → `SigmaM` → `HMF` (Tinker 2008) + `Bias` (Tinker 2010) + `MOR` (HOD default) form the dependency chain threaded through both pipelines via `conftest.py` / notebook setup cells. `XiNL` is lazy-built from `PkGrid` (halofit).

**`SelBias._P_operator` (the central routine)** — computes `(P[1], I1, I2)` via nested quadratures with two critical numerical tricks documented in the docstring and paper:
- **z-axis split by physics** (`_z_grid`): ring `|z-z_ob| < R_excl/dchi/dz` uses GL-in-z; fg/bg halves use GL in `u = ln|Δχ|` so nodes cluster near `z_ob` where `ξ_NL` spikes.
- **theta-axis split at exclusion** (inside the z-loop): `theta_excl(z)` is the lower limit of the per-z GL theta interval. With the hard `ξ_NL=0` step moved onto the boundary, `Nth=10` converges; the old fixed `(0, 2θ_lob)` grid + mask needs `Nth > 200`.

Cached results keyed by `(lob, zob)` live in `self._cache`; `bias_precompute → bias_from_precomp` separates the heavy `(lob, zob)` work from cheap per-`ltr` assembly.

**`SigmaPrj.__call__` (θ-outer refactor)** — the integrand `[1 + b·b_sel·ξ_NL]·Σ_mis` is split into `rnd` (the `1`) and `cl+LSS` (the `b·b_sel·ξ_NL`). **Default return is the `cl+LSS` piece** — the two-halo correlation excess that matches `ΔΣ_2h` in the lensing literature; `return_decomposition=True` exposes `{rnd, cl, total}`. θ is the outer loop; inside, the z-integrals of `n(M,z)·outer_weight` and `n·b·outer_weight·ξ_NL` contract with a z-independent `Σ_mis(M, R | R_mis=θ·D_A)` stack. θ-grid is log-GL on segments split at `{θ_excl,o, every requested θ_R, θ_λ, 2θ_λ, θ_max}` — the per-R breakpoint rule is load-bearing (without it, R=3 residual is +1.3 %; with it, +0.01 %). LoS-slab exclusion (`θ > θ_excl(z)`), not 3-D ball. `R_max_cMpch` (default 30) sets `θ_max = R_max/D_A(z_ob)`; see `docs/sigma_prj_refactor.md` for the full recipe.

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
- `notebooks/03_sigma_prj_audit.ipynb` — loads `validations/cache/*` artifacts and renders 9 diagnostic figures (angular scales, Δχ-only error, W(z) symmetry, N_rnd vs N_cl+LSS, Σ_prj decomposition, R_max sensitivity). Read-only; re-run validation scripts to refresh the cache.

## DES Y3 bin constants

`richness_selection.des_y3` exports `Y3_LAM_BINS` (4 richness edges), `Y3_Z_BINS` (3 redshift edges, z≤0.65), `Y3_LAM_MEAN`, `Y3_Z_MEAN`, and `iter_bins()`. Also re-exported from the top-level `richness_selection`.

## Validation scripts and cache

`validations/*.py` each writes one file under `validations/cache/` (CSV or NPZ) consumed by notebook 03. Shared setup (NFW table dir via `RICHNESS_SELECTION_NFW_DIR`, stack builder) in `validations/_common.py`. Do not commit large artifacts by hand — re-run the scripts.

## Numerical recipe

`docs/sigma_prj_refactor.md` is the authoritative step-by-step recipe for computing `⟨Σ^prj⟩`: integration limits, exclusion convention, split-at-breakpoints θ-grid, θ-outer order, pseudocode, tolerances, and regression checks. Read this before modifying `sigma_prj.py`.
