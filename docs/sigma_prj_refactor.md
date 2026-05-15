# A numerical recipe for ⟨Σ^prj(R | λ^ob, z^ob)⟩

This document specifies how to compute the two-halo projection-effect
surface-density profile `⟨Σ^prj(R | λ^ob, z^ob)⟩` around a
richness-selected cluster, in the Costanzi 2026 selection-bias model.
It is written as a step-by-step recipe that matches the implementation
in `src/richness_selection/sigma_prj.py` and the validation artifacts
in `validations/cache/`.

Throughout: distances in cMpc/h, masses in M☉/h, angles in radians.

> **Companion note.** For the lensing excess `⟨ΔΣ^prj⟩` — derivation,
> linearity-in-`R` argument, and the adaptive `θ_max = 3·max(R)/χ(z_ob)`
> recipe that replaces the legacy `R_max=30 cMpc/h` truncation — see
> `docs/delta_sigma_prj_derivation.tex`. The shared pipeline (θ-grid,
> z-grid, exclusion, b_sel evaluation) described below is reused
> unchanged; only the NFW kernel is swapped
> (`NFWMiscentered._spl` → `_dsig_spl`).


## 1. What is being computed

The full two-halo model (Costanzi 2026 Eq. 13) is

```
⟨Σ^prj(R | λ^ob, z^ob)⟩ = 2π ∫ dθ sin θ
                          · ∫ dz (dV/dzdΩ)(z) w_z(z, z^ob)
                          · ∫ dM n(M, z) [ 1 + b(M, z) · b_sel(θ; λ^ob, z^ob)
                                             · ξ_NL(|Δr|, z^ob) ]
                          · Σ_mis(R | M, z, R_mis = θ · D_A(z^ob))       (Eq. 13)
```

The `1` term is the *mean cosmological* projected surface density in
the photo-z window (call it `Σ_rnd`); the `b·b_sel·ξ_NL` term is the
*correlation-excess* two-halo contribution `Σ_cl+LSS`. In
`ΔΣ = Σ̄(<R) − Σ(R)` the uniform `Σ_rnd` cancels and is absent from
the standard two-halo lensing observable (Sheldon+ 2009, Zu+ 2014,
Melchior+ 2017). Accordingly:

- `SigmaPrj(R, lob, zob)` returns **`Σ_cl+LSS(R)` by default.**
- `SigmaPrj(R, lob, zob, return_decomposition=True)` returns
  `{'rnd': …, 'cl': …, 'total': rnd + cl}`.


## 2. Ingredients from other subpackages

The integral assumes the following pieces are available (all wired
through `SelBias`):

| symbol | object | notes |
|---|---|---|
| `χ(z)`, `D_A(z)`, `dV/dzdΩ(z)` | `Cosmology` | astropy-backed, cached |
| `n(M, z)` | `HMF` | Tinker 2008, Δ=200m |
| `b(M, z)` | `Bias` | Tinker 2010 |
| `ξ_NL(r, z)` | `XiNL` | halofit P_NL → FFTlog Hankel |
| `w_z(z, z^ob)` | `photoz.w_z` | parabolic, vanishes at ±σ_z |
| `Σ_mis(R | M, z, R_mis)` | `NFWMiscentered` | `y3_cluster_cpp` convention: `c=4`, `r_200c`, `2·r_s·ρ_eff·exp(lnf)·1e-12`, `M⊙/h / pc²` |
| `b_sel(θ; λ^ob, z^ob)` | `SelBias.b_sel_marginalised` | `_P_operator` integrated over λ^tr |
| `b_eff(λ^ob, z^ob)` | `SelBias.b_eff` | halo-bias average at fixed λ^ob |

No changes to any of these are needed for the Σ_prj recipe.


## 3. The |Δr| expression — **use the exact form**

Along the LoS, the 3-D comoving separation between a projected halo
at redshift `z` on the `θ`-ring and the target at `(z^ob, θ=0)` is

```
|Δr|² = χ(z)² + χ(z^ob)² − 2 χ(z) χ(z^ob) cos θ                      (Eq. 2)
```

**Do not approximate** by dropping the `χ χ' θ²` term (the "Δχ-only"
form). `validations/sigma_prj_dchi_only.py` demonstrates that the
resulting error on `∫ dz dV w_z ξ_NL` reaches 35 % at `θ = 0.1 θ_λ`
and > 1000 % at `θ = 2 θ_λ`, because near the ring `|Δχ| → 0` and
the transverse term dominates `|Δr|²`.


## 4. Exclusion — LoS slab, not 3-D ball

Inside the one-halo Lagrangian radius of the cluster, projected halos
are counted out by hand. The correct convention (matching the
`P1`/`I1`/`I2` operators in `SelBias._P_operator`, sel_bias.py:234–254)
is a **line-of-sight slab**:

```
ξ_NL → 0   whenever   θ ≤ θ_excl(z)                                   (Eq. 3)

cos θ_excl(z) = (χ(z)² + χ(z^ob)² − R_excl²) / (2 χ(z) χ(z^ob))
R_excl        = R_λ(λ^ob) (1 + z^ob)
```

Do not mask on `|Δr| < R_excl` (a 3-D ball cut). That subtly
over-excludes near the θ ≫ θ_excl edges and differs from the
authoritative `_P_operator` convention.


## 5. Integration-limit choices

### 5.1 θ upper limit

Set

```
θ_max = R_max_cMpch / D_A(z^ob)                                       (Eq. 4)
```

with **`R_max_cMpch = 30`** by default (in
`config.R_MAX_CMPCH`), matching the `twoD_prj_NFW` hard truncation
in Matteo's SelectionBias notebook (where `Σ_NFW(R > 30) = 0`). The
cl+LSS piece is converged within 1 % at `R_max = 30` because
`ξ_NL(r)` decays physically. The RND piece scales with `R_max` (the
NFW lookup extrapolates to a floor beyond its tabulated range) and
is *not* converged by construction — see Section 8.

### 5.2 z support

The z-integration window is the union of
- the photo-z kernel support `[z_fg_lo, z_bg_hi]` defined by the
  roots of `zmin4zkernel` / `zmax4zkernel` (parabolic `w_z` vanishing
  at `|z − z^ob| = σ_z(z^ob)`), and
- a physics-driven "ring" of half-width
  `Δz_excl = R_excl / (dχ/dz)|_{z^ob}` centred on `z^ob`,

assembled by `SelBias._z_grid` (sel_bias.py:121–171). This is the
**ring + outer-fg + outer-bg** decomposition, identical to the z-grid
used by `_P_operator`; do not replace it by a plain GL grid in z.

### 5.3 z-axis symmetry

`ξ_NL(|Δχ|)` is symmetric around `z^ob`. The asymmetry of the
z-integrand comes from `dV/dzdΩ ∝ χ²/H` and (weakly) `n(M, z)`.
`validations/sigma_prj_z_symmetry.py` quantifies:

- `A(Δz = 0.04)` ≈ +0.39 at `z̄ = 0.275`, falling to 3×10⁻³ at 0.575.
- Replacing `dV · w_z · n̄` by its `z^ob` value (keeping `ξ(|Δχ|)`
  exact) integrates to within **0.3–0.6 %** of the asymmetric answer.
- Integrating only `z ≥ z^ob` and doubling is 0.02–1.3 % off.

This is a viable further speed-up — **not** adopted in the default
pipeline because the full integral is already 0.5 s at 20 R-points.


## 6. The split-at-breakpoints θ-grid

The θ integrand has three scale-separated features that a naive log-GL
grid under-resolves:

1. `θ_excl,o = R_excl / χ(z^ob)` — the exclusion edge.
2. `θ_R = R / D_A(z^ob)` — the `Σ_mis(R, R_mis = θ D_A)` peak.
3. `θ_λ = R_λ(λ^ob)(1 + z^ob) / χ(z^ob)` and `2 θ_λ` — the
   `b_sel(θ)` sigmoid inflection and `f_A` cutoff.

`N_θ · Σ_mis` peaks **exactly at** `θ = θ_R` for every R, with a
half-max width of ~⅔ decade in θ (per
`validations/cache/ntheta_per_R.npz`).

**Recipe.** Build the sorted breakpoint set

```
B = { 0.1·min(θ_excl,o, θ_R_min, θ_λ),
      θ_excl,o,
      θ_R(R_i) for every R_i requested,    ← crucial
      θ_λ,  2·θ_λ,
      θ_max }                                                           (Eq. 5)
```

dedupe close entries, and lay down `n_per_seg = 30` Gauss-Legendre
nodes on each segment in `ln θ`. The resulting total grid has ~120–150
θ-nodes, comparable to the old default but correctly distributed.

This per-R breakpoint rule is what brought the R=3 residual from
+1.3 % to +0.01 % against scipy.quad without increasing the node budget.


## 7. Order of integration: θ outer, (z, M) inner

Fubini-wise, any order computes the same integral. Numerically:

- `Σ_mis(R, R_mis = θ·D_A)` and `b_sel(θ)` depend only on `θ`, not on
  `z` or M. Putting θ outer lets `b_sel(θ)` be evaluated once per
  θ-node via `SelBias.b_sel_marginalised` (one vectorised call).
- `ρ_eff(M)` and `R_s(M)` from the concentration relation (fixed
  `c = 4` per the C++ recipe) do **not** depend on z. So `Σ_mis(M, R, R_mis)`
  is z-independent at fixed θ: build it once per θ and reuse it for
  all z in the inner loop.
- After the z-integral of `outer_weight(z) · n(M, z) · …` and
  `outer_weight(z) · ξ_NL(|Δr|, z^ob) · n(M, z) · b(M, z) · …`,
  the mass contraction against `Σ_mis(M, R)` is a single matmul.

The resulting structure — one Python loop over θ-nodes, everything
else vectorised — runs 20 R-points in ~0.5 s at the reference point
(≈ 5× faster than the previous z-outer implementation).


## 8. Algorithm — summary pseudocode

```
INPUTS   R[Nr], lob, zob
CONFIG   n_per_seg = 30, R_max_cMpch = 30, Nz = 80, NM = 24

# ── one-time context per (lob, zob) ─────────────────────────────────
chi_o, D_A_o  = χ(zob), χ(zob)/(1+zob)
R_excl        = R_λ(lob) (1 + zob)

# z-grid: ring ∪ outer-fg ∪ outer-bg  (SelBias._z_grid)
zs, wzs       = z_ring_and_outer_grids(Nz, zob, chi_o, R_excl)
chi_z[:]      = χ(zs)
outer_weight  = wzs · dV/dzdΩ(zs) · w_z(zs, zob)

# per-z LoS-slab exclusion angle
cos_excl      = (chi_z² + chi_o² − R_excl²) / (2 chi_z chi_o)
theta_excl_z  = arccos(clip(cos_excl, −1, 1))                          # Eq. 3

# M-grid
lnMs, wM      = gl_nodes(ln 1e13, ln 10^15.5, NM)
Ms            = exp(lnMs)
n_mz          = HMF(Ms, zs)                                           # (NM, Nz)
bM_mz         = Bias(Ms, zs)                                          # (NM, Nz)

# per-M NFW scales (z-independent)
rs_M, rho_s   = NFW_rs_and_rhos(Ms)

# cl-piece bias precompute (unchanged from SelBias pipeline)
pre           = SelBias.bias_precompute(lob, zob)

# ── θ-grid (Eq. 5), b_sel at all θ in one shot ──────────────────────
theta_lam     = R_λ(lob) (1 + zob) / chi_o
theta_excl_o  = R_excl / chi_o
theta_R_arr   = R / D_A_o
theta_max     = max(R_max_cMpch / D_A_o, 3 · max(theta_R_arr))
breakpoints   = sorted({ lower, theta_excl_o, *theta_R_arr,
                          theta_lam, 2·theta_lam, theta_max })
thetas, wth   = concat[ log_GL(ln a, ln b, n_per_seg) for (a,b) in breakpoints ]
bsel_vals[:]  = SelBias.b_sel_marginalised(thetas, lob, zob, pre)

# ── θ-outer integration ─────────────────────────────────────────────
out_rnd = 0  ;  out_cl = 0
for i, θ in enumerate(thetas):
    dchi       = √( chi_z² + chi_o² − 2 chi_z chi_o cos θ )           # Eq. 2
    xi_vals    = ξ_NL(dchi, zob)
    xi_vals    = where(θ > theta_excl_z, xi_vals, 0)                  # Eq. 3

    # Σ_mis(M, R | R_mis = θ D_A)  —  24 cheap NFW spline lookups
    Sigma_mis  = NFW_sigma_mis_MR(θ · D_A_o, Ms, rs_M, rho_s, R)       # (NM, Nr)

    # z-contraction → (NM,)  weights
    w_rnd_M    = wM · Ms · (n_mz       · outer_weight[None,:]       ).sum(axis=1)
    w_cl_M     = wM · Ms · (n_mz · bM_mz · (outer_weight · xi_vals)[None,:]).sum(axis=1)

    # M-contraction → (Nr,)
    N_rnd      = w_rnd_M @ Sigma_mis
    N_cl       = w_cl_M  @ Sigma_mis

    prefac     = wth[i] · 2π · sin θ
    out_rnd   += prefac · N_rnd
    out_cl    += prefac · bsel_vals[i] · N_cl

RETURN   out_cl                     # default: two-halo correlation excess
         { rnd: out_rnd, cl: out_cl, total: out_rnd + out_cl }   (optional)
```

### Numerical tolerances

The recipe hits these targets at the reference point
`(λ^ob, z^ob) = (20, 0.5)` on `R ∈ {0.3, 1, 3, 10} cMpc/h`:

- **Accuracy vs scipy.quad (`total`):** ≤ 0.3 % on every R at
  `n_per_seg = 30`, with `Nz = 80`, `NM = 24`.
- **Convergence plateau:** `n_per_seg = 30 → 120` moves the answer by
  < 0.2 % (pinned by `tests/test_sigma_prj.py::test_n_per_seg_convergence`).
- **R_max sensitivity:** cl changes < 1 % going `R_max = 30 → 60`; RND
  changes ~15 %. This is why the default output is cl, not total
  (see Section 1 and `validations/cache/theta_max_compare.csv`).

### Performance

| piece | cost at reference |
|---|---|
| `_build_zM_context` (z + M grids, n, b, θ_excl, r_s) | 1.8 ms |
| `SelBias.bias_precompute` (P1, I1, I2, b_eff) — **outside integral** | 13.3 ms |
| `_theta_grid` (breakpoint construction) | 0.06 ms |
| `b_sel_marginalised` on 390 θ-nodes | 1.4 ms |
| **θ-loop + z,M contraction + NFW spline** (the integral itself) | **~91 ms** |
| 20 R-points, end-to-end | 0.50 s |
| 120 points (12 Y3 bins × 10 R) | ~1.3 s |

The θ-loop time is dominated by the per-M `RectBivariateSpline.ev` call
inside `Sigma_mis_per_theta` (`NM = 24` python calls per θ); a FFTlog
or shared-grid precompute of `Σ_mis` would be the next optimisation if
needed.


## 9. Checks the implementation must pass

Regression tests live in `tests/test_sigma_prj.py`:

1. `test_default_returns_cl` — `SigmaPrj(R, lob, zob)` equals the `cl`
   field of the decomposition, and `rnd + cl == total` bitwise.
2. `test_total_vs_scipy_quad` — `total` at `n_per_seg = 30` agrees
   with the legacy `scipy.quad` reference in
   `validations/sigma_prj_diag_results.md` to ≤ 0.3 % on every R.
3. `test_n_per_seg_convergence` — `n_per_seg = 30` vs 120 < 0.2 %.
4. `test_R_max_cMpch_controls_theta_grid` — `cl` stable and `rnd`
   grows when `R_max_cMpch` doubles; `theta_info['theta_max']` scales
   with the knob.
5. `test_theta_grid_breakpoints_include_each_R` — every requested `R`
   yields a breakpoint at `θ_R = R/D_A(z^ob)` in the θ-grid.

Run `pytest tests/ -q`; expect 45 passed.


## 10. Reproducing the physics studies

Each script writes a CSV / NPZ to `validations/cache/` that the audit
notebook `notebooks/03_sigma_prj_audit.ipynb` consumes.

```bash
python validations/sigma_prj_theta_lambda.py      # per-bin θ_λ, 2 θ_λ, θ_R(R=30)
python validations/sigma_prj_dchi_only.py         # Δχ-only error
python validations/sigma_prj_z_symmetry.py        # ξ(|Δχ|) symmetry + W(z)
python validations/sigma_prj_ntheta_scan.py       # N_θ(θ), N_rnd, N_cl+LSS
python validations/sigma_prj_theta_max.py         # R_max_cMpch sweep
python validations/sigma_prj_rnd_cl_vs_R.py       # Σ_rnd(R), Σ_cl(R)  per-bin
python validations/sigma_prj_diagnostics.py       # scipy.quad regression
python examples/01_sigma_prj.py                   # wall-clock benchmark
```

`validations/_common.py` expects the NFW lookup at
`$RICHNESS_SELECTION_NFW_DIR`, defaulting to
`/Users/esteves/Documents/Projetos/y3_cluster_cpp/data/nfw_off_center`.
