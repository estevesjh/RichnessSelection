# Line-by-line diff: MCostanzi/SelectionBias notebook vs this package

Reference: `Analytical modeling optical selection effects on cluster density
profile.ipynb` (MCostanzi/SelectionBias @ main, fetched 2026-08-21).
Cell numbers refer to the notebook; our side refers to `src/richness_selection`.

## Verdict summary

The notebook is the **Sigma-only** reference (no DeltaSigma anywhere).  Its
`Sigma_prj_lobsel_CIRC` (cell 15) computes the same Eq.-13 observable as our
`SigmaPrj`, but with **seven convention differences**, several of which we
flagged independently as bugs on our side.  The notebook's NFW kernel is
mass-conserving and its transverse map is comoving — both points where our
port diverged.

## The Sigma_prj master integral (cell 15) — the seven differences

| # | item | notebook | ours (`sigma_prj.py`) | impact |
|---|------|----------|----------------------|--------|
| 1 | NFW mass definition | `twoD_prj_NFW`: **M200m**, `r200m = (3M/(4pi 200 rho_m))^(1/3)`, `rho_s = rho_m * delta_char` -> reconstructed mass = **M exactly** (mass-conserving) | `nfw.py`: r200**c** via rho_crit, `rho_eff = delta_c rho_crit Omega_m` -> reconstructed mass = **Omega_m * M** | ours deposits 0.286x the neighbour mass |
| 2 | concentration | `c_of_m(M, z) = 10.14 (M/2e12)^-0.081 (1+z)^-1.01` (Duffy-like, mass-dependent) | fixed `c = 4` | shape + r_s scale per M |
| 3 | profile truncation | hard `Sigma = 0` for r > **30 cMpc/h** per halo | untruncated (table to 5000 r_s) | our 2D tail overcount x1.5 |
| 4 | transverse map | **comoving everywhere**: theta grids built as `s / chi_of_z`, kernel argument `theta * chi_of_z(ztr)` | `R_mis = theta * D_A(zob)` (physical) | our (1+z)^2 measure + xi-sampling shift |
| 5 | geometry / quadrature | point-centered: `Sigma_NFW(theta * chi_ztr)` at the measurement point x `int dphi [1 + b(M) b_sel(theta_Rtilde) xi(Rtilde)]` with `Rtilde(phi)` the exact cluster-halo separation | cluster-centered azimuthal average `Sigma_mis(R | theta D_A) * b_sel(theta) * xi(theta)` | equivalent parametrizations of the same 2D integral (both exact) — but see 4 |
| 6 | exclusion | **3D ball**: `dis < R_lambda(lob)(1+zob)` sets `b b_sel xi = -1`, i.e. the **total** integrand `1 + b b xi -> 0` (rnd removed inside too); plus floor `b b xi >= -1` everywhere | LoS slab `theta < theta_excl(z)` zeroes **xi only** (rnd survives inside) | small-R Sigma differs; his floor prevents negative totals |
| 7 | LoS weight | hard slab `prj_depth = +-50 cMpc/h`, **no w_z kernel, no Omega(z)** in Sigma_prj (kernel_z only in the b_sel Delta machinery) | parabolic `w_z` kernel over its support | rnd amplitude differs (prop. to depth); cl barely (xi support ~20) |

Additional Sigma_prj facts:

- **Units**: the docstrings claim R in pMpc/h, but the math is **comoving
  throughout** (R enters `Rtilde` against `theta * chi(zob)`); only the
  amplitude is converted at the end, `Sigma_comoving / a^2` -> Msun h /
  **pMpc^2**.  To compare with our Msun h/pc^2 comoving convention:
  `ours * 1e12 * (1+zob)^2 = his`.
- **Returns the TOTAL** `[1 + b b_sel xi]` — rnd included (finite because of
  the +-50 slab and the 30 cMpc/h profile truncation).  Our default is
  cl-only.
- theta grid: `geomspace(1e-6, 30, 50)/chi(zob)` — 50 log points, trapz.
  Mass grid: `1e13 -> 10^15.5`, 50 points, trapz in lnM (same limits as ours).
  z grid: fg/bg log-|Delta chi| composite, 50+50, trapz (we inherited this
  structure; we use GL).

## Shared substrate — near-identical

- Cosmology: Buzzard (Om=0.286, h=0.7, Ob=0.046, ns=0.96, s8=0.82); flat LCDM;
  `dV/dzdOm = c^3 (int 1/E)^2 / E` comoving == ours.  rho_crit constant differs
  in the 5th digit (2.77514e11 vs our 2.77534e11, 7e-5 relative — negligible).
- P(k): CAMB, 19 z in [0.05, 0.95] (same grid), halofit for xi_NL, LINEAR P for
  sigma(M).  He fixes As = e^3.054/1e10 (calibrated to s8=0.82); we rescale by
  (s8_target/s8_CAMB)^2.  His kmax=200, ours 100.
- xi_NL: direct 50k-point trapz Hankel, r in [1e-4, 250]; ours mcfit FFTlog
  (matched to ~1e-3 by design — `xi_nl.py` docstring).  Both evaluate xi at
  **zob** for all tracers.
- HMF: Tinker08 Delta=200m, hand-coded == our `hmf.py`.  Bias: Tinker10
  Delta=200m == our `bias.py` (verify numerically).
- MOR / P(ltr|M): DES-Y1 NC+3x2pt best fit (Mmin=10^11.3853, alpha=0.85869,
  M1=10^12.69644, sig_intr=0.18095, eps=0.28389, pivot_z0=0.4544), Poisson
  gamma-form `pltr_M` — **check `mor.py` defaults match these numbers**.
- P(lob|ltr,z): same `prj_params_DESY3_lss_lin_dep_getdist_v1.txt` (shipped in
  our `data/`), same mu/sig/tau/fmsk models; NOTE the active `fprj_model` is
  the **sigmoid variant** `b/(1+exp(-lin))^a` (a power-law variant is
  commented out) — check `plob_ltr.py`.
- Photo-z kernel: same `z_kernel_5perc_ext_z01.txt`, same parabolic form,
  same bisect bounds == our `photoz.py`.

## b_sel machinery (cell 16) — different model generation

The notebook's b_sel is NOT our P[X]-operator (paper eqs. 3-13).  It is the
earlier construction:

- `b_sel_lob_ltr_theta(ltr, z, lob, theta)`: **sigmoid interpolation** in theta
  between the intrinsic small-scale value `b_sel_lob_ltr_in` (ratio of
  Delta_prj excess over the two-halo normalization `numerator2`) and the
  large-scale `eff_bias_ltr(ltr) * boost_bias`; transition centered at
  theta_lob/2 with steepness `damping_sigmoid = 2.5`.
- `boost_bias = inter_boost_bias + slope_boost_bias * (Dprj - <Dprj>)/<Dprj>`:
  a linear boost model whose two parameters are **UNDEFINED in the public
  repo** (set in a lost cell / session state; presumably calibrated on the
  mock).
- `b_sel_lob_theta_grid_inter` — the ltr-marginalised interpolator consumed by
  Sigma_prj — is also **undefined in the repo**.
- Aperture-overlap factor `area_overlap(theta, theta_lob, theta_ltr)` and the
  `Omega_halos * f_area` occupation weights appear inside every Delta_prj
  integral.
- The mock catalog `mock_lob_sigma_catalog.fits` (cells 17-21, the validation
  target of the notebook itself) is not in the repo either.

=> An exact end-to-end reproduction requires (a) the boost parameters and the
b_sel grid construction cell, and (b) the mock catalog — ask Matteo.  Until
then the clean validation is **stage (a) below**, which does not need them.

## Validation plan

1. **Stage (a) — isolate the Sigma_prj machinery.**  Port cell 15 verbatim
   into `validations/costanzi_reference.py` and feed BOTH his recipe and our
   `SigmaPrj` the SAME b_sel(theta) (our `marginalised_bias`).  Compare
   Sigma_prj_total(R) at (lob=20, zob=0.5) after the `(1+z)^2 * 1e12` unit
   conversion.  Any residual is entirely items 1-7 of the table.
2. **Stage (b) — reconcile conventions.**  Adopt his kernel choices where they
   are the physical ones (mass-conserving M200m NFW, c(M,z), comoving map,
   truncation, ball exclusion) behind flags in `SigmaPrj`/`nfw.py`; regression
   against stage (a).
3. **Stage (c) — b_sel comparison** once the boost parameters are known.
4. DeltaSigma stays on our side only (notebook has none): signed kernel +
   `TwoHalo` anchor as validated.
