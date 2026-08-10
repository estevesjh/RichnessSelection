# Port frozen-physics b_sel/ΔΣ_prj recipe into the cosmosis pipeline — final report

**Status: done.** Production path chosen, implemented, benchmarked, wired into all three
pipeline `.ini` files, cross-validated against an independent numerical method, and the
Ω(z) survey-area question has been given a proper home in `richness_selection` (not left
as an open deferral). This report is the record of what was built, measured, and decided —
see `~/.claude/plans/merry-splashing-yao.md` for the original plan this executes.

## TL;DR

- **b_sel**: prototyped a Python replacement (`bsel_richness.py`, calling
  `richness_selection.FrozenSelBias` directly) and validated it against the current
  `b_sel_marg`+`bsel.py` combo — but **not switched into production**. The existing C++
  `b_sel_marg`+`bsel.py` combo stays as-is: it's faster (~0.086s vs `bsel_richness.py`'s
  ~0.24–0.27s), and there was no reason to trade that away. `bsel_richness.py` is kept on
  disk as a working, validated reference (not committed to `y3_cluster_cpp`, not wired
  into any `.ini`), in case a future need for the Python-side b_sel path arises.
- **ΔΣ_prj / γ_t^prj**: benchmarked 4 candidate replacements for the C++ `shear_prj`
  (`ShearPrjEvaluator`) module inside real CosmoSIS pipeline evaluations. **Option C
  (`ShearPrjFrozenPhysics.cc`) wins** — the only candidate faster than baseline, at
  **~3.2× speedup with <0.2% precision loss** (tuned resolution). Now wired into
  `mock_mcmc_cp_camb.ini`, `mock_mcmc_buzzard.ini`, `generate_mock_dv.ini`, replacing
  only `shear_prj` — `b_sel_marg`+`bsel` are untouched.
- Two significant bugs were found and fixed along the way (mass-Jacobian double-count,
  missing θ-grid breakpoint term).
- **Ω(z) ported into `richness_selection`** as a `SurveyArea` dataclass (unity / constant /
  polynomial, reusing the already-ported `omega_z_des`/`omega_z_sdss` polynomials) and wired
  into `SigmaPrj`/`DeltaSigmaPrj`'s z-integral, defaulting to unity (no behaviour change).
- **Option C's exact production configuration cross-validated against Option E running
  Vegas** (a structurally independent, stochastic Monte-Carlo integrator, not just a second
  Cuhre run) — agree to **max 0.49%, median 0.15%**, both independently within ~1% of
  baseline. This is the strongest confidence check run in this whole port.

## Part I — b_sel prototype (`bsel_richness.py`) — not switched into production

Prototyped a single Python module calling `richness_selection.FrozenSelBias` directly
(thin adapter classes `_HMFFromBlock`/`_BiasFromBlock`/`_XiNLFromBlock` wrap
already-tabulated datablock arrays, no fresh CAMB/halofit calls needed), meant to replace
`b_sel_marg` (C++) + `bsel.py` (Python, duplicate physics).

- Validated against the old `b_sel_marg`+`bsel` combo to the target tolerance.
- **Performance side-effect found and fixed** (useful regardless of the production
  decision below): `richness_selection.plob_ltr.P_lob_given_ltr` had an unvectorized
  Python loop over 128 GL nodes; vectorizing it gave an 11× speedup on that piece
  (7.42ms → 0.68ms), ~28% end-to-end on `bsel_richness.execute()` (336ms → 241ms for 12
  bins). Also fixed 7 `np.trapezoid` → `np.trapz` call sites (NumPy 2.0-only API;
  pipeline env has NumPy 1.26.4).
- **Decision: kept the existing C++ `b_sel_marg`+`bsel.py` in production.**
  `bsel_richness.py` costs ~0.24–0.27s/sample vs the existing combo's ~0.086s — slower,
  and with no other benefit strong enough to justify the trade. `bsel_richness.py` stays
  on disk as a validated reference (not committed, not wired into any `.ini`).
- Remaining `area_overlap` bottleneck (52% of `bsel_richness.execute()`) profiled and
  partially optimized; full writeup in `PERFORMANCE_bsel_richness_area_overlap.md`
  (deferred, further work not required for this port).

## Part III — ΔΣ_prj/γ_t^prj benchmarking matrix

All configs measured inside real CosmoSIS `sampler=test` evaluations, same fiducial point,
identical 12-bin×10-R wall grid, against baseline **Option A** (current production:
`b_sel_marg`+`bsel`+`shear_prj`/`ShearPrjEvaluator`, `shear_prj` module cost ≈ 0.267s).
Decision rule: eliminate on precision, then prefer the simplest survivor clearing **≥3×
speedup**.

### Option B — fully Python (`shear_prj_richness.py`, `FrozenDeltaSigmaPrj`)

- 0.588s/bin vs baseline's 0.267s → **~2.2× slower**. Precision fine (frozen-physics
  approximation validated to ~0.7–1.8% against baseline elsewhere in this repo).
- **Disqualified on speed.**

### Option D — existing `ShearPrjGsl.cc`, unmodified

- Adaptive GSL z-integration, no frozen-physics reduction (still full O(N_θ·N_z·N_M)).
- **~17× slower** than baseline. Confirms the plan's own hint that this diagnostic
  backend was never intended as a production path.
- **Disqualified on speed.**

### Option E — `ShearPrjFrozenCuhre.cc` (continuous 2-D Cuhre/Vegas integral)

Same frozen-physics reduction as Option C (exact rnd-channel z-hoist, `r_s(M)`-anchored
`a_b(z)` cl-channel drift), but the (θ, lnM) assembly is a genuine adaptive 2-D integral
over `Interp1D`-tabulated functions, instead of an explicit grid.

**Three real bugs found and fixed during validation:**

1. **Mass-Jacobian double-count** (10¹³–10¹⁴× magnitude error). `HMF_t::operator()`
   already returns `dn/dlnM` directly (confirmed from `hmf_t.hh`), not `dn/dM` — an
   extra `M = exp(lnM)` factor in the integrand and in the `anchor_M` weight was a
   genuine bug, not a convention choice. Fixed by removing it (matches production's own
   `wrnd_M[iM] += common_z*hmf_row[iM]*lnm_w_[iM]`, no `M` factor).
2. **θ lower-bound missing a term.** `theta_lo` must be
   `max(1e-8, 0.1·min(theta_excl_o, theta_R_min, theta_lam))` — I had omitted
   `theta_R_min = min(R)/D_A_o` from the `min()`, truncating away the smallest-R
   breakpoint's support region (this repo's R-grids go down to 0.2 cMpc/h, well below
   the λ-scale for most bins). Confirmed by reading `sp_detail::build_theta_grid` in
   `sigma_prj_t.hh` — production's own formula includes this term.
3. **Not a bug, but worth recording**: switching the Cuhre integration variable from
   linear θ to `u = ln(θ)` (matching production's log-GL convention) made negligible
   difference on its own — the θ-domain dynamic range (θ_lo ~1e-8 to θ_hi ~0.05–0.1,
   5–6 decades) was a real risk factor for adaptive-cubature feature-hiding, but bug #2
   turned out to be the dominant effect, not this.

**Cross-validation, not a bug**: after the two fixes, Option E still differed from
baseline by ~40–55% median. Root cause: `common_z` included an Ω(z) survey-solid-angle
factor (matching `ShearPrjGsl`'s convention per the plan's earlier decision), while
baseline `ShearPrjEvaluator` and the Python `FrozenDeltaSigmaPrj` reference both omit it
entirely. With `include_omega_z=0`, Option E matches baseline to **max 1.1%, median
0.28%** — in the same tolerance band as Option B. Vegas vs Cuhre agreed to <0.5%,
ruling out the integration algorithm itself as a source of error at any point.

- Timing: Cuhre 2.1–2.8s (~8–10× slower), Vegas 6.4s (~24× slower) than baseline.
- **Disqualified on speed**, despite the implementation now being verified correct.

### Option C — `ShearPrjFrozenPhysics.cc` (explicit fixed grid, no Cuhre) — **winner**

Same frozen-physics reduction as Option E, but the (θ, lnM) assembly is an explicit
`N_theta × N_M` grid + dot product using `sp_detail::build_theta_grid` — the *same*
log-GL, per-R-breakpoint θ grid production's own `ShearPrjEvaluator` uses — instead of a
continuous integral. `DSigma_mis(R | θ, M)` is cached once per (slice, R) in
`set_sample()`, mirroring `ShearPrjCore`'s own caching, so `evaluate()` is a pure dot
product with no NFW lookup inside it.

**Cross-validated against Option E** (independent implementation, same algorithm, Ω(z) on
for both): agree to **max 0.41%, median 0.11%**. Combined with Option E's own
Ω(z)-off-vs-baseline validation, this confirms Option C's implementation is correct.

**Resolution sweep** (`n_lnm` = mass-grid GL nodes, `n_per_seg` = θ-grid GL nodes per
log-GL segment):

| Config | `shear_prj_frozen_physics` time | Speedup vs baseline (0.267s) | Diff vs full-res |
|---|---|---|---|
| n_lnm=24, n_per_seg=10 (full) | 0.124s | 2.15× | — (reference) |
| n_lnm=16, n_per_seg=10 | 0.083s | **3.2×** | 0.20% max / 0.07% median |
| n_lnm=24, n_per_seg=6 | 0.079s | 3.4× | 1.47% max / 0.18% median |
| n_lnm=16, n_per_seg=6 | 0.055s | 4.85× | 1.41% max / 0.25% median |

`n_per_seg` (θ-grid density) is the more precision-sensitive knob; `n_lnm` (mass-grid
density) coarsens cheaply. **`n_lnm=16, n_per_seg=10` chosen for production**: clears the
≥3× bar with a comfortable margin while staying inside the ~0.2% tolerance band used
throughout this plan.

## The Ω(z) question

Production's own `sigma_prj_t.hh` (right where `wrnd_M`/`wcl_M` are built) has an explicit
comment: *"Omega(z) is the SDSS/DES effective survey solid angle... For the Sigma_prj /
DSigma_prj surface densities it cancels between the numerator and the normalisation, so
it must NOT appear here."* This directly contradicts the earlier plan decision to mirror
`ShearPrjGsl`'s convention (which does include Ω(z)) for any new C++ ΔΣ_prj module.

Empirical confirmation: with `shear_prj_frozen_physics.include_omega_z=1`, the fiducial
likelihood closure check gives **Likelihood = -151.68** (should be ~0, since the mock
data vector was generated at this exact point). With `include_omega_z=0`, closure holds
at **Likelihood = -0.004**.

**Decision for this port**: production wiring uses `include_omega_z=0` — the only
convention consistent with the existing mock data vector today. The question of whether
Ω(z) *should* eventually be included (and if so, via what model) is real, and rather than
leave it as an open deferral, `richness_selection` now has a proper home for it:

### `SurveyArea` (new: `src/richness_selection/survey_area.py`)

A frozen dataclass, `SurveyArea(kind, value, model)`, with three variants matching
`y3_cluster_cpp/src/models/omega_z_*.hh`:

- `kind="unity"` (default) — Ω(z)=1, i.e. no survey-area weighting. Reproduces
  `SigmaPrj`/`DeltaSigmaPrj`'s historical behaviour bit-for-bit — **zero change** for
  every existing caller.
- `kind="constant"` — Ω(z) = `value` (steradians), independent of z.
- `kind="polynomial"` — reuses `selection_function.survey.omega_z_des`/`omega_z_sdss`
  (already-ported piecewise polynomial fits — no coefficients duplicated) via a `model=`
  selector. Note: the existing C++ class is misleadingly named `OMEGA_Z_DES` but its
  coefficients are SDSS-era fits; the Python port already carries both under their
  correct names.

Wired into `SigmaPrj.__init__`/`DeltaSigmaPrj.__init__` as a `survey_area: SurveyArea =
SurveyArea()` parameter, multiplied into `_build_zM_context`'s `outer_weight` (shared by
both the rnd-channel hoist and the cl-channel Ψ(θ) accumulation). `FrozenDeltaSigmaPrj`
inherits this automatically (no changes needed there).

New tests (`tests/test_survey_area.py`, 8 cases): default-vs-explicit-unity bit-identical,
`kind="constant"` rescales `rnd`/`cl`/`total` by exactly the same factor (a clean,
analytically-predictable check — a constant Ω(z) multiplies the shared `outer_weight`
before the two channels diverge), `kind="polynomial"` matches
`selection_function.survey.omega_z_des`/`omega_z_sdss` directly and produces finite,
non-trivial output that shifts the rnd/cl ratio (unlike the constant case, since Ω(z)
now reweights different z-ranges differently for each channel), and invalid
`kind`/`model` raise `ValueError`. Full suite re-run after wiring: 80 passed, 3 failed —
all three failures confirmed pre-existing via `git stash` (the already-documented
`test_reference_point` issue, plus two `test_total_vs_scipy_quad` cases that also
reproduce identically on the untouched code — unrelated to this change).

This doesn't change the C++ production decision (still `include_omega_z=0`, still the
only convention consistent with the current mock data vector) — it means the *next* time
this question comes up (e.g. a regenerated mock data vector with a deliberate survey-area
convention), both the Python and C++ sides have a real, tested, pluggable implementation
to reconcile against, instead of a hardcoded absence on one side and an inline `bool`
toggle on the other.

## Production wiring

All three `.ini` files updated identically (`des-cluster-nersc/cosmosis-models/`) —
**`shear_prj` only; `[bsel]`/`[b_sel_marg]` are completely untouched**:

- New `[shear_prj_frozen_physics]` section added (wall-grid axes — `lambda_bin`, `zo_low`,
  `zo_high`, `radii` — copied verbatim from each file's own `[shear_prj]` section so the
  two are point-for-point comparable): `n_lnm=16`, `n_per_seg=10`, `n_zring=20`,
  `n_zouter=20`, `include_omega_z=0`.
- `[pipeline] modules=`: `shear_prj` → `shear_prj_frozen_physics`. `b_sel_marg bsel` is
  unchanged.
- **Old `[shear_prj]` section left untouched** (not deleted) for rollback/reference, per
  the plan's own guidance.
- `ShearPrjFrozenPhysics` publishes `dsigma_prj_frozen_physics`/`shear_prj_frozen_physics`
  *and* aliases the same `gt_{total,rnd,cl}` values to `shear_prj/{vals,rnd,cl}` — drop-in
  compatible with `likelihood_cp.py` and `generate_mock_dv.py`, which both read
  `shear_prj/vals` by name, with zero changes needed to either.

### End-to-end validation

- `mock_mcmc_cp_camb.ini`, `sampler=test`, fiducial point (with `b_sel_marg`+`bsel.py`
  restored + `shear_prj_frozen_physics`): runs clean, **Likelihood = -0.003** (baseline
  gives exactly 0.0 — the residual is consistent with the frozen-physics approximation's
  own budget, not a new bug).
- `generate_mock_dv.ini`: runs clean, internal self-consistency check reports
  `logL = -0.0 (OK)`.
- `mock_mcmc_buzzard.ini`: fails with `ValueError: likelihood_cp: data_Shear has size 180,
  expected 120` — **confirmed pre-existing** via `git stash` (reproduces identically on
  the untouched, previously-committed file). Unrelated to this port; a stale/mismatched
  data vector predating this session.

## Cross-validation: Option C (production config) vs. Option E running Vegas

The strongest confidence check available: does the *exact* production configuration
(Option C, `n_lnm=16, n_per_seg=10, include_omega_z=0`) agree with a completely
independent numerical method — not just a second Cuhre run (deterministic cubature,
structurally similar to Option C's own fixed-grid dot product), but **Vegas**, a
stochastic Monte-Carlo integrator with no shared machinery at all?

| Comparison | total max/median | rnd max/median | cl max/median |
|---|---|---|---|
| Option C vs. Option E (Vegas) | 0.49% / 0.15% | 0.38% / 0.11% | 0.64% / 0.22% |
| Option C vs. baseline | 0.96% / 0.31% | — | — |
| Option E (Vegas) vs. baseline | 1.24% / 0.30% | — | — |

Three independent implementations (production's fixed-GL `ShearPrjEvaluator`, Option C's
fixed-grid frozen reduction, Option E's Vegas-integrated frozen reduction) agree to
within ~1% of each other, all inside the tolerance band already established for the
frozen-physics approximation elsewhere in this repo (~0.7–1.8% for Option B). This is
about as strong a "things are working" signal as this kind of numerical port can get
without a from-scratch independent derivation.

## Remaining deferred work (tracked as separate tasks)

1. **`ISSUE_p_operator_cpp_vs_python_discrepancy.md`** — ~0.2–0.8% P1/I1/I2 discrepancy
   between C++ `b_sel_marg`'s `P_operator` and Python `SelBias._P_operator`, both at their
   own continuum limits. Root cause not yet found; repro instructions included.
2. **`PERFORMANCE_bsel_richness_area_overlap.md`** — `area_overlap` is 52% of
   `bsel_richness.execute()`'s runtime; one redundant-copy fix already applied (~7% win),
   further gains would need resolution/algorithm changes to the exclusion-zone geometry.

(The Ω(z) `SurveyArea` port and the Option C/Vegas cross-validation, both originally
listed here as deferred, are now done — see above.)

## Files touched this session

- New: `y3_buzzard/bsel_richness.py`, `y3_buzzard/shear_prj_richness.py` (Option B, kept
  for reference/rollback), `src/models/sigma_prj_frozen_interp_t.hh` (Option E),
  `src/models/sigma_prj_frozen_t.hh` (Option C, production), `src/modules/sigma_prj_cpu/
  ShearPrjFrozenCuhre.cc`, `src/modules/sigma_prj_cpu/ShearPrjFrozenPhysics.cc`,
  `src/richness_selection/survey_area.py` (`SurveyArea`), `tests/test_survey_area.py`.
- Modified: `src/models/nfw_dsigma_mis.hh` (additive `r_s()` accessor),
  `src/modules/sigma_prj_cpu/CMakeLists.txt` (two new targets),
  `src/richness_selection/{sel_bias,plob_ltr,mor}.py` +
  `tests/{test_integrals,test_smoke}.py` (NumPy 2.0 API fixes, ltr-marginalisation
  vectorization), `src/richness_selection/geometry.py` (`area_overlap` copy removal),
  `src/richness_selection/{sigma_prj,delta_sigma_prj,__init__}.py` (`SurveyArea` wiring),
  `des-cluster-nersc/cosmosis-models/{mock_mcmc_cp_camb,mock_mcmc_buzzard,
  generate_mock_dv}.ini` (production wiring).
- All new/modified `y3_cluster_cpp` files synced across both checkouts
  (`/pscratch/sd/j/jesteves/y3_cluster_cpp` — production — and
  `/pscratch/sd/j/jesteves/github/y3_cluster_cpp` — dev clone).
