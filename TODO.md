# TODO — validation against Matteo's code (MCostanzi/SelectionBias)

Status as of 2026-08-21, branch `fix/signed-deltasigma-kernel`.
Context: `docs/costanzi_notebook_diff.md` (line-by-line diff),
`validations/costanzi_*.py` (verbatim port, attribution ladder, b_sel
chain), closure mode (`SigmaPrj(closure=True, tmap="comoving")` with
`NFWMiscentered(kind="m200m")`) satisfies both sum rules:
cl/(rho_m b_sel xi) = 0.99 and rnd/(rho_m mean sheet) = 1.0004.

## Phase 1 — Sigma_prj machinery (no external inputs)

- [ ] Convention-matched agreement run: `costanzi_reference.py`
      (verbatim notebook recipe) vs our `SigmaPrj` with matching flags
      and the SAME b_sel, across the 12 DES-Y1-like (lambda, z) bins.
      Target <= 3% (quadrature-level) per R.  Any residual = port bug;
      iterate until flat.
- [ ] Sum-rule verdict to Matteo: his kernel gives
      cl/(rho_m b_sel xi) = 1.33 (mass-conserving M200m but
      untruncated within 30 cMpc/h); our closure mode = 0.99.
      One decision needed: adopt the closure (truncated NFW +
      unresolved counter-term) or document the 1.33 as intended.
      This is the single physics divergence left in Sigma_prj.

## Phase 2 — b_sel chain (blocked on Matteo — ask now)

- [ ] Email Matteo for: `inter_boost_bias` / `slope_boost_bias`
      calibration values, the b_sel-grid marginalisation cell
      (`b_sel_lob_theta_grid_inter` construction), and
      `mock_lob_sigma_catalog.fits` (none are in the public repo).
- [ ] With the boost params: rerun `validations/costanzi_bsel.py` —
      test the prediction boost = 1.34 (his plateau 2.86 x boost vs
      our P[X]-operator plateau 3.82).  Match => b_sel validated
      end-to-end; mismatch => reconcile the sigmoid transition vs the
      P[X] operator at theta ~ theta_lob (quantify the effect on
      Sigma_prj — expect < 2%, xi-weighted).

## Phase 3 — observable level

- [ ] DeltaSigma under closure: wire the truncated signed kernel
      (by-parts machinery in `nfw.py` is ready) or take the radial
      excess of closure-Sigma by linearity; verify
      DeltaSigma_cl/TwoHalo -> 1 at R >= 5 cMpc/h.
- [ ] Mock reproduction: with the catalog, reproduce the notebook's
      cells-19/21 ratio panels (Sigma_lob-sel / Sigma_RND, 12 bins)
      with BOTH codes — the actual paper figure, the definitive
      cross-validation.
- [ ] Regenerate the C1 envelope (B_C1) with the closure pipeline and
      compare against the published 1.08 peak — closes the
      apples-to-oranges caveat of issue #1.

## Phase 4 — production (y3_cluster_cpp)

- [ ] Port closure to C++: truncated kernel (same table machinery),
      unresolved counter-term (~30 lines), signed DeltaSigma table
      (already generated: `/pscratch/sd/j/jesteves/nfw_off_center/`,
      symlinked from the canonical `/global/common/.../nfw_off_center/`
      path — NOTE pscratch purge policy, move it once the des quota is
      cleaned), likelihood reads `shear_prj/cl` (never `vals`).
      Regenerate goldens.
- [ ] End-to-end Buzzard: rerun `validation_1h2h_buzzard.ini` — the
      issue-#1 ratio plot should now sit on the B_C1 envelope.

## Order of operations

Phase 1 and the Phase-2 email in parallel; then Phase 3; then Phase 4.
Only the sum-rule decision and the boost check need Matteo's input.
